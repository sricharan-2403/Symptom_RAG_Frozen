# Symptom RAG Analyzer — Implementation Learning Track

## 1. Purpose of This Document

This file records what we have learned and implemented so far in the Symptom RAG Analyzer project. It is meant as a developer learning guide for the team, not as end-user documentation.

The goal is simple: capture the real implementation status, explain the ideas behind each component, and keep a clear boundary between:

- what is already built and tested,
- what is only planned,
- and what is still next.

This document should be updated after each major module is completed so the project history remains easy to follow.

---

## 2. Overall RAG Pipeline So Far

The current implemented pipeline is:

Medical PDF
→ DocumentLoader
→ Document
→ TextChunker
→ Chunk
→ BiomedicalEmbeddingModel
→ Embeddings
→ QdrantVectorStore
→ BiomedicalRetriever
→ RetrievedEvidence
→ EvidenceContextBuilder
→ Biomedical Answerer
→ SymptomRAGResult
→ SymptomRAGOutputAdapter
→ finalized agent-to-agent output

This is the real pipeline currently implemented from document ingestion through
the finalized Symptom RAG output adapter.

Important boundary:

- The Answerer performs preliminary reasoning only.
- The output adapter creates the common agent-to-agent dictionary; it does not
  implement Data Fusion or add reasoning.
- The current end-to-end test uses controlled biomedical data. Real biomedical
  knowledge-base ingestion and indexing are the next development phase.

A simple way to think about it:

Medical content enters as PDF pages.
Each page becomes a structured document object.
The text is split into chunks.
Chunks are converted to vectors.
Vectors are stored in Qdrant.
A query is embedded and compared against stored vectors.
The best matching text is returned as evidence.
That evidence is formatted into a readable context block.
The Answerer sends the prepared context to Gemini and validates structured JSON.
The output adapter converts the internal result into the common agent contract.

---

## 3. Project Structure

The relevant project structure is:

```text
src/
  symptom_rag_analyzer/
    __init__.py
    config.py
    main.py
    data/
      __init__.py
      chunker.py
      chunks.py
      documents.py
      loaders.py
    embeddings/
      __init__.py
      model.py
    reasoning/
      context.py
      prompts.py
      answerer.py
    orchestrator.py
    output_adapter.py
    retrieval/
      __init__.py
      indexer.py
      search.py
    utils/
      __init__.py
      logging.py
    vector_store/
      __init__.py
      qdrant_store.py

tests/
  test_biomedical_embedding_model.py
  test_context_builder.py
  test_document_loader.py
  test_rag_retrieval.py
  test_retriever.py
  test_text_chunker.py
```

### Important directories and files

- data/
  - Holds the core data model and ingestion logic for documents and chunks.
  - This is where the project defines what a document is, what a chunk is, and how source material is loaded.

- data/documents.py
  - Defines the Document dataclass used throughout the pipeline.
  - This is the object that represents a loaded document or PDF page.

- data/loaders.py
  - Contains DocumentLoader.
  - This module converts PDF files into Document objects.
  - It does not chunk or embed anything.

- data/chunks.py
  - Defines the Chunk dataclass.
  - A Chunk is a smaller text fragment derived from a Document.

- embeddings/model.py
  - Contains BiomedicalEmbeddingModel.
  - This is the component that turns text into vectors.

- vector_store/qdrant_store.py
  - Contains QdrantVectorStore.
  - This stores embeddings and performs similarity search.

- retrieval/search.py
  - Contains RetrievedEvidence and BiomedicalRetriever.
  - This is the retrieval abstraction used to turn a query into evidence.

- reasoning/context.py
  - Contains EvidenceContextBuilder.
  - This prepares retrieved evidence into a clean, LLM-readable context string.

- reasoning/prompts.py
  - Contains SYSTEM_PROMPT and build_prompt().
  - These prepare evidence-grounded instructions and the user message for the
    reasoning LLM.

- reasoning/answerer.py
  - Contains DiagnosticCandidate, RAGAnswer, and BiomedicalAnswerer.
  - This sends prepared context to Gemini and validates the structured response.

- orchestrator.py
  - Contains the ClinicalTextClarifierOutput model, SymptomRAGResult, and
    SymptomRAGOrchestrator.
  - This coordinates retrieval, context formatting, and preliminary reasoning.

- output_adapter.py
  - Contains SymptomRAGOutputAdapter.
  - This converts an internal result into the common JSON-compatible output.

- tests/
  - Contains the implemented validation tests for document loading, chunking, embeddings, retrieval, and context building.
  - These tests are important because they document what the project actually works on.

---

## 4. Document Model

The project does not pass around plain strings everywhere. Instead, it creates a reusable Document object.

This matters because a PDF page is not just text. It also has identity and provenance.

The actual conceptual model is:

```python
@dataclass
class Document:
    filename: str
    page_number: int | None = None
    text: str = ""
    source_type: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Why we created a Document object instead of returning plain strings

A raw string loses important information:

- which file it came from,
- which page it came from,
- whether the source was PDF/TXT/etc.,
- any metadata that could be useful later.

If we only stored text, future debugging and evidence checking would become much harder.

### Fields in Document

- filename
  - The source file name.
  - Example: pneumonia.pdf or a case note.

- page_number
  - The page within the PDF.
  - This is useful because a document may have multiple pages and the same text may be found at different pages.

- text
  - The extracted page content.

- source_type
  - The type of source, such as PDF or TXT.

- metadata
  - Additional source-level details for future extension or debugging.

### Why provenance matters

Provenance means tracing where a piece of evidence came from.

This is critical for evidence-based retrieval because we must know:

- what the source was,
- which page it came from,
- which document it belongs to,
- and whether the result is trustworthy enough to show the user.

Without provenance, the system could return a relevant sentence without telling you where it came from. That would make evidence weaker and harder to audit.

---

## 5. PDF Loader

The loader is responsible for reading source PDF files and turning each page into a Document object.

### What DocumentLoader does

DocumentLoader is a simple ingestion class in data/loaders.py.

It:

- looks in a folder for PDF files,
- opens each PDF with PyMuPDF/fitz,
- reads each page,
- converts the page text into a Document,
- stores the file name, page number, source type, and metadata.

### How it uses PyMuPDF/fitz

The code calls:

```python
import fitz
```

and then does:

```python
pdf_file = Path(pdf_path)
doc = fitz.open(str(pdf_file))
```

For every page:

```python
page = doc.load_page(page_number)
text = page.get_text("text")
```

This means the loader is a page-level extractor, not a semantic processor.

### Why each PDF page becomes a Document

A PDF page is a natural unit of source content.
If we stored the whole PDFs as one giant string, we would lose page boundaries and provenance.

So the project uses one Document per page.

### What it does NOT do

DocumentLoader does not:

- split text into chunks,
- build embeddings,
- save vectors into a database,
- perform similarity search,
- or answer questions.

It only loads raw source content into structured Document objects.

### load_all() and _load_single_pdf()

- load_all()
  - scans the configured folder,
  - finds all PDF files,
  - calls _load_single_pdf() for each one,
  - returns a flat list of Document objects.

- _load_single_pdf()
  - opens one file,
  - loops through every page,
  - creates one Document per page,
  - closes the PDF at the end.

Flow:

```text
PDF → DocumentLoader → Document objects
```

---

## 6. Chunk Model

Once a document has been loaded, we often need to split the text into smaller parts.

### Why documents need to be split into smaller pieces

Large documents are hard to search and index efficiently.
A big page may contain many different ideas, and some of those ideas may be more useful for retrieval than the whole page.

Smaller pieces:

- make similarity search more focused,
- reduce noise,
- and preserve more exact matches.

### Why a Chunk object was created instead of a plain list of strings

The project uses a reusable Chunk dataclass:

```python
@dataclass
class Chunk:
    document: Document
    chunk_index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

This is better than plain strings because a chunk should remember where it came from.

### Why Chunk keeps a reference to the original Document

The chunk is not an isolated string.
It still needs to know:

- the original file,
- the page number,
- the source type,
- and any document-level metadata.

This preserves provenance across the pipeline.

### Fields in Chunk

- document
  - The original Document it was derived from.

- chunk_index
  - Its position within the document.

- text
  - The chunk content.

- metadata
  - Extra information specific to this chunk.

### Relationship

```text
Document → Chunk[]
```

This is the main idea behind the chunking stage: a document becomes many smaller units, each with source identity.

---

## 7. TextChunker

The chunking logic lives in data/chunker.py.

### What TextChunker does

TextChunker takes a list of Document objects and splits them into overlapping text segments.

The implemented logic is character-based, not sentence-based or token-based.

### Key ideas

- chunk_size
  - The maximum number of characters in each chunk.

- overlap
  - How many characters are shared between neighboring chunks.

### Why overlap is useful

Overlapping helps avoid cutting important phrases in half.
If a concept continues across the boundary between chunks, the overlapping area keeps context alive.

This is especially helpful for biomedical text where a single sentence may span a meaningful phrase or condition description.

### Why empty documents are ignored

If a page has no text, there is nothing useful to chunk or store.
The code checks for whitespace-only content and skips it.

This avoids creating useless empty chunks and keeps the vector store cleaner.

### Why chunking happens before embeddings

The system first creates chunk text, then converts those chunks to embeddings.
This is important because embeddings are generated for each chunk, not for the entire page or PDF at once.

The chunking stage creates the units of retrieval.
The embedding stage creates the searchable vectors for those units.

### Sliding-window logic in simple terms

The chunker starts at the beginning of a document, takes a window of size chunk_size, creates a chunk, then moves forward by:

```text
chunk_size - overlap
```

That means each next chunk starts slightly before the previous one ended.

This is a simple sliding-window model.

Example:

```text
Document text:  ABCDEFGHIJKLMNOP
chunk_size = 5
overlap = 2

Chunk 1: ABCDE
Chunk 2: EFGHI
Chunk 3: IJKLM
Chunk 4: MNOP
```

The actual implementation is slightly more general, but the idea is the same: keep some overlap so no information is lost at chunk boundaries.

---

## 8. Biomedical Embedding Model

The embedding layer is implemented in embeddings/model.py.

### What an embedding is

An embedding is a vector of numbers that represents the meaning of text.

The same idea behind word embeddings is used here, but at the sentence/phrase level.

Text is transformed into a long list of numbers that captures semantic similarity.

### Why we convert text into vectors

A database cannot directly compare text meaning with a similarity search unless it converts text into vectors.

Once text has an embedding, the system can compare vector distance or cosine similarity between:

- chunk embeddings,
- and query embeddings.

This lets the system find semantically similar content even if the exact words differ.

### Sentence Transformers model currently used

The project currently uses:

```python
pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
```

This is loaded through SentenceTransformer.

### Actual methods

- embed_text()
  - Takes a single string,
  - validates it is non-empty,
  - returns a NumPy embedding vector.

- embed_batch()
  - Takes a list of strings,
  - validates all entries,
  - returns a 2D array of embeddings.

### 768-dimensional output

The project expects:

```python
self.embedding_dim = 768
```

This means each embedding vector has 768 numbers.

This is a real dimensionality check used by the project and is also enforced in the vector store.

### Why the same embedding model is used for both documents/chunks and queries

The retrieval system compares query meaning against stored document/chunk meaning.

So both must live in the same embedding space.

If we used different models for storage and query, the vector comparison would not be meaningful.

### Important clarification

At this stage, we are not training the model on project-specific medical data.

We are using an existing biomedical Sentence Transformer model as a general-purpose embedding model for the project.

This is a retrieval pipeline setup, not a custom model-training pipeline.

### Simple example

```text
text → embedding model → vector of numbers
```

Example:

```text
"Pneumonia commonly causes fever and cough."
→ BiomedicalEmbeddingModel
→ [ -0.6595, -0.2579, 0.4356, ... , 768 numbers total ]
```

This vector is then used in Qdrant similarity search.

---

## 9. Qdrant Vector Store

The vector database layer is implemented in vector_store/qdrant_store.py.

### Why we need a vector database

A retrieval system needs a place to store many embeddings and search through them quickly.

Without a vector database, we would have to repeatedly scan all text by hand.

Qdrant gives us:

- efficient storage,
- vector indexing,
- cosine similarity search,
- and payload preservation.

### What Qdrant stores

The project stores a point that contains:

- the vector,
- and a payload with text and provenance metadata.

The conceptual pattern is:

```text
Chunk + embedding → Qdrant
```

### Vector + payload

Each stored item contains:

- vector: the 768-dimensional embedding,
- payload: text, filename, page, chunk index, source type, and metadata.

This is important because the vector alone is not enough for reasoning; we also need to know what text it represents.

### Cosine similarity

The collection is created with:

```python
Distance.COSINE
```

This means the search is based on cosine similarity, which is commonly used for semantic search.

### collection and vector_size = 768

The project uses a collection named:

```python
symptom_chunks
```

and the default vector size is:

```python
768
```

This matches the embedding model output dimension.

### upsert()/insert()

- upsert()
  - inserts or replaces points,
  - validates matching lengths between chunks and embeddings,
  - validates each vector dimensionality.

- insert()
  - compatibility alias to upsert().

### search()

The search method calls Qdrant with a query vector and limit:

```python
response = self.client.query_points(
    collection_name=self.collection_name,
    query=query,
    limit=top_k,
)
```

This returns the top-k matching points.

### Deterministic point IDs

Qdrant point IDs are generated by a UUID derived from the chunk identity:

- filename,
- page_number,
- source_type,
- chunk_index.

This makes the IDs deterministic for the same chunk content and source. It helps avoid duplicate or unstable IDs across runs.

### Why the payload preserves text and provenance

The payload stores:

- text,
- filename,
- page_number,
- chunk_index,
- source_type,
- document metadata,
- chunk metadata.

This is essential because after retrieval, we not only need to rank similar content; we also need to know exactly what was retrieved and where it came from.

The conceptual query path is:

```text
Query embedding → Qdrant → similar chunks
```

---

## 10. First Retrieval Integration Test

The first end-to-end retrieval test is in tests/test_rag_retrieval.py.

This test proved that the following worked together:

```text
Embedding Model → Qdrant → similarity search
```

It built a small document, embedded it, stored the vector in Qdrant, then searched for a related query.

The test validated that:

- a biomedical input text could be embedded,
- the vector could be stored in Qdrant,
- a semantically related query could be embedded,
- Qdrant could retrieve the correct chunk.

The successful result observed was:

```text
Similarity score: 0.6729074435345636
Retrieved evidence: Pneumonia can cause fever and cough, often with difficulty breathing.
```

And the test passed with:

```text
1 passed
```

### Why this test existed before the Retriever abstraction

This test came before BiomedicalRetriever existed.

At that stage, the project was validating the lower-level architecture directly:

- embedding model works,
- vector store works,
- search returns a meaningful result.

That was the first proof that the core retrieval stack was functioning before the code was wrapped in a higher-level retriever interface.

---

## 11. BiomedicalRetriever

The retrieval abstraction is implemented in retrieval/search.py.

This module creates a cleaner interface for the rest of the project.

### Purpose

BiomedicalRetriever takes a user query, converts it to a vector, searches Qdrant, and returns structured evidence objects.

### Data model: RetrievedEvidence

The dataclass is:

```python
@dataclass
class RetrievedEvidence:
    text: str
    score: float
    filename: str
    page_number: int | None
    chunk_index: int
    source_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

This is the result object that captures the retrieval result in a structured way.

### BiomedicalRetriever

The class has dependency injection for:

- embedding_model
- vector_store

This makes the retriever flexible and testable.

### retrieve(query, top_k)

The method does the following:

1. Validates that the query is a string and not empty.
2. Validates top_k is a positive integer.
3. Embeds the query using BiomedicalEmbeddingModel.
4. Calls the vector store search.
5. Converts each Qdrant result into a RetrievedEvidence object.

### Query validation

It checks:

- query must be a string,
- query cannot be empty or whitespace,
- top_k must be a positive integer.

This avoids invalid retrieval calls and keeps the system predictable.

### Qdrant search and conversion

The code pulls values from the Qdrant payload and creates RetrievedEvidence with:

- text,
- score,
- filename,
- page_number,
- chunk_index,
- source_type,
- metadata.

This preserves provenance and ranking information.

The flow is:

```text
Query
→ BiomedicalRetriever
→ BiomedicalEmbeddingModel
→ QdrantVectorStore
→ RetrievedEvidence[]
```

---

## 12. Retriever Integration Test

The retriever integration test is in tests/test_retriever.py.

This is different from the earlier test because it checks the abstraction, not the raw storage stack.

### Why it is different

Earlier test:

- tested Embedding Model + Qdrant directly.
- validated the lower-level pipeline.

This test:

- tests the Retriever abstraction.
- confirms the project now has a clean, reusable retrieval interface.
- validates the interface that a future RAG Agent will likely call.

It creates a document, embeds it, stores it in Qdrant, then calls:

```python
retriever.retrieve("What condition can cause fever and cough?", top_k=1)
```

The test verifies that the retrieved evidence has the expected text, filename, page, and source type.

It also checks invalid query handling with:

```python
with pytest.raises(ValueError):
    retriever.retrieve("   ")
```

### Successful result

The observed output was:

```text
Similarity score: 0.6729074435345636
Retrieved evidence: Pneumonia can cause fever and cough, often with difficulty breathing.
1 passed in 14.44s
```

This confirms the retrieval abstraction successfully wraps the lower-level embedding and vector-store stack.

---

## 13. EvidenceContextBuilder

The context-building layer is in reasoning/context.py.

### Why retrieved evidence must be converted into LLM-readable context

The retriever returns structured evidence objects, but an LLM or downstream module usually needs a plain text prompt block.

Without a formatting step, the evidence would still be machine-readable but not easy to consume in a prompt.

### EvidenceContextBuilder

This class provides:

```python
build_context(self, evidence: list[RetrievedEvidence]) -> str
```

It loops over evidence, creates a block for each item, and joins them into a single string.

### build_context() behavior

The output includes:

- evidence number,
- source filename,
- source type,
- page number,
- similarity score,
- and the actual evidence text.

This preserves source information and lets downstream modules understand what each evidence block is.

### Deterministic ordering

The builder preserves the order given in the evidence list.

This is useful because the retrieval order matters and should remain stable for debugging and prompt consistency.

### Empty evidence handling

If evidence is empty, the function returns an empty string.

This avoids producing a misleading or broken prompt when there are no retrieval results.

### Flow

```text
RetrievedEvidence[]
→ EvidenceContextBuilder
→ formatted evidence context
```

### Representative example

```text
[Evidence 1]
Source: pneumonia.pdf
Source type: PDF
Page: 3
Similarity: 0.91
Text: Pneumonia may cause fever and cough.

[Evidence 2]
Source: clinical_notes.txt
Source type: TXT
Page: 7
Similarity: 0.84
Text: Shortness of breath can accompany pneumonia.
```

This is the kind of text that is ready to be placed into the LLM prompt or future answer-generation step.

---

## 14. Context Builder Test

The context builder test is in tests/test_context_builder.py.

This test validates that the evidence context:

- is formatted deterministically,
- keeps the input order,
- includes the necessary fields,
- and returns an empty string for empty evidence.

The successful result is:

```text
1 passed in 7.38s
```

This confirms the context builder is stable and ready for use in the downstream prompt-building stage.

---

## 15. Biomedical Answerer and Prompt Layer

The reasoning prompt layer is in reasoning/prompts.py. It contains:

- SYSTEM_PROMPT, which defines evidence-grounded biomedical reasoning rules.
- build_prompt(clinical_context, evidence_context), which combines the
  prepared clinical information and numbered retrieved evidence into a
  deterministic user message.

The prompt instructs the model to generate preliminary diagnostic candidates,
rank them using supplied evidence, provide concise justifications, cite
supporting evidence numbers, reject unsupported conditions, and acknowledge
insufficient evidence. Retrieval similarity scores are treated as relevance
signals, not clinical probabilities. The prompt also makes clear that this is
not a final diagnosis; final synthesis and calibrated confidence belong to the
Clinical Data Fusion Agent.

### BiomedicalAnswerer

reasoning/answerer.py contains the internal models and application interface:

```python
DiagnosticCandidate(
    condition, rank, justification, supporting_evidence
)

RAGAnswer(diagnostic_candidates, limitations)
```

DiagnosticCandidate represents one preliminary diagnostic hypothesis. Its
supporting_evidence field contains the numbered evidence items produced by the
Context Builder. RAGAnswer is the internal Answerer output and is not the
final common Data Fusion output.

BiomedicalAnswerer loads GEMINI_API_KEY from the environment using
python-dotenv and uses the official google-genai SDK. Its model remains
configurable through the constructor, with the current default set to the
available Gemini Flash implementation model. It sends SYSTEM_PROMPT as the
system instruction and the output of build_prompt() as user content.

The Gemini request asks for application/json matching the RAGAnswer structure.
The response is parsed and validated before conversion into dataclass objects:
candidate conditions and justifications must be non-empty strings, ranks must
be integers, supporting evidence must be a list of integers, and limitations
must be a string. API failures are wrapped without exposing credentials.

The Answerer performs preliminary reasoning only. It does not retrieve data,
generate embeddings, access Qdrant, or produce the final Data Fusion output.

## 16. Symptom RAG Orchestrator

The first application-level orchestration layer is in orchestrator.py.
ClinicalTextClarifierOutput represents the structured input from the Clinical
Text Clarifier:

```python
{
    "clinical_text": "...",
    "diseases": [],
    "symptoms": [],
    "medications": [],
    "tests": []
}
```

The orchestrator preserves clinical_text and constructs a deterministic
retrieval query containing diseases first and symptoms second. Medications,
tests, and the full clinical text are intentionally excluded from this initial
retrieval query; the unchanged clinical_text is passed separately to the
Answerer.

The flow is:

```text
ClinicalTextClarifierOutput
→ disease/symptom retrieval query
→ BiomedicalRetriever
→ RetrievedEvidence[]
→ EvidenceContextBuilder
→ formatted evidence context
→ BiomedicalAnswerer
→ RAGAnswer
```

The orchestrator accepts Retriever, Context Builder, and Answerer instances
through its constructor. This dependency injection keeps the coordination
layer unit-testable without requiring real services. SymptomRAGResult retains
both the Answerer's RAGAnswer and the retrieved evidence so the next adapter
can expose provenance in the common output.

## 17. Important Architecture Decisions We Made

These are the main design ideas we have adopted in the implemented version of the project.

### 1. Preserve source/provenance throughout the pipeline

We keep the filename, page number, source type, and metadata all the way from Document to Chunk to Qdrant payload to RetrievedEvidence.

This is essential for evidence-based retrieval and post-hoc explanation.

### 2. Use Document and Chunk objects instead of raw strings

Raw text is too weak for production-quality retrieval.
These objects track source meaning and structure.

### 3. Keep embedding generation separate from retrieval

The embedding model is a reusable service that turns text into vectors.
The retriever does not manually implement embedding math; it uses the model.

### 4. Keep Qdrant-specific logic inside QdrantVectorStore

The rest of the application should not need to know the details of a vector database implementation.

### 5. Keep the Retriever independent from Qdrant implementation details

The retrieval interface should work as a higher-level abstraction, not as a direct dependency on low-level Qdrant APIs.

### 6. Keep context formatting separate from retrieval

Retrieval and prompt formatting are different concerns.
The retriever returns evidence objects; the context builder prepares them into readable text.

### 7. Use small components with single responsibilities

The project is intentionally decomposed into distinct roles:

- loaders,
- data models,
- chunking,
- embedding,
- vector storage,
- retrieval,
- context formatting.

This makes debugging and extension much easier.

### 8. Test each major layer independently and through integration tests

We validated the project at several levels:

- model-level behavior,
- retrieval pipeline behavior,
- and context formatting behavior.

This is a good engineering habit because it reduces the chance of hidden regressions.

---

## 18. Current Input/Output Boundary With Clinical Text Clarifier

This is the currently agreed conceptual input structure between the Clinical Text Clarifier and the RAG Agent layer.

This boundary is now represented by the typed ClinicalTextClarifierOutput model
used by the orchestrator. Procedures remain part of the broader planned input
shape, but are not yet stored on that model.

```json
{
  "clinical_text": "",
  "diseases": [],
  "symptoms": [],
  "medications": [],
  "tests": [],
  "procedures": []
}
```

### Meaning

This contract describes a clarified clinical note containing:

- the raw clinical text,
- extracted diseases,
- symptoms,
- medications,
- tests,
- and procedures.

This is the input contract currently accepted from the Clinical Text Clarifier
by our RAG Agent orchestration layer.

The exact inter-agent contract will be finalized during integration.

The broader multi-agent workflow is still incomplete because Data Fusion is
owned by another team, but the Symptom RAG side of this boundary is implemented.

---

## 19. Finalized Common Output Contract

The output adapter now produces the common outer structure for the Symptom RAG
agent.

```python
{
  "agent": "symptom_rag",
  "status": "success",
  "query_context": {
    "diseases": [],
    "symptoms": [],
    "medications": [],
    "tests": [],
    "procedures": []
  },
  "diagnostic_candidates": [],
  "evidence": [
    {
      "content": "",
      "source": "",
      "source_type": "",
      "location": "",
      "relevance_score": 0.0
    }
  ],
  "metadata": {
    "evidence_count": 0
  },
  "limitations": "",
  "error": null
}
```

### SymptomRAGOutputAdapter

The adapter receives ClinicalTextClarifierOutput and SymptomRAGResult. It copies
the query context, preserves candidate and evidence ordering, maps page numbers
to locations such as "Page 42", and leaves missing page numbers as an empty
string. It reports the actual retrieved evidence count and sets error to null
for successful output.

The adapter preserves RAGAnswer.limitations exactly. It does not create
candidates, fuse results, call Gemini, retrieve evidence, or implement the Data
Fusion Agent. All returned values are ordinary JSON-compatible Python values.

Procedures are currently represented by an empty list because the current
ClinicalTextClarifierOutput model does not contain procedure information.

---

## 20. Tests Completed So Far

The following table records the tests that are actually known from the current project history.

| test file | what it validates | result |
| --- | --- | --- |
| tests/test_biomedical_embedding_model.py | Biomedical embedding model loads and produces a 768-dimensional embedding vector for text. | Smoke check confirmed vector shape (768,) and embedding dimension 768. |
| tests/test_rag_retrieval.py | Embedding Model → Qdrant → similarity search works end-to-end. | Similarity score: 0.6729074435345636; Retrieved evidence: Pneumonia can cause fever and cough, often with difficulty breathing.; 1 passed |
| tests/test_retriever.py | Retriever abstraction returns structured RetrievedEvidence objects and validates bad queries. | Similarity score: 0.6729074435345636; Retrieved evidence: Pneumonia can cause fever and cough, often with difficulty breathing.; 1 passed in 14.44s |
| tests/test_context_builder.py | Context formatting is deterministic and preserves evidence ordering. | 1 passed in 7.38s |
| tests/test_answerer.py | Real Gemini Answerer integration, input validation, and structured RAGAnswer fields. | Integration request and edge-case behavior verified. |
| tests/test_orchestrator.py | Orchestrator coordinates retrieval, context formatting, and Answerer with injected dependencies. | End-to-end orchestration flow verified with controlled pneumonia data. |
| tests/test_output_adapter.py | Internal RAGAnswer and RetrievedEvidence are converted to the common JSON-compatible contract without Gemini. | Adapter mapping, locations, counts, limitations, and serialization verified. |
| tests/test_symptom_rag_agent_end_to_end.py | Complete CTC → orchestration → retrieval → Answerer → output adapter pipeline. | Final end-to-end integration test passed and produced a JSON-compatible Symptom RAG output. |

We also verified the full current suite with:

```text
11 passed in 64.94s (0:01:04)
```

The final end-to-end test used an in-memory Qdrant collection and controlled
biomedical test data about pneumonia, fever, cough, and difficulty breathing.
It verified the final output shape and JSON serialization without asserting
exact Gemini wording.

---

## 21. Current Progress

```text
Project Structure              ✅
Document Model                 ✅
PDF Loader                     ✅
Chunk Model                    ✅
Text Chunker                   ✅
Biomedical Embeddings          ✅
Qdrant Vector Store            ✅
End-to-End Retrieval Test      ✅
Retriever                      ✅
RAG Prompt/Context             ✅
LLM Answerer                   ✅
RAG Orchestrator               ✅
Common Output Adapter          ✅
Agent Integration Boundary     ✅
Data Fusion Integration        ⏳
```

This is the current status of the project as implemented and validated so far.

---

## 22. What We Understand So Far

The mental model of the current implemented system is:

- Documents are loaded.
- Documents become chunks.
- Chunks become embeddings.
- Embeddings are stored in Qdrant with provenance.
- A clinical query becomes an embedding.
- Qdrant retrieves similar biomedical chunks.
- The Retriever converts those results into RetrievedEvidence.
- The Context Builder converts evidence into LLM-ready context.
- The Answerer builds a prompt, sends it to Gemini using google-genai, and
  validates a structured RAGAnswer.
- The Orchestrator retains both the Answerer output and retrieved evidence in a
  SymptomRAGResult.
- The Output Adapter converts that result into the finalized agent-to-agent
  dictionary.

The current integration proves the application flow with controlled biomedical
test data. The next development phase is ingestion and indexing of the real
biomedical knowledge base.

This is the implemented Symptom RAG reasoning and output path. It is not yet
the complete multi-agent product because Data Fusion remains a separate phase.

---

## 23. Next Development Phase

The next development phase is real biomedical knowledge-base ingestion and
indexing. The current end-to-end integration uses controlled biomedical test
data, which verifies component connectivity and output structure but does not
replace indexing the intended production biomedical corpus.

---
