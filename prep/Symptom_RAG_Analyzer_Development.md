# Symptom RAG Analyzer --- Development

## Technology Stack

-   **Python 3.x** --- Core implementation
-   **Pandas** --- Dataset analysis and preprocessing
-   **Sentence Transformers** --- Biomedical embedding generation
-   **BioBERT-based embedding model** --- Converts clinical text into
    semantic vectors
-   **Qdrant** --- Vector database for storing and retrieving medical
    embeddings
-   **PyTest / Python test scripts** --- Component and pipeline
    validation

## Workflow

1.  **Load Clinical Data**
    -   Receive symptom--disease records from the medical dataset.
    -   Analyze and preprocess the records using Python/Pandas.
2.  **Create Medical Documents**
    -   Convert each clinical record into a structured **Document**.
    -   Represent relevant content as **Chunks** for retrieval.
3.  **Generate Biomedical Embeddings**
    -   Pass each medical document/chunk through the **BioBERT-based
        embedding model**.
    -   Convert clinical text into dense biomedical vectors.
4.  **Store in Vector Database**
    -   Store the generated embeddings and associated metadata in
        **Qdrant**.
    -   Build a searchable biomedical vector index.
5.  **Receive Clinical Query**
    -   Accept a clinical symptom/query as input.
    -   Generate its corresponding biomedical embedding.
6.  **Semantic Retrieval**
    -   Compare the query embedding against stored medical embeddings.
    -   Use **cosine similarity** to measure semantic relevance.
7.  **Retrieve Top-K Evidence**
    -   Rank the retrieved medical passages according to similarity.
    -   Select the **Top-K most relevant clinical evidence**.
8.  **Build Evidence Context**
    -   Pass the retrieved evidence to the **Evidence Context Builder**.
    -   Organize the evidence into a structured, LLM-ready context.
9.  **Return Structured Evidence**
    -   Return the retrieved evidence along with:
        -   Source
        -   Source type
        -   Location
        -   Relevance score
        -   Evidence count
10. **Validate the Pipeline**
    -   Test embedding generation, Qdrant retrieval, biomedical
        retriever, and context construction independently.
    -   Validate the complete RAG flow.

## Output

``` text
{
    agent: "Symptom RAG Analyzer",
    status: "success",
    query_context: "...",
    evidence: [
        {
            content: "...",
            source: "...",
            source_type: "...",
            location: "...",
            relevance_score: "..."
        }
    ],
    metadata: {
        evidence_count: N
    },
    error: null
}
```

## Overall Architecture

``` text
Clinical Dataset
      ↓
Document Processing
      ↓
Medical Documents / Chunks
      ↓
BioBERT-based Embeddings
      ↓
Qdrant Vector Database
      ↓
Clinical Symptom Query
      ↓
Query Embedding
      ↓
Cosine Similarity Search
      ↓
Top-K Relevant Evidence
      ↓
Evidence Context Builder
      ↓
Structured Clinical Evidence
```

## Current Development Status

### Implemented

`Data Processing → Biomedical Embeddings → Qdrant Indexing → Semantic Retrieval → Top-K Evidence → Evidence Context Construction → Testing`

### Next

`Knowledge Graph → Graph-RAG → Evidence Fusion → Multi-Agent Clinical Reasoning`
