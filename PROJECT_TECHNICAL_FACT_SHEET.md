# Project Technical Fact Sheet

This document is the repository-grounded technical source of truth for the current implementation of this project. It intentionally separates what is implemented, tested, integrated, and actually used at runtime.

## 1. Scope and evidence standard

This project currently contains a Python biomedical retrieval pipeline with a persistent Qdrant vector store, a Gemini-based reasoning step, and a symptom-oriented RAG workflow. The claim below is limited to what is directly evidenced in the repository.

Important evidence rules used in this document:

- If a feature is supported by actual code and tests, it is marked as implemented and tested.
- If a feature exists only as a file/class stub or as a design intent in documentation, it is marked as not implemented or not connected to runtime.
- If a feature is referenced in one place but not actually wired into the runtime code path, it is clearly separated from runtime use.
- Anything not verifiable from the repository is stated as `NOT VERIFIED FROM REPOSITORY`.

## 2. High-level status summary

| Area | Status | Evidence |
|---|---|---|
| Symptom-oriented biomedical RAG | IMPLEMENTED, TESTED, INTEGRATED, CURRENTLY USED AT RUNTIME | [src/symptom_rag_analyzer/orchestrator.py](src/symptom_rag_analyzer/orchestrator.py), [src/symptom_rag_analyzer/retrieval/search.py](src/symptom_rag_analyzer/retrieval/search.py), [src/symptom_rag_analyzer/reasoning/answerer.py](src/symptom_rag_analyzer/reasoning/answerer.py), [tests/test_e2e_real_rag.py](tests/test_e2e_real_rag.py) |
| Literature ingestion and PMC parsing | IMPLEMENTED, TESTED | [src/symptom_rag_analyzer/data/literature_parser.py](src/symptom_rag_analyzer/data/literature_parser.py), [src/symptom_rag_analyzer/ingestion/literature_embedding_indexer.py](src/symptom_rag_analyzer/ingestion/literature_embedding_indexer.py), [tests/test_literature_chunker.py](tests/test_literature_chunker.py) |
| Symptom dataset ingestion | IMPLEMENTED, TESTED | [ingest.py](ingest.py), [src/symptom_rag_analyzer/data/dataset_loaders.py](src/symptom_rag_analyzer/data/dataset_loaders.py) |
| Multi-agent architecture | PARTIALLY DESCRIBED IN DOCUMENTATION; NOT IMPLEMENTED AS SEPARATE AGENTS IN CODE | [README.md](README.md), [src/symptom_rag_analyzer/orchestrator.py](src/symptom_rag_analyzer/orchestrator.py) |
| Graph Retrieval / PrimeKG / NetworkX | NOT PRESENT IN REPOSITORY | Search across repo returned no matches for PrimeKG/NetworkX/graph retrieval symbols |
| Clinical Text Clarifier as a standalone agent | NOT IMPLEMENTED AS A REAL AGENT | [src/symptom_rag_analyzer/orchestrator.py](src/symptom_rag_analyzer/orchestrator.py) exposes a dataclass `ClinicalTextClarifierOutput`, not a functional agent |

## 3. Repository structure and runtime artifacts

The repository includes:

- Application package under [src/symptom_rag_analyzer](src/symptom_rag_analyzer)
- Runtime data under [data](data)
- Literature-processing scripts under [scripts](scripts)
- Tests under [tests](tests)
- Root ingestion script: [ingest.py](ingest.py)
- Dependency manifest: [requirements.txt](requirements.txt)
- Packaging metadata: [pyproject.toml](pyproject.toml)

Key runtime data folders currently present in the repository include:

- [data/qdrant](data/qdrant) — persistent local Qdrant storage used by the code
- [data/qdrant_server](data/qdrant_server) — additional Qdrant server snapshot / server data
- [data/raw](data/raw) — raw source data including symptom-disease dataset and PMC XML inputs
- [data/processed](data/processed) — processed PubMed and PMC manifest data

## 4. Actual implemented project architecture

### 4.1 Symptom RAG pipeline (the concrete runtime path)

The active runtime pipeline is a single symptom-focused retrieval-and-reasoning stack built around the orchestrator and vector store.

Flow:

1. Structured clinical context is represented as `ClinicalTextClarifierOutput` in [src/symptom_rag_analyzer/orchestrator.py](src/symptom_rag_analyzer/orchestrator.py).
2. The orchestrator builds a retrieval query from disease and symptom fields using `_build_retrieval_query()`.
3. `BiomedicalRetriever.retrieve()` embeds the query with `BiomedicalEmbeddingModel.embed_text()` and calls `QdrantVectorStore.search()` in [src/symptom_rag_analyzer/retrieval/search.py](src/symptom_rag_analyzer/retrieval/search.py).
4. `EvidenceContextBuilder.build_context()` turns retrieved chunks into a numbered passage list in [src/symptom_rag_analyzer/reasoning/context.py](src/symptom_rag_analyzer/reasoning/context.py).
5. `BiomedicalAnswerer.answer()` sends the clinical context and the evidence context to Gemini using the Google GenAI SDK and a JSON schema in [src/symptom_rag_analyzer/reasoning/answerer.py](src/symptom_rag_analyzer/reasoning/answerer.py).
6. `SymptomRAGOutputAdapter.adapt()` converts the internal answer to a common output contract in [src/symptom_rag_analyzer/output_adapter.py](src/symptom_rag_analyzer/output_adapter.py).

This is the clearest end-to-end implementation in the repo.

### 4.2 Embedded model and retrieval vector stack

The embedding model is configured in [src/symptom_rag_analyzer/embeddings/model.py](src/symptom_rag_analyzer/embeddings/model.py):

- Model: `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`
- Embedding dimension: `768`
- Back-end: `sentence-transformers` `SentenceTransformer`
- Retrieval metric: cosine distance in Qdrant via `VectorParams(size=768, distance=Distance.COSINE)` in [src/symptom_rag_analyzer/vector_store/qdrant_store.py](src/symptom_rag_analyzer/vector_store/qdrant_store.py)

The Qdrant vector store supports:

- collection creation if missing
- chunk payload insertion
- deterministic point IDs
- vector search and ID retrieval

### 4.3 Critical value and collection audit

This section reconciles the numerical and collection claims called out during the repository audit.

| Item | Repository evidence | Status |
|---|---|---|
| `symptom_chunks` | Used in the default symptom vector store and the E2E runtime test; also described in README | VERIFIED |
| `literature_chunks` | Default collection name in [src/symptom_rag_analyzer/ingestion/literature_embedding_indexer.py](src/symptom_rag_analyzer/ingestion/literature_embedding_indexer.py) and benchmark/protection logic in [scripts/benchmark_literature_embedding_ingestion.py](scripts/benchmark_literature_embedding_ingestion.py) | VERIFIED |
| `literature_chunks_benchmark` | Declared in README and created in [scripts/benchmark_literature_embedding_ingestion.py](scripts/benchmark_literature_embedding_ingestion.py) as a benchmark-only collection | VERIFIED AS A BENCHMARK ARTIFACT |
| `768` dimensions | Explicit in [src/symptom_rag_analyzer/embeddings/model.py](src/symptom_rag_analyzer/embeddings/model.py) and [src/symptom_rag_analyzer/vector_store/qdrant_store.py](src/symptom_rag_analyzer/vector_store/qdrant_store.py) | VERIFIED |
| `top_k` / `top-k` retrieval | `SymptomRAGOrchestrator(top_k=5)` and `BiomedicalRetriever.retrieve(query, top_k=top_k)` in [src/symptom_rag_analyzer/orchestrator.py](src/symptom_rag_analyzer/orchestrator.py) and [src/symptom_rag_analyzer/retrieval/search.py](src/symptom_rag_analyzer/retrieval/search.py) | VERIFIED |
| cosine distance metric | `Distance.COSINE` in [src/symptom_rag_analyzer/vector_store/qdrant_store.py](src/symptom_rag_analyzer/vector_store/qdrant_store.py) | VERIFIED |
| query construction | `_build_retrieval_query()` creates a disease + symptom query in [src/symptom_rag_analyzer/orchestrator.py](src/symptom_rag_analyzer/orchestrator.py) | VERIFIED |
| `gemini-3.6-flash` | Default model in [src/symptom_rag_analyzer/reasoning/answerer.py](src/symptom_rag_analyzer/reasoning/answerer.py) and README | VERIFIED |
| `1000` | `SAMPLE_CHUNK_COUNT = 1000` in [scripts/benchmark_literature_embedding_ingestion.py](scripts/benchmark_literature_embedding_ingestion.py) | VERIFIED AS A BENCHMARK SAMPLE SIZE |
| `150` | `LiteratureSectionChunker(target_size=1000, overlap=150)` in [src/symptom_rag_analyzer/ingestion/literature_embedding_indexer.py](src/symptom_rag_analyzer/ingestion/literature_embedding_indexer.py) and benchmark script | VERIFIED |
| `201,139` | `FULL_CORPUS_WINDOW_COUNT = 201_139` in [scripts/benchmark_literature_embedding_ingestion.py](scripts/benchmark_literature_embedding_ingestion.py) | PARTIALLY VERIFIED: benchmark-based estimate, not a confirmed production corpus size |
| `2,524` | No repository file contains this exact value as a definitive count. The manifest in [data/processed/pubmed_fulltext_manifest.csv](data/processed/pubmed_fulltext_manifest.csv) contains a row count and a `pmc` subset count that is recorded as repository evidence; the specific `2,524` literal is not present in the repo. | NOT VERIFIED FROM REPOSITORY |
| `82,313` | No repository file contains a matching count or a source for this figure. | NOT VERIFIED FROM REPOSITORY |
| `201,134` | No repository file contains this exact figure. The nearby benchmark code uses `201_139`, not `201_134`. | CONTRADICTED BY REPOSITORY EVIDENCE / NOT VERIFIED FROM REPOSITORY |
| `5 skipped records` | No repository file or script contains a literal "5 skipped records" count. The benchmark script validates protected collections, but does not declare a skipped-record total. | NOT VERIFIED FROM REPOSITORY |
| `100 tokens` | No repository file or tokenizer configuration records this exact token-capacity value as a canonical global constant. The code uses tokenizer-derived limits and chunk windows rather than a fixed 100-token window size. | NOT VERIFIED FROM REPOSITORY |
| `98 tokens` | No repository file or tokenizer configuration records this exact value as a canonical model limit. | NOT VERIFIED FROM REPOSITORY |
| literature/runtime integration | The literature indexer and symptom runtime are both implemented separately, and the benchmark script protects production collections while building the benchmark collection; there is no direct code path proving that the live runtime question-answering path uses `literature_chunks` as an active retrieval source in the same way as `symptom_chunks`. | PARTIALLY VERIFIED |

Important clarification: the repository contains a real literature pipeline and a real symptom RAG pipeline, but the strongest evidence for the live runtime path is the symptom RAG pipeline using `symptom_chunks`. The literature collection is implemented and separate, not proven to be the active retrieval source for the main runtime answerer.

### 4.3 Indexing pipeline for the symptom dataset

The symptom-disease knowledge base is ingested from a dataset CSV and mapping file. The concrete workflow is:

- loader: [src/symptom_rag_analyzer/data/dataset_loaders.py](src/symptom_rag_analyzer/data/dataset_loaders.py)
- entry point: [ingest.py](ingest.py)
- indexer: [src/symptom_rag_analyzer/retrieval/indexer.py](src/symptom_rag_analyzer/retrieval/indexer.py)

The loader:

- reads the training CSV from `data/raw/symptom-disease-dataset/`
- resolves disease IDs through `mapping.json`
- deduplicates exact `(text, label)` records
- converts each unique row into a `Document` object containing:
  - `filename`: `symptom-disease-train-dataset.csv`
  - `text`: `Disease: <name>\nSymptoms: <symptom text>`
  - metadata with `disease_id`, `disease_name`, `raw_symptom_text`, and data source provenance

The indexer:

- loads all documents
- splits them into overlapping character chunks via `TextChunker`
- embeds the chunk texts with `BiomedicalEmbeddingModel.embed_batch()`
- writes them to Qdrant

The default collection name used here is `symptom_chunks`.

### 4.4 Literature ingestion and biomedical retrieval pipeline

This repository includes a separate literature pipeline for biomedical literature, which is clearly distinct from the symptom dataset pipeline.

#### Source of literature

The repository includes scripts and processed CSVs that explicitly describe PubMed and PMC as the literature source. Examples:

- [scripts/select_pubmed_literature.py](scripts/select_pubmed_literature.py) reads `pubmed_complete_metadata.csv` and selects candidate literature records.
- [scripts/create_pubmed_fulltext_manifest.py](scripts/create_pubmed_fulltext_manifest.py) creates `pubmed_fulltext_manifest.csv` with `acquisition_route` values `pmc`, `doi_candidate`, or `metadata_only`.
- [scripts/ingest_literature.py](scripts/ingest_literature.py) resolves the authoritative PMC manifest and does the local ingestion into `literature_chunks`.

The actual runtime data presently present includes a PMC manifest with 3,783 rows and 2,526 rows with `acquisition_route == 'pmc'`, verified from the file at [data/processed/pubmed_fulltext_manifest.csv](data/processed/pubmed_fulltext_manifest.csv). This is the strongest evidence for the actual literature source count in the repository.

#### Document format and parsing

The PMC literature parser is implemented in [src/symptom_rag_analyzer/data/literature_parser.py](src/symptom_rag_analyzer/data/literature_parser.py):

- Parses one PMC JATS XML file into a `LiteratureArticle`
- Handles namespaces via `_local_name()` and `_attribute()` helpers
- Extracts identifiers, authors, publication metadata, MeSH terms, abstract sections, body sections, and section hierarchy
- Ignores references, captions, tables, and figure content via `_IGNORED_TEXT_TAGS`
- Normalizes section titles using `normalize_section_title()` and categorizes them via `categorize_section()`

Important implementation details:

- abstract sections are retained separately from body sections
- section paths are preserved as a list
- section categories are normalized to labels such as `introduction`, `methods`, `results`, `discussion`, `diagnosis`, `clinical_presentation`, etc.

#### Section-aware chunking and windowing

The literature chunking logic is implemented in:

- [src/symptom_rag_analyzer/data/literature_chunker.py](src/symptom_rag_analyzer/data/literature_chunker.py)
- [src/symptom_rag_analyzer/data/literature_models.py](src/symptom_rag_analyzer/data/literature_models.py)
- [src/symptom_rag_analyzer/embeddings/literature_embedding_windows.py](src/symptom_rag_analyzer/embeddings/literature_embedding_windows.py)

Key facts from the code:

- `LiteratureSectionChunker.__init__(target_size=1000, overlap=150)`
- chunking is paragraph-aware and section-isolated
- oversized paragraphs are recursively split with overlap using `self.target_size - self.overlap` step size
- chunks are created per section and preserve `section_title`, `normalized_section_title`, `section_path`, `category`, `source`, and `level`
- chunk indices restart per section
- abstract sections are processed before body sections

Window splitting is tokenizer-bounded:

- `LiteratureEmbeddingWindowSplitter` obtains the model tokenizer and `max_seq_length`
- `content_token_capacity = max_tokens - special_token_count`
- windows are split without losing or repeating content tokens
- the splitter validates that the reconstructed text matches the original chunk exactly

This is the actual implemented literature chunking flow used before embedding.

#### Literature embedding and Qdrant upload

The production literature path is defined in [src/symptom_rag_analyzer/ingestion/literature_embedding_indexer.py](src/symptom_rag_analyzer/ingestion/literature_embedding_indexer.py):

- default collection: `literature_chunks`
- default Qdrant path: `data/qdrant`
- embedding batch size: 32
- deterministic UUIDs per literature window using article, section, and chunk identity
- metadata payload includes article metadata, section metadata, and window metadata
- indexer enforces that the Qdrant vector size matches the model dimension and that the collection is not `symptom_chunks`

This confirms the actual literature vector store is a separate collection from the symptom dataset collection.

## 5. RAG data flow, as implemented

The end-to-end flow in code is:

`source data -> document loading -> chunking -> embedding -> vector indexing -> query embedding -> vector retrieval -> evidence formatting -> LLM reasoning -> structured output`

Verified implementation path:

1. Symptom dataset: [src/symptom_rag_analyzer/data/dataset_loaders.py](src/symptom_rag_analyzer/data/dataset_loaders.py)
2. PDF/Document loading: [src/symptom_rag_analyzer/data/loaders.py](src/symptom_rag_analyzer/data/loaders.py)
3. Generic chunking: [src/symptom_rag_analyzer/data/chunker.py](src/symptom_rag_analyzer/data/chunker.py)
4. Biomedical embedding: [src/symptom_rag_analyzer/embeddings/model.py](src/symptom_rag_analyzer/embeddings/model.py)
5. Qdrant ingestion: [src/symptom_rag_analyzer/vector_store/qdrant_store.py](src/symptom_rag_analyzer/vector_store/qdrant_store.py)
6. Retrieval: [src/symptom_rag_analyzer/retrieval/search.py](src/symptom_rag_analyzer/retrieval/search.py)
7. Evidence prompt assembly: [src/symptom_rag_analyzer/reasoning/context.py](src/symptom_rag_analyzer/reasoning/context.py)
8. Gemini answer generation: [src/symptom_rag_analyzer/reasoning/answerer.py](src/symptom_rag_analyzer/reasoning/answerer.py)
9. Output adaptation: [src/symptom_rag_analyzer/output_adapter.py](src/symptom_rag_analyzer/output_adapter.py)

The actual repository does not contain a separate retrieval orchestration framework beyond this concrete stack.

## 6. Actual runtime configuration and environment

The runtime requirements and configuration are specified in:

- [README.md](README.md)
- [.env.example](.env.example)
- [pyproject.toml](pyproject.toml)
- [requirements.txt](requirements.txt)

The concrete runtime config in repo evidence is:

- Python >= 3.10
- Google Gemini API key via `GEMINI_API_KEY`
- Qdrant persistent storage in `data/qdrant`
- `gemini-3.6-flash` in [src/symptom_rag_analyzer/reasoning/answerer.py](src/symptom_rag_analyzer/reasoning/answerer.py)
- `NCBI_API_KEY` and `NCBI_EMAIL` are recognized in [.env.example](.env.example) but no active code path in the repository uses them directly in the runtime RAG path
- `PMC_REQUEST_DELAY` is configurable but not wired into the active runtime path by direct evidence

## 7. Evidence-backed test status

The repository includes many tests, but not all are end-to-end production integrations.

### Implemented and tested

- Unit tests for the literature chunker: [tests/test_literature_chunker.py](tests/test_literature_chunker.py)
- Retrieval and indexer tests: [tests/test_retriever.py](tests/test_retriever.py), [tests/test_indexer.py](tests/test_indexer.py)
- Output adapter tests: [tests/test_output_adapter.py](tests/test_output_adapter.py)
- Orchestrator tests: [tests/test_orchestrator.py](tests/test_orchestrator.py)
- Real persistent Qdrant E2E test: [tests/test_e2e_real_rag.py](tests/test_e2e_real_rag.py)

### Runtime-use status

The strongest runtime-use evidence is the real E2E test that connects to the persistent collections under [data/qdrant](data/qdrant) and asserts the symptom RAG pipeline behaves successfully. This is not merely a mock or synthetic test.

## 8. Multi-agent and graph-component status

### 8.1 Multi-agent orchestration

The project documentation speaks about a multi-agent clinical decision support system, but the actual code implements a single orchestrator and an answerer. There is no evidence of separate agent classes for:

- Clinical Text Clarifier
- Graph Retrieval Agent
- Data Fusion Agent
- final Clinical Decision Manager

The actual repository code contains exactly one orchestrator class plus a dataclass contract for clinical text; there is no independent multi-agent runtime loop or inter-agent message infrastructure.

This means:

- `IMPLEMENTED`: single symptom RAG pipeline and output adapter
- `TESTED`: orchestrator and E2E tests
- `INTEGRATED`: yes, into the package runtime path
- `CURRENTLY USED AT RUNTIME`: yes, in the symptom-oriented retrieval pipeline
- `NOT IMPLEMENTED`: the broader multi-agent architecture described in the project narrative

### 8.2 Graph Retrieval / Medical Knowledge Graph

There are no repository code references for:

- PrimeKG
- NetworkX medical graph
- graph retrieval agent
- graph database loader
- adjacency-based clinical reasoning

The repository-wide search returned no actual graph retrieval implementation. Therefore this component is:

- `NOT VERIFIED FROM REPOSITORY` as an implemented runtime module
- `NOT IMPLEMENTED` in the current codebase

## 9. Contradictions and caveats

The repository contains at least two notable contradictions between documentation and implementation:

1. README says the persistent Qdrant database contains `symptom_chunks`, `literature_chunks`, and `literature_chunks_benchmark`.
   - The code explicitly targets `symptom_chunks` and `literature_chunks`.
   - `literature_chunks_benchmark` is present as a data artifact directory and is not referenced by the source package logic in the main runtime path.
   - Conclusion: the benchmark collection is evidence of a data artifact, but not a confirmed active runtime component in the package code.

2. The project context presents a broad multi-agent clinical decision support system and graph-based knowledge retrieval.
   - The actual implementation is a single symptom RAG pipeline built around Qdrant + Gemini + output adapter.
   - The graph and multi-agent parts are not reflected in the actual package or runtime code.
   - Conclusion: the project documentation describes a broader research direction, but the repository currently implements a narrower code path.

## 10. Final implementation summary

The repository currently implements a functioning symptom-oriented biomedical RAG pipeline that:

- accepts structured clinical context
- embeds disease/symptom queries into a BioBERT-based vector space
- retrieves relevant chunk evidence from a local Qdrant index
- builds evidence context for Gemini
- produces ranked diagnostic candidates with supporting evidence numbers
- emits a JSON-compatible agent output

The repository also contains a literature indexing pipeline that:

- parses local PMC JATS XML files
- extracts section-aware article metadata and text
- chunks sections with overlap and hierarchy preservation
- token-window-splits the chunks
- embeds them and stores them in a distinct `literature_chunks` collection

However, the repository does not contain a complete multi-agent orchestration system and does not implement a graph-based retrieval component in code. Those are therefore best understood as aspirational project scope rather than current implementation fact.

## 11. Evidence file references

Core runtime files:

- [src/symptom_rag_analyzer/orchestrator.py](src/symptom_rag_analyzer/orchestrator.py)
- [src/symptom_rag_analyzer/retrieval/search.py](src/symptom_rag_analyzer/retrieval/search.py)
- [src/symptom_rag_analyzer/reasoning/answerer.py](src/symptom_rag_analyzer/reasoning/answerer.py)
- [src/symptom_rag_analyzer/output_adapter.py](src/symptom_rag_analyzer/output_adapter.py)
- [src/symptom_rag_analyzer/vector_store/qdrant_store.py](src/symptom_rag_analyzer/vector_store/qdrant_store.py)
- [src/symptom_rag_analyzer/embeddings/model.py](src/symptom_rag_analyzer/embeddings/model.py)

Literature-specific files:

- [src/symptom_rag_analyzer/data/literature_parser.py](src/symptom_rag_analyzer/data/literature_parser.py)
- [src/symptom_rag_analyzer/data/literature_chunker.py](src/symptom_rag_analyzer/data/literature_chunker.py)
- [src/symptom_rag_analyzer/embeddings/literature_embedding_windows.py](src/symptom_rag_analyzer/embeddings/literature_embedding_windows.py)
- [src/symptom_rag_analyzer/ingestion/literature_embedding_indexer.py](src/symptom_rag_analyzer/ingestion/literature_embedding_indexer.py)
- [scripts/select_pubmed_literature.py](scripts/select_pubmed_literature.py)
- [scripts/create_pubmed_fulltext_manifest.py](scripts/create_pubmed_fulltext_manifest.py)
- [scripts/ingest_literature.py](scripts/ingest_literature.py)

Data and docs:

- [README.md](README.md)
- [.env.example](.env.example)
- [requirements.txt](requirements.txt)
- [pyproject.toml](pyproject.toml)
- [data/raw/symptom-disease-dataset/mapping.json](data/raw/symptom-disease-dataset/mapping.json)
- [data/processed/pubmed_fulltext_manifest.csv](data/processed/pubmed_fulltext_manifest.csv)

This project currently implements a concrete biomedical symptom RAG pipeline and a literature embedding pipeline, but not the broader multi-agent or graph-retrieval architecture described in the narrative context.
