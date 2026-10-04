# Symptom RAG Analyzer

A frozen, production-validated **Symptom RAG Analyzer** component for the Agentic Graph-RAG Clinical Decision Support System.

This repository contains the Vector RAG component responsible for retrieving biomedical evidence and generating preliminary diagnostic candidates from structured clinical information.

> **Important:** This repository contains the application code and configuration required to run the Symptom RAG Analyzer. The validated production Qdrant vector data is maintained separately and is **not stored in GitHub**.

---

# 1. What This Agent Does

The Symptom RAG Analyzer receives structured clinical information from the **Clinical Text Clarifier (CTC)** and performs evidence retrieval using biomedical semantic embeddings.

The agent:

1. Receives structured clinical information from the Clinical Text Clarifier.
2. Constructs a retrieval query from the available disease and symptom information.
3. Generates a biomedical embedding for the query.
4. Searches two independent Qdrant collections:
   - `symptom_chunks`
   - `literature_chunks`
5. Retrieves the top-k evidence from each collection.
6. Preserves the two evidence sources separately.
7. Builds a structured evidence context.
8. Sends the clinical information and retrieved evidence to the Answerer LLM.
9. Produces preliminary diagnostic candidates supported by retrieved evidence.
10. Returns a structured output for downstream clinical data fusion.

The Symptom RAG Analyzer is **not the final diagnostic decision-maker**.

Its output is intended to be consumed by the downstream **Clinical Data Fusion Agent**, where additional evidence such as Graph RAG and web-based evidence can be incorporated.

---

# 2. Position in the Overall System

The overall architecture is:

```text
                    Clinical Text
                         |
                         v
              Clinical Text Clarifier
                         |
                         v
                 Structured Clinical
                     Information
                         |
             +-----------+-----------+
             |                       |
             v                       v
       Vector RAG                 Graph RAG
             |                       |
             |                       |
             +-----------+-----------+
                         |
                         v
                  Evidence Fusion
                         |
                         v
                 Clinical Data
                  Fusion Agent
                         |
                         v
               Clinical Decision
                    Support
```

The Vector RAG branch implemented in this repository is:

```text
Clinical Text Clarifier
          |
          v
Clinical Query Construction
          |
          v
Biomedical Embedding
          |
          v
       Qdrant
     /         \
    v           v
symptom_chunks  literature_chunks
    |           |
    +-----+-----+
          |
          v
   Retrieved Evidence
          |
          v
    Evidence Context
          |
          v
     Answerer LLM
          |
          v
 Diagnostic Candidates
```

---

# 3. Input Expected from the Clinical Text Clarifier

The Symptom RAG Analyzer expects a structured CTC output.

The current interface is:

```python
ClinicalTextClarifierOutput(
    clinical_text: str,
    diseases: list[str],
    symptoms: list[str],
    medications: list[str],
    tests: list[str]
)
```

Example:

```json
{
  "clinical_text": "A patient presents with sudden central chest pressure radiating to the left arm with sweating and nausea.",
  "diseases": [],
  "symptoms": [
    "central chest pressure",
    "chest pain radiating to left arm",
    "sweating",
    "nausea"
  ],
  "medications": [],
  "tests": []
}
```

The `clinical_text` is preserved and passed to the Answerer for clinical reasoning.

The current retrieval query is constructed from:

```text
Diseases
+
Symptoms
```

The current orchestrator does **not** use medications or tests for retrieval-query construction.

This is an intentional implementation decision in the current frozen version.

---

# 4. Query Construction

The retrieval query is constructed from the structured disease and symptom information.

Example:

```text
Diseases:
Symptoms: central chest pressure, chest pain radiating to left arm, sweating, nausea
```

If diseases are available:

```text
Diseases: Myocardial infarction
Symptoms: chest pain, sweating, nausea
```

This query is converted into a biomedical embedding before retrieval.

---

# 5. Biomedical Embedding Model

The same biomedical embedding model is used for the retrieval query and the indexed vector data.

Model:

```text
pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
```

Embedding dimension:

```text
768
```

Similarity metric:

```text
Cosine similarity
```

The model is used for semantic biomedical retrieval.

No fine-tuning is performed by the current Symptom RAG Analyzer.

---

# 6. Retrieval Architecture

The system uses **two independent Qdrant collections**.

```text
                    Clinical Query
                         |
                         v
                Biomedical Embedding
                         |
              +----------+----------+
              |                     |
              v                     v
       symptom_chunks        literature_chunks
              |                     |
              v                     v
        Top-k Symptom          Top-k Literature
          Evidence                Evidence
              |                     |
              +----------+----------+
                         |
                         v
                 Evidence Context
```

The two collections are queried independently.

Their raw similarity scores are **not blindly cross-ranked against each other**.

This preserves the distinction between:

- symptom-disease dataset evidence
- biomedical literature evidence

---

# 7. Qdrant Collections

The validated production environment contains:

```text
Qdrant
http://localhost:6333
```

with the following collections:

```text
symptom_chunks
literature_chunks
```

Current validated production state:

```text
symptom_chunks
    Vectors: 2,690
    Dimension: 768
    Distance: Cosine

literature_chunks
    Vectors: 201,134
    Dimension: 768
    Distance: Cosine
```

Expected production structure:

```text
Qdrant
|
+-- symptom_chunks
|      +-- 2,690 vectors
|      +-- 768 dimensions
|      +-- Cosine similarity
|
+-- literature_chunks
       +-- 201,134 vectors
       +-- 768 dimensions
       +-- Cosine similarity
```

---

# 8. Where is the Qdrant Data?

The Qdrant vector database is **not stored inside this Git repository**.

The validated production Qdrant instance currently runs locally using Docker.

Current endpoint:

```text
http://localhost:6333
```

Current Docker container:

```text
symptom-rag-qdrant
```

The repository intentionally does not include the vector database because the production vector data is large and should not be committed to Git.

Therefore:

> **Cloning this repository alone does not provide the indexed Qdrant collections.**

The integration developer must obtain the validated Qdrant production backup/export separately and restore it into a local Qdrant instance.

The required final state on the integration machine is:

```text
http://localhost:6333
        |
        +-- symptom_chunks
        |      2,690 vectors
        |
        +-- literature_chunks
               201,134 vectors
```

The Qdrant handoff procedure is documented separately from the source-code repository.

---

# 9. Retrieval Process

For every clinical query:

### Step 1 — Clinical Query

The Symptom RAG Analyzer receives the structured CTC output.

### Step 2 — Query Construction

Diseases and symptoms are converted into a retrieval query.

### Step 3 — Biomedical Embedding

The retrieval query is converted into a 768-dimensional biomedical embedding.

### Step 4 — Independent Retrieval

The embedding is searched against:

```text
symptom_chunks
```

and:

```text
literature_chunks
```

independently.

### Step 5 — Top-k Evidence

The current default is:

```text
top_k = 5
```

per collection.

Therefore a normal retrieval produces:

```text
5 symptom evidence items
+
5 literature evidence items
=
10 total evidence items
```

### Step 6 — Evidence Context

The retrieved passages are formatted into a structured evidence context.

### Step 7 — Answerer

The clinical information and retrieved evidence are sent to the Answerer LLM.

### Step 8 — Diagnostic Candidates

The Answerer produces preliminary diagnostic candidates supported by retrieved evidence.

---

# 10. Evidence Sources

The Symptom RAG Analyzer currently uses two evidence sources.

## 10.1 Symptom-Disease Dataset

Qdrant collection:

```text
symptom_chunks
```

Source type:

```text
symptom-disease-dataset
```

Current production size:

```text
2,690 vectors
```

This collection provides symptom-to-disease semantic evidence.

---

## 10.2 Biomedical Literature

Qdrant collection:

```text
literature_chunks
```

Source type:

```text
pubmed-pmc-literature
```

Current production size:

```text
201,134 vectors
```

This collection contains biomedical literature retrieved and processed from PubMed/PMC sources.

---

# 11. Evidence Numbering

Evidence is numbered globally after retrieval.

For example:

```text
Evidence 1
Evidence 2
Evidence 3
Evidence 4
Evidence 5
```

are the symptom-dataset results.

```text
Evidence 6
Evidence 7
Evidence 8
Evidence 9
Evidence 10
```

are the literature results.

The Answerer refers to these evidence numbers when explaining diagnostic candidates.

Example:

```text
Candidate:
Acute coronary syndrome

Supporting evidence:
[1, 3]
```

---

# 12. Answerer LLM

The Answerer is implemented inside the Symptom RAG Analyzer.

Current provider:

```text
OpenRouter
```

Current model configuration:

```text
openai/gpt-5.6-luna
```

API base URL:

```text
https://openrouter.ai/api/v1
```

The API key is supplied through the environment:

```text
OPENROUTER_API_KEY
```

The API key must **never** be committed to GitHub.

---

# 13. Reasoning Rules

The Answerer follows several important constraints.

### Evidence grounding

A diagnostic candidate should be supported by at least one retrieved evidence passage.

### Evidence references

Candidates contain references to the numbered evidence items.

### No unsupported diagnosis

The Answerer should reject or omit candidates that are not supported by the retrieved evidence and clinical information.

### Similarity is not clinical confidence

Qdrant similarity scores are retrieval-ranking signals.

They are **not**:

```text
diagnostic probability
clinical confidence
disease likelihood
```

### Limitations

The system explicitly reports limitations when important clinical information is missing.

### Preliminary reasoning

The output represents preliminary evidence-based reasoning.

It is not the final clinical diagnosis.

Final synthesis belongs to the downstream Clinical Data Fusion stage.

---

# 14. Output of the Symptom RAG Analyzer

The output is structured so that another agent can consume it directly.

High-level structure:

```json
{
  "agent": "symptom_rag",
  "status": "success",
  "query_context": {},
  "diagnostic_candidates": [],
  "evidence": [],
  "metadata": {},
  "limitations": "",
  "error": null
}
```

---

# 15. Diagnostic Candidate Output

Each diagnostic candidate follows this structure:

```json
{
  "condition": "Acute coronary syndrome",
  "rank": 1,
  "justification": "Supported by the clinical presentation and retrieved evidence.",
  "supporting_evidence": [1, 3]
}
```

The candidate contains:

```text
condition
rank
justification
supporting_evidence
```

---

# 16. Evidence Output

Each retrieved evidence item follows this structure:

```json
{
  "evidence_id": 1,
  "content": "Retrieved biomedical evidence...",
  "source": "source_document",
  "source_type": "symptom-disease-dataset",
  "location": "page/chunk information",
  "relevance_score": 0.799,
  "retrieval_rank": 1
}
```

Important:

```text
relevance_score
```

is the vector retrieval similarity score.

It must not be interpreted as clinical confidence.

---

# 17. Metadata Output

The output also contains retrieval metadata.

Example:

```json
{
  "symptom_evidence_count": 5,
  "literature_evidence_count": 5,
  "total_evidence_count": 10,
  "retrieval_top_k": 5
}
```

This allows downstream agents to understand how much evidence was retrieved.

---

# 18. What the Next Agent Should Connect To

The Symptom RAG Analyzer is designed to sit between the Clinical Text Clarifier and the downstream Clinical Data Fusion Agent.

The integration flow is:

```text
Clinical Text Clarifier
        |
        | structured clinical information
        v
Symptom RAG Analyzer
        |
        | structured Symptom RAG output
        v
Clinical Data Fusion Agent
```

The integration developer should therefore connect:

```text
CTC output
    ↓
SymptomRAGOrchestrator
    ↓
Symptom RAG output adapter
    ↓
Clinical Data Fusion
```

The downstream agent should consume the structured output rather than depending on internal RAG implementation details.

---

# 19. What the Integration Developer Should NOT Modify

The following components are considered part of the frozen Symptom RAG baseline.

Do not modify them unless a deliberate architecture change is agreed upon.

### Retrieval architecture

Do not:

```text
merge symptom_chunks and literature_chunks
```

into a single collection.

Do not blindly cross-rank their raw similarity scores.

### Embedding model

Do not replace the production biomedical embedding model without re-indexing and validating the affected collections.

Current model:

```text
pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
```

### Qdrant collection names

Keep:

```text
symptom_chunks
literature_chunks
```

### Output contract

Downstream integration should use the existing output schema.

### Reasoning constraints

Do not remove the evidence-grounding and limitation rules from the Answerer.

### Graph RAG boundary

Graph RAG is a separate branch.

Do not insert Graph RAG retrieval logic inside the Vector RAG implementation.

The intended architecture is:

```text
                    Clinical Query
                         |
             +-----------+-----------+
             |                       |
             v                       v
        Vector RAG                Graph RAG
             |                       |
             +-----------+-----------+
                         |
                         v
                 Clinical Data Fusion
```

---

# 20. What Is Included in This Repository

The repository contains the source code required for the Symptom RAG Analyzer, including:

```text
src/
tests/
scripts/
requirements.txt
pyproject.toml
.env.example
README.md
```

The repository also contains the configuration and test infrastructure used during development and validation.

---

# 21. What Is NOT Included in This Repository

The following are intentionally excluded from GitHub:

```text
.env
API keys
Qdrant persistent storage
large vector databases
local Docker storage
local backups
generated runtime artifacts
raw/generated datasets where excluded by .gitignore
```

In particular:

```text
data/qdrant_server/
```

is not committed.

---

# 22. Environment Configuration

Create a local `.env` file from:

```text
.env.example
```

The relevant configuration includes:

```text
OPENROUTER_API_KEY=

NCBI_API_KEY=
NCBI_EMAIL=

PMC_REQUEST_DELAY=
```

Never commit `.env`.

---

# 23. Basic Setup

Clone the repository:

```powershell
git clone https://github.com/sricharan-2403/Symptom_RAG_Frozen.git
cd Symptom_RAG_Frozen
```

Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Configure the environment variables in:

```text
.env
```

---

# 24. Qdrant Setup for Integration

The Symptom RAG Analyzer expects Qdrant to be available at:

```text
http://localhost:6333
```

The integration machine must have a Qdrant instance running locally.

After restoring the validated production Qdrant data, verify that these collections exist:

```text
symptom_chunks
literature_chunks
```

Expected validated counts:

```text
symptom_chunks       2,690
literature_chunks    201,134
```

> The validated Qdrant production data will be provided separately to the integration developer. It is not expected to be obtained by cloning this GitHub repository.

---

# 25. Running the Symptom RAG Analyzer

Once:

```text
Python environment
+
environment variables
+
Qdrant
+
validated Qdrant collections
```

are available, the RAG pipeline can be executed using the provided tests.

The interactive tester accepts CTC-style JSON input and runs the real production retrieval and reasoning pipeline.

Example structure:

```json
{
  "clinical_text": "Patient presents with...",
  "diseases": [],
  "symptoms": [
    "symptom 1",
    "symptom 2"
  ],
  "medications": [],
  "tests": []
}
```

---

# 26. Validation Status

The current Symptom RAG Analyzer has been validated using real retrieval and Answerer execution.

The output contract has been tested for:

```text
✓ CTC input construction
✓ Biomedical query construction
✓ Symptom retrieval
✓ Literature retrieval
✓ Evidence source separation
✓ Evidence numbering
✓ Diagnostic candidate generation
✓ Evidence references
✓ Limitations field
✓ Output adapter schema
✓ Retrieval metadata
✓ End-to-end pipeline execution
```

The validated production Qdrant state is:

```text
symptom_chunks       2,690 vectors
literature_chunks    201,134 vectors
```

---

# 27. Example End-to-End Flow

Example clinical input:

```text
A 58-year-old patient presents with sudden central chest
pressure that began 45 minutes ago while resting. The discomfort
radiates to the left arm and jaw and is associated with sweating,
nausea and shortness of breath.
```

The CTC provides structured information.

The Symptom RAG Analyzer then:

```text
CTC Output
    ↓
Diseases + Symptoms
    ↓
Biomedical Embedding
    ↓
+-----------------------+
| symptom_chunks        |
| literature_chunks     |
+-----------------------+
    ↓
Top-k Evidence
    ↓
Evidence Context
    ↓
Answerer LLM
    ↓
Diagnostic Candidates
    ↓
Structured Symptom RAG Output
```

The output is then passed to the downstream Clinical Data Fusion Agent.

---

# 28. Architectural Boundary

The Symptom RAG Analyzer owns:

```text
Clinical Query Construction
Biomedical Embedding
Vector Retrieval
Evidence Organization
Evidence-Grounded Reasoning
Structured RAG Output
```

The Symptom RAG Analyzer does NOT own:

```text
Clinical Text Clarification
Graph RAG
Web Intelligence
Final Clinical Data Fusion
Final Diagnosis
```

This separation is intentional.

---

# 29. Frozen Baseline

This repository represents the **frozen baseline** of the Symptom RAG Analyzer.

The baseline includes:

```text
✓ Independent symptom and literature retrieval
✓ Biomedical embeddings
✓ Persistent Qdrant retrieval
✓ Evidence-grounded Answerer
✓ Structured output adapter
✓ End-to-end validation
✓ Production-sized Qdrant collections
```

Future changes should be treated as explicit architecture or experiment changes rather than silently modifying the frozen baseline.

---

# 30. Integration Handoff — Quick Reference

For another developer to integrate this agent, the required information is:

```text
INPUT
↓
Clinical Text Clarifier structured output

RETRIEVAL
↓
Biomedical embedding model
↓
Qdrant
↓
symptom_chunks + literature_chunks
↓
Top-5 from each collection

REASONING
↓
OpenRouter Answerer

OUTPUT
↓
Structured Symptom RAG result

NEXT AGENT
↓
Clinical Data Fusion Agent
```

Required external dependency:

```text
Qdrant Server
```

Required production collections:

```text
symptom_chunks
literature_chunks
```

Required collection counts:

```text
2,690
201,134
```

Required endpoint:

```text
http://localhost:6333
```

---

# 31. Important Integration Rule

**Do not treat this repository as a standalone final clinical diagnosis system.**

It is one agent/component inside the larger Agentic Graph-RAG Clinical Decision Support System.

Its responsibility is to retrieve biomedical evidence and produce preliminary, evidence-grounded diagnostic candidates that can be consumed by downstream agents.

```text
                  THIS REPOSITORY
                        |
                        v
              +-------------------+
              | Symptom RAG Agent |
              +-------------------+
                        |
                        v
              Preliminary Evidence
              & Diagnostic Candidates
                        |
                        v
             Clinical Data Fusion
                        |
                        v
              Final System Output
```

---

# 32. Repository

GitHub:

```text
https://github.com/sricharan-2403/Symptom_RAG_Frozen
```

This repository should be treated as the frozen integration baseline for the Symptom RAG Analyzer.
