# Symptom RAG Analyzer

Biomedical RAG analyzer for clinical decision support, combining biomedical retrieval, persistent Qdrant knowledge bases, evidence construction, and LLM-based reasoning.

## Requirements

- Python 3.10+
- Pinned dependencies in `requirements.txt`
- Runtime artifacts from the shared Google Drive folder
- A Gemini API key

## Installation

```powershell
git clone https://github.com/sricharan-2403/Symptom_RAG_Analyzer.git
cd Symptom_RAG_Analyzer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Runtime Artifacts

Large runtime data is intentionally not stored in GitHub. Copy the contents of the shared Drive folder `Symptom_RAG_Analyser_Runtime_Artifacts` into the project so the layout looks like this:

```
data/
├── qdrant/
├── processed/
└── raw/
    └── pmc/
```

**Drive mapping:**

| Drive folder  | Project folder   |
|---------------|------------------|
| `qdrant/`     | `data/qdrant/`   |
| `processed/`  | `data/processed/`|
| `pmc/`        | `data/raw/pmc/`  |

The persistent Qdrant database contains the following collections:

- `symptom_chunks`
- `literature_chunks`
- `literature_chunks_benchmark`

`literature_chunks` contains the biomedical literature collection.

## Environment Variable

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_key_here
```

> **Never commit the actual API key.** `.env` is ignored by Git.

## Run the Real RAG Example

From the project root:

```powershell
python tests/test_e2e_real_rag.py
```

This runs the end-to-end pipeline against the persistent `data/qdrant/` knowledge base using the `symptom_chunks` collection.

### Example Input

The real E2E test uses:

```json
{
  "diseases": ["Myocardial Infarction"],
  "symptoms": ["chest pain", "sweating", "shortness of breath"],
  "medications": ["Sildenafil"],
  "tests": []
}
```

### Output

The output adapter produces JSON containing fields including:

- `agent`
- `status`
- `query_context`
- `diagnostic_candidates`
- `evidence`
- `error`
- `limitations`
- `metadata`

## Current LLM Configuration

- **Provider:** Google Gemini
- **Model:** `gemini-3.6-flash`
- **API key:** `GEMINI_API_KEY`