"""
OpenRouter Answerer smoke test.

Tests the real retrieval pipeline and sends the same production prompt
to an OpenRouter-hosted Answerer model.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# =============================================================================
# PROJECT PATH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

from symptom_rag_analyzer.orchestrator import (  # noqa: E402
    ClinicalTextClarifierOutput,
)
from symptom_rag_analyzer.retrieval.search import BiomedicalRetriever  # noqa: E402
from symptom_rag_analyzer.reasoning.context import EvidenceContextBuilder  # noqa: E402
from symptom_rag_analyzer.reasoning.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    build_prompt,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

load_dotenv(PROJECT_ROOT / ".env")

MODEL = "openai/gpt-5.6-luna"


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 80)
    print("OPENROUTER ANSWERER SMOKE TEST")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Check OpenRouter API key
    # -------------------------------------------------------------------------

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY was not found in the environment."
        )

    print("\n✓ OpenRouter API key detected")

    # -------------------------------------------------------------------------
    # 2. Exact CTC output used by the current project
    # -------------------------------------------------------------------------

    clinical_context = ClinicalTextClarifierOutput(
        clinical_text=(
            "A patient presents with chest pain, sweating, and "
            "shortness of breath. The patient has a history of "
            "myocardial infarction and is taking Sildenafil."
        ),
        diseases=["Myocardial Infarction"],
        symptoms=[
            "chest pain",
            "sweating",
            "shortness of breath",
        ],
        medications=["Sildenafil"],
        tests=[],
    )

    print("\n✓ Clinical Text Clarifier output constructed")

    # -------------------------------------------------------------------------
    # 3. Initialize real biomedical retriever
    # -------------------------------------------------------------------------

    retriever = BiomedicalRetriever()

    print("✓ BiomedicalRetriever initialized")

    # -------------------------------------------------------------------------
    # 4. Build retrieval query
    #
    # Same decision as the production RAG pipeline:
    # diseases + symptoms are used for vector retrieval.
    # -------------------------------------------------------------------------

    retrieval_query = "\n".join(
        [
            f"Diseases: {', '.join(clinical_context.diseases)}",
            f"Symptoms: {', '.join(clinical_context.symptoms)}",
        ]
    )

    print("\nRetrieval query:")
    print("-" * 80)
    print(retrieval_query)
    print("-" * 80)

    # -------------------------------------------------------------------------
    # 5. Real hybrid vector retrieval
    # -------------------------------------------------------------------------

    vector_result = retriever.retrieve(
        retrieval_query,
        top_k=5,
    )

    retrieved_evidence = vector_result.all_evidence

    print("\n✓ Retrieval completed")
    print(f"  Symptom evidence: {len(vector_result.symptom_evidence)}")
    print(f"  Literature evidence: {len(vector_result.literature_evidence)}")
    print(f"  Total evidence: {len(retrieved_evidence)}")

    # -------------------------------------------------------------------------
    # 6. Build production evidence context
    # -------------------------------------------------------------------------

    context_builder = EvidenceContextBuilder()

    evidence_context = context_builder.build_context(
        retrieved_evidence
    )

    print("\n✓ Evidence context built")
    print(f"  Context length: {len(evidence_context)} characters")

    # -------------------------------------------------------------------------
    # 7. Build exact production Answerer prompt
    # -------------------------------------------------------------------------

    user_prompt = build_prompt(
        clinical_context.clinical_text,
        evidence_context,
    )

    print("✓ Production Answerer prompt built")

    # -------------------------------------------------------------------------
    # 8. Initialize OpenRouter client
    # -------------------------------------------------------------------------

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    print("✓ OpenRouter client initialized")

    # -------------------------------------------------------------------------
    # 9. Call OpenRouter Answerer
    # -------------------------------------------------------------------------

    print("\n6. CALLING OPENROUTER ANSWERER")
    print("-" * 80)

    response = client.chat.completions.create(
        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        # 4096 is more than sufficient for the structured RAG response.
        max_tokens=4096,

        temperature=0,

        # ---------------------------------------------------------------------
        # Structured JSON output
        # ---------------------------------------------------------------------

        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "rag_answer",
                "strict": True,

                "schema": {
                    "type": "object",

                    "properties": {

                        "diagnostic_candidates": {
                            "type": "array",

                            "items": {
                                "type": "object",

                                "properties": {

                                    "condition": {
                                        "type": "string"
                                    },

                                    "rank": {
                                        "type": "integer"
                                    },

                                    "justification": {
                                        "type": "string"
                                    },

                                    "supporting_evidence": {
                                        "type": "array",

                                        "items": {
                                            "type": "integer"
                                        }
                                    },
                                },

                                "required": [
                                    "condition",
                                    "rank",
                                    "justification",
                                    "supporting_evidence",
                                ],

                                "additionalProperties": False,
                            },
                        },

                        "limitations": {
                            "type": "string"
                        },
                    },

                    "required": [
                        "diagnostic_candidates",
                        "limitations",
                    ],

                    "additionalProperties": False,
                },
            },
        },

        # ---------------------------------------------------------------------
        # IMPORTANT CHANGE
        #
        # Previously:
        #
        #     require_parameters=True
        #
        # That caused OpenRouter to reject the request because the available
        # provider endpoints did not all satisfy the requested parameters.
        #
        # We now allow OpenRouter to route to a compatible endpoint.
        # ---------------------------------------------------------------------

        extra_body={
            "provider": {
                "require_parameters": False,
            }
        },
    )

    # -------------------------------------------------------------------------
    # 10. Extract response
    # -------------------------------------------------------------------------

    raw_content = response.choices[0].message.content

    if not raw_content:
        raise RuntimeError(
            "OpenRouter returned an empty response."
        )

    print("\n✓ OpenRouter Answerer succeeded")

    # -------------------------------------------------------------------------
    # 11. Parse JSON
    # -------------------------------------------------------------------------

    try:
        answer = json.loads(raw_content)

    except json.JSONDecodeError as exc:

        print("\nRaw response:")
        print(raw_content)

        raise RuntimeError(
            "OpenRouter returned invalid JSON."
        ) from exc

    # -------------------------------------------------------------------------
    # 12. Validate top-level output
    # -------------------------------------------------------------------------

    if not isinstance(answer, dict):
        raise RuntimeError(
            "Answerer output must be a JSON object."
        )

    if "diagnostic_candidates" not in answer:
        raise RuntimeError(
            "Missing 'diagnostic_candidates' in Answerer output."
        )

    if "limitations" not in answer:
        raise RuntimeError(
            "Missing 'limitations' in Answerer output."
        )

    candidates = answer["diagnostic_candidates"]

    if not isinstance(candidates, list):
        raise RuntimeError(
            "'diagnostic_candidates' must be a list."
        )

    # -------------------------------------------------------------------------
    # 13. Validate candidate structure
    # -------------------------------------------------------------------------

    for candidate in candidates:

        required_fields = {
            "condition",
            "rank",
            "justification",
            "supporting_evidence",
        }

        missing = required_fields - set(candidate.keys())

        if missing:
            raise RuntimeError(
                f"Candidate missing fields: {sorted(missing)}"
            )

        if not isinstance(candidate["supporting_evidence"], list):
            raise RuntimeError(
                "'supporting_evidence' must be a list."
            )

        # Validate that every evidence reference points to one of the
        # 10 retrieved evidence items.
        for evidence_id in candidate["supporting_evidence"]:

            if not isinstance(evidence_id, int):
                raise RuntimeError(
                    "Supporting evidence IDs must be integers."
                )

            if not 1 <= evidence_id <= len(retrieved_evidence):
                raise RuntimeError(
                    f"Invalid evidence ID: {evidence_id}. "
                    f"Valid range: 1-{len(retrieved_evidence)}"
                )

    # -------------------------------------------------------------------------
    # 14. Print final result
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("OPENROUTER ANSWERER RESULT")
    print("=" * 80)

    print(
        json.dumps(
            answer,
            indent=2,
            ensure_ascii=False,
        )
    )

    # -------------------------------------------------------------------------
    # 15. Final validation summary
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    print("✓ JSON response received")
    print("✓ diagnostic_candidates present")
    print("✓ limitations present")
    print("✓ Candidate schema valid")
    print("✓ Supporting evidence references valid")
    print("✓ OpenRouter Answerer smoke test PASSED")

    print("=" * 80)


if __name__ == "__main__":
    main()