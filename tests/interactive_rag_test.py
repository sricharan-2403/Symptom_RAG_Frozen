"""Interactive clinical testing harness for the Symptom RAG Analyzer."""

from __future__ import annotations

import json
from typing import Any

from symptom_rag_analyzer.orchestrator import (
    ClinicalTextClarifierOutput,
    SymptomRAGOrchestrator,
)
from symptom_rag_analyzer.output_adapter import SymptomRAGOutputAdapter


def print_separator(char: str = "=", width: int = 78) -> None:
    """Print a visual separator."""
    print(char * width)


def print_json(data: Any) -> None:
    """Print JSON in a readable format."""
    print(json.dumps(data, indent=2, ensure_ascii=False))


def build_clinical_context(
    payload: dict[str, Any],
) -> ClinicalTextClarifierOutput:
    """Convert interactive JSON into the internal CTC output structure."""

    required_fields = [
        "clinical_text",
        "diseases",
        "symptoms",
        "medications",
        "tests",
    ]

    for field in required_fields:
        if field not in payload:
            raise ValueError(f"Missing required field: '{field}'")

    return ClinicalTextClarifierOutput(
        clinical_text=payload["clinical_text"],
        diseases=payload["diseases"],
        symptoms=payload["symptoms"],
        medications=payload["medications"],
        tests=payload["tests"],
    )


def print_retrieval_results(
    symptom_evidence: list,
    literature_evidence: list,
) -> None:
    """Display retrieved evidence while preserving source separation."""

    print_separator("-")
    print("RETRIEVED SYMPTOM-DISEASE EVIDENCE")
    print_separator("-")

    for rank, evidence in enumerate(symptom_evidence, start=1):
        print(f"\n[{rank}] Score: {evidence.score:.6f}")
        print(f"Source: {evidence.filename}")
        print(f"Text: {evidence.text}")

    print_separator("-")
    print("RETRIEVED BIOMEDICAL LITERATURE EVIDENCE")
    print_separator("-")

    for rank, evidence in enumerate(literature_evidence, start=1):
        print(f"\n[{rank}] Score: {evidence.score:.6f}")
        print(f"Source: {evidence.filename}")
        print(f"Text: {evidence.text}")


def print_diagnostic_candidates(answer) -> None:
    """Display Answerer diagnostic candidates."""

    print_separator("-")
    print("DIAGNOSTIC CANDIDATES")
    print_separator("-")

    if not answer.diagnostic_candidates:
        print("No diagnostic candidates returned.")
        return

    for candidate in answer.diagnostic_candidates:
        print(f"\nRank {candidate.rank}: {candidate.condition}")
        print(f"Justification: {candidate.justification}")
        print(
            "Supporting evidence: "
            + ", ".join(map(str, candidate.supporting_evidence))
        )


def print_limitations(answer) -> None:
    """Display Answerer limitations."""

    print_separator("-")
    print("LIMITATIONS")
    print_separator("-")
    print(answer.limitations)


def run_case(
    orchestrator: SymptomRAGOrchestrator,
    adapter: SymptomRAGOutputAdapter,
    payload: dict[str, Any],
) -> None:
    """Run one complete clinical case through the real RAG pipeline."""

    clinical_context = build_clinical_context(payload)

    print_separator()
    print("CLINICAL INPUT")
    print_separator()
    print_json(payload)

    retrieval_query = orchestrator._build_retrieval_query(
        clinical_context
    )

    print_separator("-")
    print("RETRIEVAL QUERY")
    print_separator("-")
    print(retrieval_query)

    print_separator("-")
    print("RUNNING REAL RAG PIPELINE...")
    print_separator("-")

    # Run the production pipeline exactly once.
    result = orchestrator.analyze(clinical_context)

    # Separate the evidence returned by that single retrieval operation.
    symptom_evidence = [
        evidence
        for evidence in result.evidence
        if evidence.source_type == "symptom-disease-dataset"
    ]

    literature_evidence = [
        evidence
        for evidence in result.evidence
        if evidence.source_type == "pubmed-pmc-literature"
    ]

    print("✓ RAG pipeline completed")
    print(f"  Symptom evidence      : {len(symptom_evidence)}")
    print(f"  Literature evidence   : {len(literature_evidence)}")
    print(f"  Total evidence        : {len(result.evidence)}")
    print(
        f"  Diagnostic candidates : "
        f"{len(result.answer.diagnostic_candidates)}"
    )

    print_retrieval_results(
        symptom_evidence,
        literature_evidence,
    )

    print_diagnostic_candidates(result.answer)

    print_limitations(result.answer)

    # Convert the internal result to the common agent contract.
    final_output = adapter.adapt(
        clinical_context,
        result,
    )

    print_separator("-")
    print("FINAL AGENT OUTPUT")
    print_separator("-")
    print_json(final_output)

    print_separator()
    print("CASE COMPLETED")
    print_separator()


def main() -> None:
    """Run the interactive clinical testing loop."""

    print_separator()
    print("SYMPTOM RAG — INTERACTIVE CLINICAL TESTER")
    print_separator()

    print(
        """
This tester runs the REAL production pipeline:

Clinical Text Clarifier output
        ↓
Disease + Symptom retrieval query
        ↓
Qdrant symptom_chunks + literature_chunks
        ↓
Evidence Context Builder
        ↓
OpenRouter Answerer
        ↓
Output Adapter
        ↓
Common Symptom-RAG Agent Output

Enter 'exit' at any time to stop.
"""
    )

    print_separator()

    print("Initializing Symptom RAG Orchestrator...")

    orchestrator = SymptomRAGOrchestrator(top_k=5)
    adapter = SymptomRAGOutputAdapter()

    print("✓ Orchestrator initialized")
    print()

    while True:
        print_separator()
        print("NEW CLINICAL CASE")
        print_separator()

        print("Enter the clinical case as a JSON object.")
        print(
            'Example: {"clinical_text": "...", '
            '"diseases": [], '
            '"symptoms": [], '
            '"medications": [], '
            '"tests": []}'
        )
        print()

        raw_input = input("JSON > ").strip()

        if raw_input.lower() in {"exit", "quit", "q"}:
            print("\nExiting interactive tester.")
            break

        if not raw_input:
            print("Please enter a JSON object.")
            continue

        try:
            payload = json.loads(raw_input)

            if not isinstance(payload, dict):
                raise ValueError("Input must be a JSON object.")

            run_case(
                orchestrator,
                adapter,
                payload,
            )

        except json.JSONDecodeError as error:
            print(f"\n✗ Invalid JSON: {error}")

        except Exception as error:
            print(f"\n✗ Clinical case failed: {error}")
            print(
                "The interactive tester caught the error and is "
                "ready for another case."
            )


if __name__ == "__main__":
    main()