import json

from symptom_rag_analyzer.orchestrator import (
    ClinicalTextClarifierOutput,
    SymptomRAGOrchestrator,
)


def main() -> None:
    print("=" * 80)
    print("ANSWERER SMOKE TEST — CURRENT HYBRID VECTOR RAG")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. EXACT CURRENT CTC OUTPUT
    # ------------------------------------------------------------------
    ctc_output = {
        "diseases": [
            "Myocardial Infarction"
        ],
        "symptoms": [
            "chest pain",
            "sweating",
            "shortness of breath"
        ],
        "medications": [
            "Sildenafil"
        ],
        "tests": [],
        "procedures": []
    }

    print("\n1. CTC OUTPUT")
    print("-" * 80)
    print(json.dumps(ctc_output, indent=2))

    # ------------------------------------------------------------------
    # 2. BUILD CLINICAL CONTEXT
    # ------------------------------------------------------------------
    clinical_context = ClinicalTextClarifierOutput(
        clinical_text=(
            "Patient with Myocardial Infarction presenting with "
            "chest pain, sweating, and shortness of breath. "
            "Current medication: Sildenafil."
        ),
        diseases=ctc_output["diseases"],
        symptoms=ctc_output["symptoms"],
        medications=ctc_output["medications"],
        tests=ctc_output["tests"],
    )

    print("\n2. CLINICAL CONTEXT")
    print("-" * 80)
    print(f"Diseases:    {clinical_context.diseases}")
    print(f"Symptoms:    {clinical_context.symptoms}")
    print(f"Medications: {clinical_context.medications}")
    print(f"Tests:       {clinical_context.tests}")

    # ------------------------------------------------------------------
    # 3. INITIALIZE REAL ORCHESTRATOR
    # ------------------------------------------------------------------
    print("\n3. INITIALIZING REAL VECTOR RAG")
    print("-" * 80)

    orchestrator = SymptomRAGOrchestrator(
        top_k=5,
    )

    print("✓ BiomedicalRetriever initialized")
    print("✓ symptom_chunks configured")
    print("✓ literature_chunks configured")
    print("✓ EvidenceContextBuilder initialized")
    print("✓ BiomedicalAnswerer initialized")

    # ------------------------------------------------------------------
    # 4. RETRIEVE REAL EVIDENCE
    # ------------------------------------------------------------------
    print("\n4. RUNNING RETRIEVAL")
    print("-" * 80)

    query = orchestrator._build_retrieval_query(
        clinical_context
    )

    print("Retrieval query:")
    print(query)

    vector_result = orchestrator.retriever.retrieve(
        query,
        top_k=5,
    )

    print("\n✓ Retrieval completed")
    print(f"Symptom evidence: {len(vector_result.symptom_evidence)}")
    print(f"Literature evidence: {len(vector_result.literature_evidence)}")
    print(f"Total evidence: {len(vector_result.all_evidence)}")

    assert len(vector_result.symptom_evidence) == 5
    assert len(vector_result.literature_evidence) == 5

    # ------------------------------------------------------------------
    # 5. BUILD REAL EVIDENCE CONTEXT
    # ------------------------------------------------------------------
    print("\n5. BUILDING EVIDENCE CONTEXT")
    print("-" * 80)

    evidence_context = orchestrator.context_builder.build_context(
        vector_result.all_evidence
    )

    print("✓ Evidence context built")
    print(f"Context length: {len(evidence_context)} characters")

    assert "SYMPTOM-DISEASE DATABASE EVIDENCE" in evidence_context
    assert "BIOMEDICAL LITERATURE EVIDENCE" in evidence_context
    assert "[Evidence 1]" in evidence_context
    assert "[Evidence 10]" in evidence_context

    # ------------------------------------------------------------------
    # 6. CALL REAL GEMINI ANSWERER
    # ------------------------------------------------------------------
    print("\n6. CALLING REAL GEMINI ANSWERER")
    print("-" * 80)
    print("This is the only external LLM call in this test.")
    print("Waiting for Gemini response...\n")

    try:
        answer = orchestrator.answerer.answer(
            clinical_context.clinical_text,
            evidence_context,
        )

    except Exception as exc:
        print("\n" + "=" * 80)
        print("❌ ANSWERER FAILED")
        print("=" * 80)

        print(f"\nError type: {type(exc).__name__}")
        print(f"Error: {exc}")

        print("\nIMPORTANT:")
        print("Retrieval and evidence-context construction already passed.")
        print("Therefore this failure occurred during the Answerer/Gemini call.")

        raise

    # ------------------------------------------------------------------
    # 7. DISPLAY STRUCTURED ANSWER
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("✓ GEMINI ANSWERER SUCCEEDED")
    print("=" * 80)

    print("\n7. RAW ANSWER OBJECT")
    print("-" * 80)

    print(answer)

    print("\n8. DIAGNOSTIC CANDIDATES")
    print("-" * 80)

    for candidate in answer.diagnostic_candidates:
        print(f"Rank: {candidate.rank}")
        print(f"Condition: {candidate.condition}")
        print(f"Justification: {candidate.justification}")
        print(
            f"Supporting evidence: "
            f"{candidate.supporting_evidence}"
        )
        print()

    print("\n9. LIMITATIONS")
    print("-" * 80)
    print(answer.limitations)

    print("\n" + "=" * 80)
    print("PASS: REAL ANSWERER COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()