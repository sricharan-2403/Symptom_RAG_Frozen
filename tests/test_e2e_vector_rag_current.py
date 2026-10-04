import json

from symptom_rag_analyzer.orchestrator import (
    ClinicalTextClarifierOutput,
    SymptomRAGOrchestrator,
)


def main() -> None:
    print("=" * 80)
    print("END-TO-END VECTOR RAG TEST")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. EXACT CTC OUTPUT
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
    # 2. CONSTRUCT CLINICAL TEXT CLARIFIER OUTPUT
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

    print("\n2. CLINICAL TEXT CLARIFIER OUTPUT")
    print("-" * 80)
    print(f"Diseases:    {clinical_context.diseases}")
    print(f"Symptoms:    {clinical_context.symptoms}")
    print(f"Medications: {clinical_context.medications}")
    print(f"Tests:       {clinical_context.tests}")

    # ------------------------------------------------------------------
    # 3. INITIALIZE REAL VECTOR RAG
    # ------------------------------------------------------------------

    print("\n3. INITIALIZING VECTOR RAG")
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
    # 4. RUN COMPLETE PIPELINE
    # ------------------------------------------------------------------

    print("\n4. RUNNING COMPLETE VECTOR RAG PIPELINE")
    print("-" * 80)

    try:
        result = orchestrator.analyze(clinical_context)

    except Exception as error:
        print("\n❌ END-TO-END PIPELINE FAILED")
        print("-" * 80)
        print(f"Error type: {type(error).__name__}")
        print(f"Error: {error}")

        raise

    # ------------------------------------------------------------------
    # 5. RETRIEVAL VALIDATION
    # ------------------------------------------------------------------

    print("\n5. RETRIEVAL VALIDATION")
    print("-" * 80)

    symptom_evidence = [
        item
        for item in result.evidence
        if item.source_type == "symptom-disease-dataset"
    ]

    literature_evidence = [
        item
        for item in result.evidence
        if item.source_type == "pubmed-pmc-literature"
    ]

    print(f"Symptom evidence:    {len(symptom_evidence)}")
    print(f"Literature evidence: {len(literature_evidence)}")
    print(f"Total evidence:      {len(result.evidence)}")

    assert len(symptom_evidence) == 5
    assert len(literature_evidence) == 5
    assert len(result.evidence) == 10

    print("✓ 5 symptom results retrieved")
    print("✓ 5 literature results retrieved")
    print("✓ 10 total evidence items retrieved")

    # ------------------------------------------------------------------
    # 6. PRINT RETRIEVED EVIDENCE
    # ------------------------------------------------------------------

    print("\n6. RETRIEVED EVIDENCE")
    print("-" * 80)

    for index, evidence in enumerate(result.evidence, start=1):
        print(
            f"[Evidence {index}] "
            f"score={evidence.score:.4f} | "
            f"source={evidence.source_type} | "
            f"file={evidence.filename}"
        )

    # ------------------------------------------------------------------
    # 7. ANSWERER OUTPUT
    # ------------------------------------------------------------------

    print("\n7. ANSWERER OUTPUT")
    print("-" * 80)

    print(f"Diagnostic candidates: {len(result.answer.diagnostic_candidates)}")

    for candidate in result.answer.diagnostic_candidates:
        print(
            f"[Rank {candidate.rank}] "
            f"{candidate.condition}"
        )
        print(
            f"  Supporting evidence: "
            f"{candidate.supporting_evidence}"
        )
        print(
            f"  Justification: "
            f"{candidate.justification}"
        )

    print("\nLimitations:")
    print(result.answer.limitations)

    # ------------------------------------------------------------------
    # 8. INTERNAL RAG RESULT
    # ------------------------------------------------------------------

    print("\n8. INTERNAL RAG RESULT")
    print("-" * 80)

    print(
        json.dumps(
            {
                "diagnostic_candidates": [
                    {
                        "condition": candidate.condition,
                        "rank": candidate.rank,
                        "justification": candidate.justification,
                        "supporting_evidence": candidate.supporting_evidence,
                    }
                    for candidate in result.answer.diagnostic_candidates
                ],
                "limitations": result.answer.limitations,
                "evidence_count": len(result.evidence),
            },
            indent=2,
        )
    )

    print("\n" + "=" * 80)
    print("PASS: COMPLETE VECTOR RAG PIPELINE EXECUTED")
    print("=" * 80)


if __name__ == "__main__":
    main()