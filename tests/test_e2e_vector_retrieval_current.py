import json

from symptom_rag_analyzer.orchestrator import (
    ClinicalTextClarifierOutput,
    SymptomRAGOrchestrator,
)


def main() -> None:
    print("=" * 80)
    print("END-TO-END VECTOR RETRIEVAL TEST")
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
    # 3. INITIALIZE RETRIEVER ONLY
    # ------------------------------------------------------------------

    print("\n3. INITIALIZING BIOMEDICAL RETRIEVER")
    print("-" * 80)

    orchestrator = SymptomRAGOrchestrator(
        top_k=5,
    )

    print("✓ Retriever initialized")

    # ------------------------------------------------------------------
    # 4. BUILD EXACT RETRIEVAL QUERY
    # ------------------------------------------------------------------

    query = orchestrator._build_retrieval_query(clinical_context)

    print("\n4. RETRIEVAL QUERY")
    print("-" * 80)
    print(query)

    # ------------------------------------------------------------------
    # 5. RETRIEVE FROM BOTH COLLECTIONS
    # ------------------------------------------------------------------

    print("\n5. RUNNING HYBRID VECTOR RETRIEVAL")
    print("-" * 80)

    vector_result = orchestrator.retriever.retrieve(
        query,
        top_k=5,
    )

    print("✓ Query embedded")
    print("✓ Symptom collection searched")
    print("✓ Literature collection searched")

    # ------------------------------------------------------------------
    # 6. VALIDATE COUNTS
    # ------------------------------------------------------------------

    print("\n6. RETRIEVAL VALIDATION")
    print("-" * 80)

    symptom_evidence = vector_result.symptom_evidence
    literature_evidence = vector_result.literature_evidence

    print(f"Symptom evidence:    {len(symptom_evidence)}")
    print(f"Literature evidence: {len(literature_evidence)}")
    print(f"Total evidence:      {len(vector_result.all_evidence)}")

    assert len(symptom_evidence) == 5
    assert len(literature_evidence) == 5
    assert len(vector_result.all_evidence) == 10

    # ------------------------------------------------------------------
    # 7. VALIDATE SOURCE SEPARATION
    # ------------------------------------------------------------------

    assert all(
        item.source_type == "symptom-disease-dataset"
        for item in symptom_evidence
    )

    assert all(
        item.source_type == "pubmed-pmc-literature"
        for item in literature_evidence
    )

    print("✓ 5 symptom-disease results")
    print("✓ 5 biomedical-literature results")
    print("✓ 10 total results")
    print("✓ Source separation preserved")

    # ------------------------------------------------------------------
    # 8. PRINT RETRIEVED EVIDENCE
    # ------------------------------------------------------------------

    print("\n7. SYMPTOM-DISEASE EVIDENCE")
    print("-" * 80)

    for index, evidence in enumerate(symptom_evidence, start=1):
        print(
            f"[Symptom {index}] "
            f"score={evidence.score:.4f}"
        )
        print(f"Source: {evidence.filename}")
        print(f"Text: {evidence.text[:300]}")
        print()

    print("\n8. BIOMEDICAL LITERATURE EVIDENCE")
    print("-" * 80)

    for index, evidence in enumerate(literature_evidence, start=1):
        print(
            f"[Literature {index}] "
            f"score={evidence.score:.4f}"
        )
        print(f"Source: {evidence.filename}")
        print(f"Text: {evidence.text[:300]}")
        print()

    # ------------------------------------------------------------------
    # 9. BUILD FINAL EVIDENCE CONTEXT
    # ------------------------------------------------------------------

    print("\n9. BUILDING EVIDENCE CONTEXT")
    print("-" * 80)

    evidence_context = orchestrator.context_builder.build_context(
        vector_result.all_evidence
    )

    print("✓ EvidenceContextBuilder completed")
    print(f"Context length: {len(evidence_context)} characters")

    assert "SYMPTOM-DISEASE DATABASE EVIDENCE" in evidence_context
    assert "BIOMEDICAL LITERATURE EVIDENCE" in evidence_context
    assert "[Evidence 1]" in evidence_context
    assert "[Evidence 10]" in evidence_context

    print("✓ Symptom section present")
    print("✓ Literature section present")
    print("✓ Evidence numbering present")

    # ------------------------------------------------------------------
    # 10. FINAL PASS
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PASS: EXACT CTC INPUT → HYBRID VECTOR RETRIEVAL VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    main()