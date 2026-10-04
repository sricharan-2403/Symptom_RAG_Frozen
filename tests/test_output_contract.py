"""End-to-end validation of the Symptom RAG output contract."""

from __future__ import annotations

import json

from symptom_rag_analyzer.orchestrator import (
    ClinicalTextClarifierOutput,
    SymptomRAGOrchestrator,
)
from symptom_rag_analyzer.output_adapter import SymptomRAGOutputAdapter


def main() -> None:
    print("=" * 70)
    print("SYMPTOM RAG OUTPUT CONTRACT TEST")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Exact CTC test case already used in our retrieval/Answerer tests
    # ------------------------------------------------------------------
    clinical_context = ClinicalTextClarifierOutput(
        clinical_text=(
            "Patient has myocardial infarction with chest pain, "
            "sweating, and shortness of breath. "
            "Patient is taking Sildenafil."
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

    # ------------------------------------------------------------------
    # 2. Run the REAL RAG pipeline
    # ------------------------------------------------------------------
    orchestrator = SymptomRAGOrchestrator(top_k=5)

    print("✓ SymptomRAGOrchestrator initialized")
    print("\nRunning real retrieval + OpenRouter Answerer...")

    result = orchestrator.analyze(clinical_context)

    print("✓ RAG pipeline completed")
    print(f"  Evidence returned: {len(result.evidence)}")
    print(
        f"  Diagnostic candidates: "
        f"{len(result.answer.diagnostic_candidates)}"
    )

    # ------------------------------------------------------------------
    # 3. Adapt internal result to external agent contract
    # ------------------------------------------------------------------
    adapter = SymptomRAGOutputAdapter()

    output = adapter.adapt(
        clinical_context=clinical_context,
        result=result,
    )

    print("✓ Output adapter completed")

    # ------------------------------------------------------------------
    # 4. Validate top-level contract
    # ------------------------------------------------------------------
    assert output["agent"] == "symptom_rag"
    assert output["status"] == "success"
    assert output["error"] is None

    print("✓ Top-level contract valid")

    # ------------------------------------------------------------------
    # 5. Validate query context
    # ------------------------------------------------------------------
    query_context = output["query_context"]

    assert query_context["diseases"] == ["Myocardial Infarction"]

    assert query_context["symptoms"] == [
        "chest pain",
        "sweating",
        "shortness of breath",
    ]

    assert query_context["medications"] == ["Sildenafil"]
    assert query_context["tests"] == []
    assert query_context["procedures"] == []

    print("✓ Query context preserved correctly")

    # ------------------------------------------------------------------
    # 6. Validate hybrid evidence counts
    # ------------------------------------------------------------------
    metadata = output["metadata"]

    assert metadata["symptom_evidence_count"] == 5
    assert metadata["literature_evidence_count"] == 5
    assert metadata["total_evidence_count"] == 10
    assert metadata["retrieval_top_k"] == 5

    print("✓ Evidence counts are correct")
    print("  Symptom evidence   : 5")
    print("  Literature evidence: 5")
    print("  Total evidence     : 10")
    print("  Retrieval top-k    : 5")

    # ------------------------------------------------------------------
    # 7. Validate evidence IDs
    # ------------------------------------------------------------------
    evidence = output["evidence"]

    assert len(evidence) == 10

    evidence_ids = [
        item["evidence_id"]
        for item in evidence
    ]

    assert evidence_ids == list(range(1, 11))

    print("✓ Evidence IDs are sequential: 1..10")

    # ------------------------------------------------------------------
    # 8. Validate source separation and retrieval ranks
    # ------------------------------------------------------------------
    symptom_evidence = [
        item
        for item in evidence
        if item["source_type"] == "symptom-disease-dataset"
    ]

    literature_evidence = [
        item
        for item in evidence
        if item["source_type"] == "pubmed-pmc-literature"
    ]

    assert len(symptom_evidence) == 5
    assert len(literature_evidence) == 5

    symptom_ranks = [
        item["retrieval_rank"]
        for item in symptom_evidence
    ]

    literature_ranks = [
        item["retrieval_rank"]
        for item in literature_evidence
    ]

    assert symptom_ranks == [1, 2, 3, 4, 5]
    assert literature_ranks == [1, 2, 3, 4, 5]

    print("✓ Source separation preserved")
    print("  Symptom ranks   : 1..5")
    print("  Literature ranks: 1..5")

    # ------------------------------------------------------------------
    # 9. Validate evidence fields
    # ------------------------------------------------------------------
    required_evidence_fields = {
        "evidence_id",
        "content",
        "source",
        "source_type",
        "location",
        "relevance_score",
        "retrieval_rank",
    }

    for item in evidence:
        assert required_evidence_fields.issubset(item.keys())
        assert isinstance(item["evidence_id"], int)
        assert isinstance(item["content"], str)
        assert isinstance(item["source"], str)
        assert isinstance(item["source_type"], str)
        assert isinstance(item["relevance_score"], float)
        assert item["retrieval_rank"] in [1, 2, 3, 4, 5]

    print("✓ Evidence object schema valid")

    # ------------------------------------------------------------------
    # 10. Validate diagnostic candidates
    # ------------------------------------------------------------------
    candidates = output["diagnostic_candidates"]

    assert isinstance(candidates, list)

    valid_evidence_ids = set(evidence_ids)

    for candidate in candidates:
        assert "condition" in candidate
        assert "rank" in candidate
        assert "justification" in candidate
        assert "supporting_evidence" in candidate

        assert isinstance(candidate["condition"], str)
        assert isinstance(candidate["rank"], int)
        assert isinstance(candidate["justification"], str)

        for evidence_id in candidate["supporting_evidence"]:
            assert evidence_id in valid_evidence_ids

    print("✓ Diagnostic candidate schema valid")
    print(f"  Candidates returned: {len(candidates)}")

    # ------------------------------------------------------------------
    # 11. Validate limitations
    # ------------------------------------------------------------------
    assert isinstance(output["limitations"], str)
    assert output["limitations"].strip()

    print("✓ Limitations field present")

    # ------------------------------------------------------------------
    # 12. Print final JSON contract
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL RAG OUTPUT")
    print("=" * 70)

    print(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
    )

    # ------------------------------------------------------------------
    # 13. Final success marker
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ALL OUTPUT CONTRACT CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()