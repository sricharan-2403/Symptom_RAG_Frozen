"""Convert internal Symptom RAG results to the common agent output contract."""

from __future__ import annotations

from typing import Any

from symptom_rag_analyzer.orchestrator import (
    ClinicalTextClarifierOutput,
    SymptomRAGResult,
)


class SymptomRAGOutputAdapter:
    """Adapt a successful internal RAG result without adding reasoning or fusion."""

    def adapt(
        self,
        clinical_context: ClinicalTextClarifierOutput,
        result: SymptomRAGResult,
    ) -> dict[str, Any]:
        """Return the JSON-compatible common output for an internal RAG result."""

        if not isinstance(
            clinical_context,
            ClinicalTextClarifierOutput,
        ):
            raise TypeError(
                "clinical_context must be a ClinicalTextClarifierOutput instance"
            )

        if not isinstance(result, SymptomRAGResult):
            raise TypeError(
                "result must be a SymptomRAGResult instance"
            )

        symptom_evidence_count = sum(
            1
            for evidence in result.evidence
            if evidence.source_type == "symptom-disease-dataset"
        )

        literature_evidence_count = sum(
            1
            for evidence in result.evidence
            if evidence.source_type == "pubmed-pmc-literature"
        )

        evidence_output = []

        symptom_rank = 0
        literature_rank = 0

        for evidence_id, evidence in enumerate(result.evidence, start=1):

            if evidence.source_type == "symptom-disease-dataset":
                symptom_rank += 1
                retrieval_rank = symptom_rank

            elif evidence.source_type == "pubmed-pmc-literature":
                literature_rank += 1
                retrieval_rank = literature_rank

            else:
                retrieval_rank = None

            evidence_output.append(
                {
                    "evidence_id": evidence_id,
                    "content": evidence.text,
                    "source": evidence.filename,
                    "source_type": evidence.source_type,
                    "location": (
                        f"Page {evidence.page_number}"
                        if evidence.page_number is not None
                        else ""
                    ),
                    "relevance_score": evidence.score,
                    "retrieval_rank": retrieval_rank,
                }
            )

        return {
            "agent": "symptom_rag",
            "status": "success",

            "query_context": {
                "diseases": list(clinical_context.diseases),
                "symptoms": list(clinical_context.symptoms),
                "medications": list(clinical_context.medications),
                "tests": list(clinical_context.tests),
                "procedures": [],
            },

            "diagnostic_candidates": [
                {
                    "condition": candidate.condition,
                    "rank": candidate.rank,
                    "justification": candidate.justification,
                    "supporting_evidence": list(
                        candidate.supporting_evidence
                    ),
                }
                for candidate in result.answer.diagnostic_candidates
            ],

            "evidence": evidence_output,

            "metadata": {
                "symptom_evidence_count": symptom_evidence_count,
                "literature_evidence_count": literature_evidence_count,
                "total_evidence_count": len(result.evidence),
                "retrieval_top_k": max(
                    symptom_evidence_count,
                    literature_evidence_count,
                    default=0,
                ),
            },

            "limitations": result.answer.limitations,
            "error": None,
        }