"""Build prompt context from retrieved evidence."""

from symptom_rag_analyzer.retrieval.search import RetrievedEvidence


class EvidenceContextBuilder:
	"""Format retrieved evidence as deterministic prompt context."""

	def build_context(self, evidence: list[RetrievedEvidence]) -> str:
		"""
		Return formatted evidence grouped by source type with global numbering.

		Symptom-database evidence and biomedical-literature evidence remain
		explicitly separated, while evidence numbers remain globally unique
		across both sections.
		"""
		if not evidence:
			return ""

		symptom_evidence = [
			item
			for item in evidence
			if item.source_type == "symptom-disease-dataset"
		]

		literature_evidence = [
			item
			for item in evidence
			if item.source_type == "pubmed-pmc-literature"
		]

		sections = []
		next_evidence_number = 1

		if symptom_evidence:
			section, next_evidence_number = self._build_section(
				"SYMPTOM-DISEASE DATABASE EVIDENCE",
				symptom_evidence,
				next_evidence_number,
			)
			sections.append(section)

		if literature_evidence:
			section, next_evidence_number = self._build_section(
				"BIOMEDICAL LITERATURE EVIDENCE",
				literature_evidence,
				next_evidence_number,
			)
			sections.append(section)

		return "\n\n".join(sections)

	@staticmethod
	def _build_section(
		title: str,
		evidence: list[RetrievedEvidence],
		start_number: int,
	) -> tuple[str, int]:
		"""Build one evidence section and return the next global number."""

		blocks = [f"=== {title} ==="]

		evidence_number = start_number

		for item in evidence:
			blocks.append(
				"\n".join(
					[
						f"[Evidence {evidence_number}]",
						f"Source: {item.filename}",
						f"Source type: {item.source_type}",
						f"Page: {item.page_number}",
						f"Similarity: {item.score}",
						f"Text: {item.text}",
					]
				)
			)

			evidence_number += 1

		return "\n\n".join(blocks), evidence_number