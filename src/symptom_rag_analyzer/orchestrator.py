"""Application orchestration for the Symptom RAG Analyzer."""

from __future__ import annotations

from dataclasses import dataclass

from symptom_rag_analyzer.reasoning.answerer import BiomedicalAnswerer, RAGAnswer
from symptom_rag_analyzer.reasoning.context import EvidenceContextBuilder
from symptom_rag_analyzer.retrieval.search import BiomedicalRetriever, RetrievedEvidence


@dataclass
class ClinicalTextClarifierOutput:
	"""Structured clinical information produced by the Clinical Text Clarifier."""

	clinical_text: str
	diseases: list[str]
	symptoms: list[str]
	medications: list[str]
	tests: list[str]

	def __post_init__(self) -> None:
		if not isinstance(self.clinical_text, str):
			raise TypeError("clinical_text must be a string")
		if not self.clinical_text.strip():
			raise ValueError("clinical_text cannot be empty or whitespace")

		for field_name in ("diseases", "symptoms", "medications", "tests"):
			values = getattr(self, field_name)
			if not isinstance(values, list):
				raise TypeError(f"{field_name} must be a list of strings")
			if any(not isinstance(value, str) for value in values):
				raise TypeError(f"{field_name} must be a list of strings")


@dataclass
class SymptomRAGResult:
	"""Internal RAG result containing the Answerer output and retrieved evidence."""

	answer: RAGAnswer
	evidence: list[RetrievedEvidence]


class SymptomRAGOrchestrator:
	"""Coordinate retrieval, evidence formatting, and preliminary reasoning."""

	def __init__(
		self,
		retriever: BiomedicalRetriever | None = None,
		context_builder: EvidenceContextBuilder | None = None,
		answerer: BiomedicalAnswerer | None = None,
		top_k: int = 5,
	) -> None:
		if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
			raise ValueError("top_k must be a positive integer")

		self.retriever = retriever if retriever is not None else BiomedicalRetriever()
		self.context_builder = (
			context_builder if context_builder is not None else EvidenceContextBuilder()
		)
		self.answerer = answerer if answerer is not None else BiomedicalAnswerer()
		self.top_k = top_k

	def analyze(self, clinical_context: ClinicalTextClarifierOutput) -> SymptomRAGResult:
		"""Run the RAG pipeline for structured Clinical Text Clarifier output."""
		if not isinstance(clinical_context, ClinicalTextClarifierOutput):
			raise TypeError(
				"clinical_context must be a ClinicalTextClarifierOutput instance"
			)

		query = self._build_retrieval_query(clinical_context)

		vector_result = self.retriever.retrieve(
			query,
			top_k=self.top_k,
		)

		retrieved_evidence = vector_result.all_evidence

		evidence_context = self.context_builder.build_context(
			retrieved_evidence,
		)

		answer = self.answerer.answer(
			clinical_context.clinical_text,
			evidence_context,
		)

		return SymptomRAGResult(
			answer=answer,
			evidence=retrieved_evidence,
		)

	@staticmethod
	def _build_retrieval_query(clinical_context: ClinicalTextClarifierOutput) -> str:
		"""Build a deterministic query with diseases and symptoms first."""
		sections = [
			f"Diseases: {', '.join(clinical_context.diseases)}",
			f"Symptoms: {', '.join(clinical_context.symptoms)}",
		]
		return "\n".join(sections)