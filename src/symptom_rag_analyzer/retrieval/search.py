"""Application-level hybrid vector retrieval for biomedical evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient

from symptom_rag_analyzer.embeddings.model import BiomedicalEmbeddingModel
from symptom_rag_analyzer.vector_store.qdrant_store import QdrantVectorStore


@dataclass
class RetrievedEvidence:
	"""A retrieved chunk with its similarity score and source information."""

	text: str
	score: float
	filename: str
	page_number: int | None
	chunk_index: int
	source_type: str
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorRetrievalResult:
	"""Evidence retrieved independently from the symptom and literature collections."""

	symptom_evidence: list[RetrievedEvidence]
	literature_evidence: list[RetrievedEvidence]

	@property
	def all_evidence(self) -> list[RetrievedEvidence]:
		"""Return all vector evidence in deterministic source-group order."""
		return [
			*self.symptom_evidence,
			*self.literature_evidence,
		]


class BiomedicalRetriever:
	"""Coordinate query embedding and independent vector retrieval."""

	def __init__(
		self,
		embedding_model: BiomedicalEmbeddingModel | None = None,
		symptom_store: QdrantVectorStore | None = None,
		literature_store: QdrantVectorStore | None = None,
		qdrant_client: QdrantClient | None = None,
	) -> None:
		self.embedding_model = (
			embedding_model
			if embedding_model is not None
			else BiomedicalEmbeddingModel()
		)

		client = (
			qdrant_client
			if qdrant_client is not None
			else QdrantClient(url="http://localhost:6333")
		)

		self.symptom_store = (
			symptom_store
			if symptom_store is not None
			else QdrantVectorStore(
				collection_name="symptom_chunks",
				vector_size=768,
				client=client,
			)
		)

		self.literature_store = (
			literature_store
			if literature_store is not None
			else QdrantVectorStore(
				collection_name="literature_chunks",
				vector_size=768,
				client=client,
			)
		)

	def retrieve(
		self,
		query: str,
		top_k: int = 5,
	) -> VectorRetrievalResult:
		"""
		Retrieve top-k evidence independently from both collections.

		The clinical query is embedded once and the same query vector is
		used for both symptom and literature retrieval.
		"""
		if not isinstance(query, str):
			raise ValueError("query must be a string")

		if not query.strip():
			raise ValueError("query cannot be empty or whitespace")

		if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
			raise ValueError("top_k must be a positive integer")

		# Embed the query exactly once.
		query_embedding = self.embedding_model.embed_text(query)

		# Independent retrieval from the symptom collection.
		symptom_results = self.symptom_store.search(
			query_embedding,
			top_k=top_k,
		)

		# Independent retrieval from the literature collection.
		literature_results = self.literature_store.search(
			query_embedding,
			top_k=top_k,
		)

		return VectorRetrievalResult(
			symptom_evidence=[
				self._to_evidence(result)
				for result in symptom_results
			],
			literature_evidence=[
				self._to_evidence(result)
				for result in literature_results
			],
		)

	@staticmethod
	def _to_evidence(result: Any) -> RetrievedEvidence:
		"""Convert a Qdrant search result into application-level evidence."""

		payload = result.payload or {}

		document_metadata = payload.get("document_metadata", {})
		chunk_metadata = payload.get("chunk_metadata", {})

		metadata = dict(payload.get("metadata", {}))
		metadata.update(document_metadata)
		metadata.update(chunk_metadata)

		# Literature payloads use source_file instead of filename.
		filename = payload.get("filename", "")
		if not filename:
			filename = payload.get("source_file", "")

		# Preserve literature-specific provenance for downstream reasoning.
		for key in (
			"pmid",
			"pmcid",
			"doi",
			"title",
			"journal",
			"publication_date",
			"publication_year",
			"section_title",
			"normalized_section_title",
			"section_path",
			"category",
			"source",
			"literature_chunk_index",
			"embedding_window_index",
			"embedding_window_count",
		):
			if key in payload:
				metadata[key] = payload[key]

		return RetrievedEvidence(
			text=payload.get("text", ""),
			score=result.score,
			filename=filename,
			page_number=payload.get("page_number"),
			chunk_index=payload.get(
				"chunk_index",
				payload.get("literature_chunk_index", 0),
			),
			source_type=payload.get("source_type", "unknown"),
			metadata=metadata,
		)