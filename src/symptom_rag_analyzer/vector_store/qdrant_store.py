"""Qdrant-backed storage and similarity search for embedded chunks."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from symptom_rag_analyzer.data.chunks import Chunk


class QdrantVectorStore:
	"""Store and search chunk embeddings in a local Qdrant collection."""

	def __init__(
		self,
		collection_name: str = "symptom_chunks",
		vector_size: int = 768,
		path: str | Path | None = None,
		client: QdrantClient | None = None,
	) -> None:
		if vector_size <= 0:
			raise ValueError("vector_size must be positive")

		self.collection_name = collection_name
		self.vector_size = vector_size
		self.client = client or (
			QdrantClient(path=str(path)) if path is not None else QdrantClient(":memory:")
		)
		self._ensure_collection()

	def _ensure_collection(self) -> None:
		if not self.client.collection_exists(self.collection_name):
			self.client.create_collection(
				collection_name=self.collection_name,
				vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
			)

	def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
		"""Insert or replace embeddings and their source chunk payloads."""
		if len(chunks) != len(embeddings):
			raise ValueError("chunks and embeddings must have the same length")

		points = []
		for chunk, embedding in zip(chunks, embeddings):
			vector = list(embedding)
			if len(vector) != self.vector_size:
				raise ValueError(
					f"embedding must have dimension {self.vector_size}, got {len(vector)}"
				)
			points.append(
				PointStruct(
					id=self._point_id(chunk),
					vector=vector,
					payload=self._chunk_payload(chunk),
				)
			)

		if points:
			self.client.upsert(collection_name=self.collection_name, points=points)

	def insert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
		"""Compatibility alias for :meth:`upsert`."""
		self.upsert(chunks, embeddings)

	def upsert_vectors(
		self,
		point_ids: Sequence[str],
		vectors: Sequence[Sequence[float]],
		payloads: Sequence[Mapping[str, Any]],
	) -> int:
		"""Upsert vectors with caller-provided deterministic IDs and payloads."""
		if len(point_ids) != len(vectors) or len(point_ids) != len(payloads):
			raise ValueError("point_ids, vectors, and payloads must have the same length")
		if len(set(point_ids)) != len(point_ids):
			raise ValueError("point_ids must be unique within an upsert batch")

		points = []
		for point_id, vector, payload in zip(point_ids, vectors, payloads):
			values = [float(value) for value in vector]
			if len(values) != self.vector_size:
				raise ValueError(
					f"embedding must have dimension {self.vector_size}, got {len(values)}"
				)
			points.append(
				PointStruct(id=point_id, vector=values, payload=dict(payload))
			)

		if points:
			self.client.upsert(collection_name=self.collection_name, points=points)
		return len(points)

	def retrieve_by_ids(
		self,
		point_ids: Sequence[str],
		*,
		with_vectors: bool = True,
		with_payload: bool = True,
	) -> list[Any]:
		"""Read points by ID from this store's configured collection."""
		if not point_ids:
			return []
		return self.client.retrieve(
			collection_name=self.collection_name,
			ids=list(point_ids),
			with_vectors=with_vectors,
			with_payload=with_payload,
		)

	def search(self, query_vector: Sequence[float], top_k: int = 5) -> list[Any]:
		"""Return the top-k cosine-similar Qdrant search results."""
		query = list(query_vector)
		if len(query) != self.vector_size:
			raise ValueError(
				f"query_vector must have dimension {self.vector_size}, got {len(query)}"
			)
		if top_k <= 0:
			raise ValueError("top_k must be positive")

		response = self.client.query_points(
			collection_name=self.collection_name,
			query=query,
			limit=top_k,
		)
		return list(response.points)

	@staticmethod
	def _point_id(chunk: Chunk) -> str:
		document_metadata = chunk.document.metadata

		if chunk.document.source_type == "symptom-disease-dataset":
			identity = {
				"source_type": chunk.document.source_type,
				"disease_id": document_metadata.get("disease_id"),
				"raw_symptom_text": document_metadata.get("raw_symptom_text"),
				"chunk_index": chunk.chunk_index,
			}
		else:
			identity = {
				"filename": chunk.document.filename,
				"page_number": chunk.document.page_number,
				"source_type": chunk.document.source_type,
				"chunk_index": chunk.chunk_index,
			}

		return str(
			uuid.uuid5(
				uuid.NAMESPACE_URL,
				json.dumps(identity, sort_keys=True, ensure_ascii=False),
			)
		)

	@staticmethod
	def _chunk_payload(chunk: Chunk) -> dict[str, Any]:
		return {
			"text": chunk.text,
			"filename": chunk.document.filename,
			"page_number": chunk.document.page_number,
			"chunk_index": chunk.chunk_index,
			"source_type": chunk.document.source_type,
			"document_metadata": dict(chunk.document.metadata),
			"chunk_metadata": dict(chunk.metadata),
		}
