"""Application-level orchestration for biomedical document indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from symptom_rag_analyzer.data.chunker import TextChunker
from symptom_rag_analyzer.data.loaders import DocumentLoader
from symptom_rag_analyzer.embeddings.model import BiomedicalEmbeddingModel
from symptom_rag_analyzer.vector_store.qdrant_store import QdrantVectorStore


@dataclass
class IndexingResult:
	"""Summary of documents, chunks, embeddings, and the target collection."""

	document_count: int
	chunk_count: int
	embedding_count: int
	collection_name: str


class BiomedicalIndexer:
	"""Coordinate PDF loading, chunking, embedding, and Qdrant insertion."""

	def __init__(
		self,
		folder_path: str | Path | None = None,
		document_loader: DocumentLoader | None = None,
		chunker: TextChunker | None = None,
		embedding_model: BiomedicalEmbeddingModel | None = None,
		vector_store: QdrantVectorStore | None = None,
		chunk_size: int = 1000,
		overlap: int = 100,
		collection_name: str = "symptom_chunks",
		vector_size: int = 768,
		qdrant_path: str | Path | None = None,
	) -> None:
		if document_loader is None and folder_path is None:
			raise ValueError("folder_path is required when document_loader is not provided")

		self.document_loader = (
			document_loader if document_loader is not None else DocumentLoader(folder_path)
		)
		self.chunker = (
			chunker if chunker is not None else TextChunker(chunk_size, overlap)
		)
		self.embedding_model = (
			embedding_model
		if embedding_model is not None
		else BiomedicalEmbeddingModel()
		)
		self.vector_store = (
			vector_store
			if vector_store is not None
			else QdrantVectorStore(
				collection_name=collection_name,
				vector_size=vector_size,
				path=qdrant_path,
			)
		)

	def index(self) -> IndexingResult:
		"""Load, chunk, embed, and insert the configured PDF corpus."""
		documents = self.document_loader.load_all()
		chunks = self.chunker.chunk_documents(documents)

		if not chunks:
			return IndexingResult(
				document_count=len(documents),
				chunk_count=0,
				embedding_count=0,
				collection_name=self.vector_store.collection_name,
			)

		embeddings = self.embedding_model.embed_batch([chunk.text for chunk in chunks])
		if len(embeddings) != len(chunks):
			raise ValueError("number of embeddings must equal number of chunks")

		self.vector_store.insert(chunks, embeddings)
		return IndexingResult(
			document_count=len(documents),
			chunk_count=len(chunks),
			embedding_count=len(embeddings),
			collection_name=self.vector_store.collection_name,
		)
