from numbers import Real

import pytest
from qdrant_client import QdrantClient

from symptom_rag_analyzer.data.chunks import Chunk
from symptom_rag_analyzer.data.documents import Document
from symptom_rag_analyzer.embeddings.model import BiomedicalEmbeddingModel
from symptom_rag_analyzer.retrieval.search import BiomedicalRetriever, RetrievedEvidence
from symptom_rag_analyzer.vector_store.qdrant_store import QdrantVectorStore


def test_retriever_end_to_end():
	document = Document(
		filename="pneumonia_case.txt",
		page_number=1,
		text="Pneumonia can cause fever and cough, often with difficulty breathing.",
		source_type="TXT",
	)
	chunk = Chunk(
		document=document,
		chunk_index=0,
		text=document.text,
	)

	embedding_model = BiomedicalEmbeddingModel()
	chunk_embedding = embedding_model.embed_text(chunk.text)

	vector_store = QdrantVectorStore(
		collection_name="retriever_integration_test",
		vector_size=768,
		client=QdrantClient(":memory:"),
	)
	vector_store.insert([chunk], [chunk_embedding])

	retriever = BiomedicalRetriever(
		embedding_model=embedding_model,
		vector_store=vector_store,
	)
	results = retriever.retrieve(
		"What condition can cause fever and cough?",
		top_k=1,
	)

	assert len(results) == 1
	evidence = results[0]
	assert isinstance(evidence, RetrievedEvidence)
	assert evidence.text == document.text
	assert evidence.filename == document.filename
	assert evidence.page_number == document.page_number
	assert evidence.chunk_index == chunk.chunk_index
	assert evidence.source_type == document.source_type
	assert isinstance(evidence.score, Real)

	print(f"Similarity score: {evidence.score}")
	print(f"Retrieved evidence: {evidence.text}")

	with pytest.raises(ValueError):
		retriever.retrieve("   ")
