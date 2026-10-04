from qdrant_client import QdrantClient

from symptom_rag_analyzer.data.chunks import Chunk
from symptom_rag_analyzer.data.documents import Document
from symptom_rag_analyzer.embeddings.model import BiomedicalEmbeddingModel
from symptom_rag_analyzer.vector_store.qdrant_store import QdrantVectorStore


def test_end_to_end_rag_embedding_and_vector_search():
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
	assert chunk_embedding.shape == (768,)

	client = QdrantClient(":memory:")
	vector_store = QdrantVectorStore(
		collection_name="rag_integration_test",
		vector_size=768,
		client=client,
	)
	vector_store.insert([chunk], [chunk_embedding])

	query = "What condition can cause fever and cough?"
	query_embedding = embedding_model.embed_text(query)
	results = vector_store.search(query_embedding, top_k=1)

	assert results
	payload = results[0].payload
	assert payload["text"] == chunk.text
	assert payload["filename"] == document.filename
	assert payload["page_number"] == document.page_number
	assert payload["chunk_index"] == chunk.chunk_index

	print(f"Similarity score: {results[0].score}")
	print(f"Retrieved evidence: {payload['text']}")
