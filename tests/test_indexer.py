from symptom_rag_analyzer.data.chunks import Chunk
from symptom_rag_analyzer.data.documents import Document
from symptom_rag_analyzer.retrieval.indexer import BiomedicalIndexer, IndexingResult


class FakeDocumentLoader:
	def __init__(self, documents):
		self.documents = documents
		self.load_calls = 0

	def load_all(self):
		self.load_calls += 1
		return self.documents


class FakeChunker:
	def __init__(self, chunks):
		self.chunks = chunks
		self.chunk_calls = []

	def chunk_documents(self, documents):
		self.chunk_calls.append(documents)
		return self.chunks


class FakeEmbeddingModel:
	def __init__(self):
		self.embed_batch_calls = []
		self.embed_text_calls = 0

	def embed_batch(self, texts):
		self.embed_batch_calls.append(texts)
		return [[index, index + 1] for index in range(len(texts))]


class FakeVectorStore:
	collection_name = "test_collection"

	def __init__(self):
		self.insert_calls = []

	def insert(self, chunks, embeddings):
		self.insert_calls.append((chunks, embeddings))


def test_indexer_loads_chunks_embeds_in_order_and_inserts():
	document = Document(
		filename="pneumonia.pdf",
		page_number=1,
		text="Pneumonia causes fever and cough.",
		source_type="PDF",
	)
	chunks = [
		Chunk(document=document, chunk_index=0, text="Pneumonia causes fever."),
		Chunk(document=document, chunk_index=1, text="Pneumonia causes cough."),
	]
	loader = FakeDocumentLoader([document])
	chunker = FakeChunker(chunks)
	embedding_model = FakeEmbeddingModel()
	vector_store = FakeVectorStore()

	result = BiomedicalIndexer(
		document_loader=loader,
		chunker=chunker,
		embedding_model=embedding_model,
		vector_store=vector_store,
	).index()

	assert loader.load_calls == 1
	assert chunker.chunk_calls == [[document]]
	assert embedding_model.embed_batch_calls == [
		["Pneumonia causes fever.", "Pneumonia causes cough."]
	]
	assert embedding_model.embed_text_calls == 0
	assert vector_store.insert_calls == [(chunks, [[0, 1], [1, 2]])]
	assert result == IndexingResult(1, 2, 2, "test_collection")


def test_indexer_handles_empty_documents_and_zero_chunks_without_embedding_or_insert():
	loader = FakeDocumentLoader([])
	chunker = FakeChunker([])
	embedding_model = FakeEmbeddingModel()
	vector_store = FakeVectorStore()

	result = BiomedicalIndexer(
		document_loader=loader,
		chunker=chunker,
		embedding_model=embedding_model,
		vector_store=vector_store,
	).index()

	assert result == IndexingResult(0, 0, 0, "test_collection")
	assert embedding_model.embed_batch_calls == []
	assert vector_store.insert_calls == []
