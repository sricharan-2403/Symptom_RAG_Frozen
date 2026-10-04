from numbers import Real

import fitz
from qdrant_client import QdrantClient

from symptom_rag_analyzer.data.chunker import TextChunker
from symptom_rag_analyzer.data.loaders import DocumentLoader
from symptom_rag_analyzer.embeddings.model import BiomedicalEmbeddingModel
from symptom_rag_analyzer.retrieval.indexer import BiomedicalIndexer
from symptom_rag_analyzer.retrieval.search import BiomedicalRetriever
from symptom_rag_analyzer.vector_store.qdrant_store import QdrantVectorStore


def test_real_pdf_ingestion_indexes_and_retrieves_biomedical_text(tmp_path):
	pdf_directory = tmp_path / "biomedical"
	pdf_directory.mkdir()
	pdf_path = pdf_directory / "pneumonia_fixture.pdf"
	expected_text = (
		"Pneumonia commonly causes fever, cough, and difficulty breathing."
	)

	pdf = fitz.open()
	page = pdf.new_page()
	page.insert_text((72, 72), expected_text)
	pdf.save(pdf_path)
	pdf.close()

	embedding_model = BiomedicalEmbeddingModel()
	vector_store = QdrantVectorStore(
		collection_name="real_pdf_ingestion_test",
		vector_size=768,
		client=QdrantClient(":memory:"),
	)
	indexer = BiomedicalIndexer(
		document_loader=DocumentLoader(pdf_directory),
		chunker=TextChunker(chunk_size=100, overlap=10),
		embedding_model=embedding_model,
		vector_store=vector_store,
	)

	result = indexer.index()

	assert result.document_count == 1
	assert result.chunk_count >= 1
	assert result.embedding_count == result.chunk_count
	assert result.collection_name == "real_pdf_ingestion_test"

	retriever = BiomedicalRetriever(
		embedding_model=embedding_model,
		vector_store=vector_store,
	)
	retrieved_evidence = retriever.retrieve(
		"pneumonia fever cough difficulty breathing",
		top_k=1,
	)

	assert retrieved_evidence
	evidence = retrieved_evidence[0]
	assert evidence.filename == "pneumonia_fixture.pdf"
	assert evidence.source_type == "PDF"
	assert evidence.page_number == 1
	assert expected_text in evidence.text
	assert isinstance(evidence.score, Real)
