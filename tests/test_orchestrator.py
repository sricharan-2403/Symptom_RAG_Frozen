from qdrant_client import QdrantClient

from symptom_rag_analyzer.data.chunks import Chunk
from symptom_rag_analyzer.data.documents import Document
from symptom_rag_analyzer.embeddings.model import BiomedicalEmbeddingModel
from symptom_rag_analyzer.orchestrator import (
	ClinicalTextClarifierOutput,
	SymptomRAGResult,
	SymptomRAGOrchestrator,
)
from symptom_rag_analyzer.reasoning.answerer import BiomedicalAnswerer, RAGAnswer
from symptom_rag_analyzer.reasoning.context import EvidenceContextBuilder
from symptom_rag_analyzer.retrieval.search import BiomedicalRetriever
from symptom_rag_analyzer.vector_store.qdrant_store import QdrantVectorStore


def test_symptom_rag_orchestrator_end_to_end():
	clinical_context = ClinicalTextClarifierOutput(
		clinical_text="Patient presents with fever, cough, and difficulty breathing.",
		diseases=["pneumonia"],
		symptoms=["fever", "cough", "difficulty breathing"],
		medications=[],
		tests=[],
	)

	document = Document(
		filename="pneumonia_guidelines.pdf",
		page_number=42,
		text=(
			"Pneumonia commonly causes fever and cough. "
			"Difficulty breathing may occur in patients with pneumonia."
		),
		source_type="PDF",
	)
	chunk = Chunk(document=document, chunk_index=0, text=document.text)

	embedding_model = BiomedicalEmbeddingModel()
	chunk_embedding = embedding_model.embed_text(chunk.text)

	vector_store = QdrantVectorStore(
		collection_name="orchestrator_integration_test",
		vector_size=768,
		client=QdrantClient(":memory:"),
	)
	vector_store.insert([chunk], [chunk_embedding])

	retriever = BiomedicalRetriever(
		embedding_model=embedding_model,
		vector_store=vector_store,
	)
	context_builder = EvidenceContextBuilder()
	answerer = BiomedicalAnswerer()
	orchestrator = SymptomRAGOrchestrator(
		retriever=retriever,
		context_builder=context_builder,
		answerer=answerer,
	)

	query = orchestrator._build_retrieval_query(clinical_context)
	assert "Diseases: pneumonia" in query
	assert "Symptoms: fever, cough, difficulty breathing" in query
	assert "medications" not in query.lower()
	assert "tests" not in query.lower()

	result = orchestrator.analyze(clinical_context)

	assert isinstance(result, SymptomRAGResult)
	assert isinstance(result.answer, RAGAnswer)
	assert isinstance(result.answer.diagnostic_candidates, list)
	assert isinstance(result.answer.limitations, str)
	print(f"Returned RAGAnswer:\n{result!r}")
