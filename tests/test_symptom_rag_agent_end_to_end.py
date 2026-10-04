import json

from qdrant_client import QdrantClient

from symptom_rag_analyzer.data.chunks import Chunk
from symptom_rag_analyzer.data.documents import Document
from symptom_rag_analyzer.embeddings.model import BiomedicalEmbeddingModel
from symptom_rag_analyzer.orchestrator import (
	ClinicalTextClarifierOutput,
	SymptomRAGOrchestrator,
	SymptomRAGResult,
)
from symptom_rag_analyzer.output_adapter import SymptomRAGOutputAdapter
from symptom_rag_analyzer.reasoning.answerer import BiomedicalAnswerer, RAGAnswer
from symptom_rag_analyzer.reasoning.context import EvidenceContextBuilder
from symptom_rag_analyzer.retrieval.search import BiomedicalRetriever
from symptom_rag_analyzer.vector_store.qdrant_store import QdrantVectorStore


def test_complete_symptom_rag_agent_end_to_end():
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
		collection_name="symptom_rag_agent_end_to_end_test",
		vector_size=768,
		client=QdrantClient(":memory:"),
	)
	vector_store.insert([chunk], [chunk_embedding])

	retriever = BiomedicalRetriever(
		embedding_model=embedding_model,
		vector_store=vector_store,
	)
	orchestrator = SymptomRAGOrchestrator(
		retriever=retriever,
		context_builder=EvidenceContextBuilder(),
		answerer=BiomedicalAnswerer(),
	)

	result = orchestrator.analyze(clinical_context)

	assert isinstance(result, SymptomRAGResult)
	assert isinstance(result.answer, RAGAnswer)
	assert result.evidence

	output = SymptomRAGOutputAdapter().adapt(clinical_context, result)

	assert isinstance(output, dict)
	assert output["agent"] == "symptom_rag"
	assert output["status"] == "success"
	assert output["query_context"]["diseases"] == ["pneumonia"]
	assert output["query_context"]["symptoms"] == [
		"fever",
		"cough",
		"difficulty breathing",
	]
	assert isinstance(output["diagnostic_candidates"], list)
	assert isinstance(output["evidence"], list)
	assert output["evidence"]
	assert output["metadata"]["evidence_count"] == len(output["evidence"])
	assert output["error"] is None
	assert isinstance(output["limitations"], str)

	json.dumps(output)
	print(json.dumps(output, indent=2))
