import json

from symptom_rag_analyzer.orchestrator import ClinicalTextClarifierOutput, SymptomRAGResult
from symptom_rag_analyzer.output_adapter import SymptomRAGOutputAdapter
from symptom_rag_analyzer.reasoning.answerer import DiagnosticCandidate, RAGAnswer
from symptom_rag_analyzer.retrieval.search import RetrievedEvidence


def test_symptom_rag_output_adapter_converts_internal_result():
	clinical_context = ClinicalTextClarifierOutput(
		clinical_text="Patient presents with fever and cough.",
		diseases=["pneumonia"],
		symptoms=["fever", "cough"],
		medications=["antibiotic"],
		tests=["chest x-ray"],
	)
	answer = RAGAnswer(
		diagnostic_candidates=[
			DiagnosticCandidate("pneumonia", 1, "Supported by evidence.", [1, 2])
		],
		limitations="Preliminary reasoning only.",
	)
	evidence = [
		RetrievedEvidence("Pneumonia causes fever.", 0.82, "guide.pdf", 42, 0, "PDF"),
		RetrievedEvidence("Cough can occur.", 0.71, "notes.txt", None, 1, "TXT"),
	]
	result = SymptomRAGResult(answer=answer, evidence=evidence)

	output = SymptomRAGOutputAdapter().adapt(clinical_context, result)

	assert output["agent"] == "symptom_rag"
	assert output["status"] == "success"
	assert output["query_context"] == {
		"diseases": ["pneumonia"],
		"symptoms": ["fever", "cough"],
		"medications": ["antibiotic"],
		"tests": ["chest x-ray"],
		"procedures": [],
	}
	assert output["diagnostic_candidates"] == [
		{
			"condition": "pneumonia",
			"rank": 1,
			"justification": "Supported by evidence.",
			"supporting_evidence": [1, 2],
		}
	]
	assert output["evidence"] == [
		{
			"content": "Pneumonia causes fever.",
			"source": "guide.pdf",
			"source_type": "PDF",
			"location": "Page 42",
			"relevance_score": 0.82,
		},
		{
			"content": "Cough can occur.",
			"source": "notes.txt",
			"source_type": "TXT",
			"location": "",
			"relevance_score": 0.71,
		},
	]
	assert output["metadata"] == {"evidence_count": 2}
	assert output["limitations"] == "Preliminary reasoning only."
	assert output["error"] is None
	json.dumps(output)