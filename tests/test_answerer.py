import pytest

from symptom_rag_analyzer.reasoning.answerer import BiomedicalAnswerer, RAGAnswer


def test_biomedical_answerer_returns_structured_rag_answer():
	clinical_context = "Patient presents with fever, cough, and difficulty breathing."
	evidence_context = """[Evidence 1]
Source: pneumonia_guidelines.pdf
Source type: PDF
Page: 42
Similarity: 0.82
Text: Pneumonia commonly causes fever and cough.

[Evidence 2]
Source: pneumonia_guidelines.pdf
Source type: PDF
Page: 43
Similarity: 0.78
Text: Difficulty breathing may occur in patients with pneumonia."""

	result = BiomedicalAnswerer().answer(clinical_context, evidence_context)

	assert isinstance(result, RAGAnswer)
	assert isinstance(result.diagnostic_candidates, list)
	assert isinstance(result.limitations, str)

	print(f"Returned RAGAnswer:\n{result!r}")

	for candidate in result.diagnostic_candidates:
		assert isinstance(candidate.condition, str)
		assert candidate.condition.strip()
		assert isinstance(candidate.rank, int)
		assert not isinstance(candidate.rank, bool)
		assert isinstance(candidate.justification, str)
		assert candidate.justification.strip()
		assert isinstance(candidate.supporting_evidence, list)
		assert all(
			isinstance(evidence_number, int)
			and not isinstance(evidence_number, bool)
			for evidence_number in candidate.supporting_evidence
		)


def test_biomedical_answerer_rejects_empty_clinical_context():
	with pytest.raises(ValueError):
		BiomedicalAnswerer().answer("   ", "valid evidence")


def test_biomedical_answerer_rejects_invalid_clinical_context_type():
	with pytest.raises(TypeError):
		BiomedicalAnswerer().answer(None, "valid evidence")


def test_biomedical_answerer_rejects_invalid_evidence_context_type():
	with pytest.raises(TypeError):
		BiomedicalAnswerer().answer("valid clinical context", None)


def test_biomedical_answerer_allows_empty_evidence_context():
	result = BiomedicalAnswerer().answer("valid clinical context", "")

	assert isinstance(result, RAGAnswer)
