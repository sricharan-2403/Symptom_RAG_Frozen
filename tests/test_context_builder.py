from symptom_rag_analyzer.reasoning.context import EvidenceContextBuilder
from symptom_rag_analyzer.retrieval.search import RetrievedEvidence


def test_build_context_formats_evidence_deterministically():
	evidence = [
		RetrievedEvidence(
			text="Pneumonia may cause fever and cough.",
			score=0.91,
			filename="pneumonia.pdf",
			page_number=3,
			chunk_index=0,
			source_type="PDF",
		),
		RetrievedEvidence(
			text="Shortness of breath can accompany pneumonia.",
			score=0.84,
			filename="clinical_notes.txt",
			page_number=7,
			chunk_index=2,
			source_type="TXT",
		),
	]
	builder = EvidenceContextBuilder()

	context = builder.build_context(evidence)
	expected = """[Evidence 1]
Source: pneumonia.pdf
Source type: PDF
Page: 3
Similarity: 0.91
Text: Pneumonia may cause fever and cough.

[Evidence 2]
Source: clinical_notes.txt
Source type: TXT
Page: 7
Similarity: 0.84
Text: Shortness of breath can accompany pneumonia."""

	assert context == expected
	assert context.index("[Evidence 1]") < context.index("[Evidence 2]")
	assert builder.build_context(evidence) == context
	assert builder.build_context([]) == ""
