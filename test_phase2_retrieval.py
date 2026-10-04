from symptom_rag_analyzer.retrieval.search import BiomedicalRetriever


def main() -> None:
	query = "fever, cough, shortness of breath"

	retriever = BiomedicalRetriever()

	result = retriever.retrieve(query, top_k=5)

	print("\n===== PHASE 2 RETRIEVAL TEST =====")

	print(f"\nQuery: {query}")

	print("\n--- SYMPTOM EVIDENCE ---")
	print(f"Count: {len(result.symptom_evidence)}")

	for index, evidence in enumerate(result.symptom_evidence, start=1):
		print(f"\n[{index}] Score: {evidence.score:.4f}")
		print(f"Source type: {evidence.source_type}")
		print(f"Chunk: {evidence.chunk_index}")
		print(f"Text: {evidence.text[:300]}")

	print("\n--- LITERATURE EVIDENCE ---")
	print(f"Count: {len(result.literature_evidence)}")

	for index, evidence in enumerate(result.literature_evidence, start=1):
		print(f"\n[{index}] Score: {evidence.score:.4f}")
		print(f"Source type: {evidence.source_type}")
		print(f"Chunk: {evidence.chunk_index}")
		print(f"Text: {evidence.text[:300]}")

	print("\n--- SUMMARY ---")
	print(f"Symptom results: {len(result.symptom_evidence)}")
	print(f"Literature results: {len(result.literature_evidence)}")
	print(f"Total vector evidence: {len(result.all_evidence)}")

	assert len(result.symptom_evidence) == 5
	assert len(result.literature_evidence) == 5
	assert len(result.all_evidence) == 10

	assert all(
		item.source_type == "symptom-disease-dataset"
		for item in result.symptom_evidence
	)

	assert all(
		item.source_type == "pubmed-pmc-literature"
		for item in result.literature_evidence
	)

	print("\nPASS: Phase 2 independent retrieval is working.")


if __name__ == "__main__":
	main()