from qdrant_client import QdrantClient

from symptom_rag_analyzer.embeddings.model import BiomedicalEmbeddingModel
from symptom_rag_analyzer.vector_store.qdrant_store import QdrantVectorStore


QUERY = "fever, cough, shortness of breath"

print("=" * 70)
print("DOCKER QDRANT RETRIEVAL SMOKE TEST")
print("=" * 70)

# Connect to Docker Qdrant
client = QdrantClient(url="http://localhost:6333")

# Load the same biomedical embedding model used during ingestion
embedding_model = BiomedicalEmbeddingModel()

# Embed the clinical query
query_vector = embedding_model.embed_text(QUERY)

print(f"\nQuery: {QUERY}")
print(f"Query vector dimension: {len(query_vector)}")


for collection_name in ["symptom_chunks", "literature_chunks"]:

    print("\n" + "-" * 70)
    print(f"COLLECTION: {collection_name}")
    print("-" * 70)

    store = QdrantVectorStore(
        collection_name=collection_name,
        vector_size=768,
        client=client,
    )

    results = store.search(query_vector, top_k=5)

    print(f"Retrieved results: {len(results)}")

    for i, result in enumerate(results, start=1):
        payload = result.payload or {}

        print(f"\n[{i}] Score: {result.score:.4f}")
        print(f"    Point ID: {result.id}")
        print(f"    Source type: {payload.get('source_type')}")
        print(f"    Filename: {payload.get('filename')}")
        print(f"    Page: {payload.get('page_number')}")
        print(f"    Chunk: {payload.get('chunk_index')}")

        document_metadata = payload.get("document_metadata", {})

        if collection_name == "symptom_chunks":
            print(f"    Disease ID: {document_metadata.get('disease_id')}")
            print(f"    Disease: {document_metadata.get('disease_name')}")
            print(
                f"    Raw symptoms: "
                f"{document_metadata.get('raw_symptom_text')}"
            )

        text = payload.get("text", "")
        print(f"    Text: {text[:500]}")


print("\n" + "=" * 70)
print("SMOKE TEST COMPLETE")
print("=" * 70)