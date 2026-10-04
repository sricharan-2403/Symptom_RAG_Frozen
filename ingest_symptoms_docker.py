import sys
from pathlib import Path

sys.path.insert(0, "src")

from qdrant_client import QdrantClient
from symptom_rag_analyzer.data.dataset_loaders import SymptomDiseaseDatasetLoader
from symptom_rag_analyzer.data.chunker import TextChunker
from symptom_rag_analyzer.embeddings.model import BiomedicalEmbeddingModel
from symptom_rag_analyzer.vector_store.qdrant_store import QdrantVectorStore


DATA = Path("data/raw/symptom-disease-dataset")
BATCH_SIZE = 100

loader = SymptomDiseaseDatasetLoader(
    DATA / "symptom-disease-train-dataset.csv",
    DATA / "mapping.json",
)

docs = loader.load_all()
chunks = TextChunker(1000, 100).chunk_documents(docs)

print(f"Documents: {len(docs)}")
print(f"Chunks: {len(chunks)}")

assert len(docs) == 1993
assert len(chunks) == 2690

model = BiomedicalEmbeddingModel()

client = QdrantClient(url="http://localhost:6333")

store = QdrantVectorStore(
    collection_name="symptom_chunks",
    vector_size=768,
    client=client,
)

total = 0

print("Starting batched ingestion...")

for start in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[start:start + BATCH_SIZE]

    embeddings = model.embed_batch(
        [chunk.text for chunk in batch]
    )

    assert len(embeddings) == len(batch)

    store.insert(batch, embeddings)

    total += len(batch)

    print(f"Uploaded {total}/{len(chunks)}")

info = client.get_collection("symptom_chunks")
literature = client.get_collection("literature_chunks")

print()
print("===== FINAL VALIDATION =====")
print("Documents:", len(docs))
print("Chunks:", len(chunks))
print("Embeddings uploaded:", total)
print("symptom_chunks:", info.points_count)
print("literature_chunks:", literature.points_count)

print(
    "PASS:",
    len(docs) == 1993
    and len(chunks) == 2690
    and total == 2690
    and info.points_count == 2690
    and literature.points_count == 201134
)
