"""Integration test for SymptomDiseaseDatasetLoader with BiomedicalIndexer pipeline.

Verifies that the dataset loader integrates correctly with the existing indexing
pipeline through dependency injection: loader → indexer → chunker → embeddings → vector_store
"""

import csv
import json
import tempfile
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import Mock

import numpy as np
import pytest

from symptom_rag_analyzer.data.chunks import Chunk
from symptom_rag_analyzer.data.dataset_loaders import SymptomDiseaseDatasetLoader
from symptom_rag_analyzer.data.chunker import TextChunker
from symptom_rag_analyzer.retrieval.indexer import BiomedicalIndexer


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def dataset_mapping(temp_dir):
    """Create a valid mapping.json file for dataset loader."""
    mapping = {
        "Heart Attack": 447,
        "Type 2 Diabetes": 1024,
        "Asthma": 76,
        "Depression": 278,
        "Pneumonia": 800,
    }
    mapping_file = temp_dir / "mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f)
    return mapping_file


@pytest.fixture
def dataset_csv(temp_dir):
    """Create a training CSV with both unique and duplicate records."""
    csv_file = temp_dir / "train.csv"
    rows = [
        # Unique records
        {"text": "chest pain, shortness of breath, diaphoresis", "label": "447"},
        {"text": "polyuria, polydipsia, fatigue, weight loss", "label": "1024"},
        {"text": "wheezing, breathing difficulties, cough", "label": "76"},
        # Exact duplicate (should be deduplicated)
        {"text": "chest pain, shortness of breath, diaphoresis", "label": "447"},
        # More unique records
        {"text": "persistent sadness, loss of interest", "label": "278"},
        {"text": "fever, productive cough, chest pain", "label": "800"},
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_file


class FakeEmbeddingModel:
    """Lightweight fake embedding model for testing.
    
    Returns deterministic random vectors based on text hash.
    """

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.embed_batch_call_count = 0
        self.embed_batch_calls = []

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate fake embeddings from text.
        
        Returns deterministic vectors based on text content.
        """
        self.embed_batch_call_count += 1
        self.embed_batch_calls.append(texts)

        embeddings = []
        for text in texts:
            # Create deterministic but different vector per text
            seed = hash(text) % (2**31)
            rng = np.random.RandomState(seed)
            embedding = rng.randn(self.embedding_dim).astype(np.float32)
            # Normalize to unit length
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)


class FakeVectorStore:
    """Lightweight fake vector store for testing.
    
    Tracks inserted chunks and embeddings without actual storage.
    """

    def __init__(self, collection_name: str = "test_collection", vector_size: int = 768):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.inserted_chunks = []
        self.inserted_embeddings = []
        self.insert_call_count = 0

    def insert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Track inserted chunks and embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        self.insert_call_count += 1
        self.inserted_chunks.extend(chunks)
        self.inserted_embeddings.extend(embeddings)


class TestSymptomDiseaseDatasetIndexerIntegration:
    """Integration tests for dataset loader with indexer pipeline."""

    def test_dataset_loader_produces_documents(self, dataset_csv, dataset_mapping):
        """Verify dataset loader produces Document objects."""
        loader = SymptomDiseaseDatasetLoader(dataset_csv, dataset_mapping)
        documents = loader.load_all()

        # Should have 5 unique documents (6 rows - 1 duplicate)
        assert len(documents) == 5

        # All should have required Document fields
        for doc in documents:
            assert doc.filename == "symptom-disease-train-dataset.csv"
            assert doc.page_number is None
            assert doc.text.startswith("Disease: ")
            assert "Symptoms: " in doc.text
            assert doc.source_type == "symptom-disease-dataset"
            assert doc.metadata["dataset_split"] == "train"

    def test_duplicate_records_removed(self, dataset_csv, dataset_mapping):
        """Verify exact duplicate (text, label) pairs are deduplicated."""
        loader = SymptomDiseaseDatasetLoader(dataset_csv, dataset_mapping)
        documents = loader.load_all()

        # CSV has 6 rows, but row 4 is exact duplicate of row 1
        # Expected: 5 unique documents
        assert len(documents) == 5

        # Verify the duplicate is indeed removed by checking symptom texts
        symptom_texts = [doc.metadata["raw_symptom_text"] for doc in documents]
        # "chest pain, shortness of breath, diaphoresis" should appear once
        assert symptom_texts.count("chest pain, shortness of breath, diaphoresis") == 1

    def test_indexer_accepts_dataset_loader_via_dependency_injection(
        self, dataset_csv, dataset_mapping
    ):
        """Verify BiomedicalIndexer accepts dataset loader through DI."""
        loader = SymptomDiseaseDatasetLoader(dataset_csv, dataset_mapping)
        fake_embed = FakeEmbeddingModel()
        fake_store = FakeVectorStore()
        chunker = TextChunker(chunk_size=100, overlap=10)

        # Inject all dependencies
        indexer = BiomedicalIndexer(
            document_loader=loader,
            chunker=chunker,
            embedding_model=fake_embed,
            vector_store=fake_store,
            collection_name="test_collection",
        )

        # Verify dependencies were accepted
        assert indexer.document_loader is loader
        assert indexer.chunker is chunker
        assert indexer.embedding_model is fake_embed
        assert indexer.vector_store is fake_store

    def test_documents_passed_to_chunker(self, dataset_csv, dataset_mapping):
        """Verify documents are passed from loader to chunker."""
        loader = SymptomDiseaseDatasetLoader(dataset_csv, dataset_mapping)
        fake_embed = FakeEmbeddingModel()
        fake_store = FakeVectorStore()
        chunker = TextChunker(chunk_size=100, overlap=10)

        indexer = BiomedicalIndexer(
            document_loader=loader,
            chunker=chunker,
            embedding_model=fake_embed,
            vector_store=fake_store,
        )

        result = indexer.index()

        # Verify documents were chunked
        assert result.document_count == 5  # 5 unique documents

    def test_chunk_texts_passed_to_embed_batch(self, dataset_csv, dataset_mapping):
        """Verify chunk texts are passed to embedding model in order."""
        loader = SymptomDiseaseDatasetLoader(dataset_csv, dataset_mapping)
        fake_embed = FakeEmbeddingModel()
        fake_store = FakeVectorStore()
        chunker = TextChunker(chunk_size=150, overlap=10)

        indexer = BiomedicalIndexer(
            document_loader=loader,
            chunker=chunker,
            embedding_model=fake_embed,
            vector_store=fake_store,
        )

        result = indexer.index()

        # Verify embed_batch was called
        assert fake_embed.embed_batch_call_count == 1

        # Verify all chunk texts were passed to embed_batch
        embedded_texts = fake_embed.embed_batch_calls[0]
        assert len(embedded_texts) == result.chunk_count

    def test_chunks_and_embeddings_remain_aligned(self, dataset_csv, dataset_mapping):
        """Verify chunks and embeddings stay aligned through pipeline."""
        loader = SymptomDiseaseDatasetLoader(dataset_csv, dataset_mapping)
        fake_embed = FakeEmbeddingModel()
        fake_store = FakeVectorStore()
        chunker = TextChunker(chunk_size=150, overlap=10)

        indexer = BiomedicalIndexer(
            document_loader=loader,
            chunker=chunker,
            embedding_model=fake_embed,
            vector_store=fake_store,
        )

        result = indexer.index()

        # Verify alignment: number of inserted chunks == number of inserted embeddings
        assert len(fake_store.inserted_chunks) == len(fake_store.inserted_embeddings)
        assert len(fake_store.inserted_chunks) == result.chunk_count

    def test_vector_store_receives_chunks_and_embeddings(self, dataset_csv, dataset_mapping):
        """Verify vector store insert() receives matching chunks and embeddings."""
        loader = SymptomDiseaseDatasetLoader(dataset_csv, dataset_mapping)
        fake_embed = FakeEmbeddingModel()
        fake_store = FakeVectorStore()
        chunker = TextChunker(chunk_size=200, overlap=10)

        indexer = BiomedicalIndexer(
            document_loader=loader,
            chunker=chunker,
            embedding_model=fake_embed,
            vector_store=fake_store,
        )

        result = indexer.index()

        # Verify insert was called exactly once
        assert fake_store.insert_call_count == 1

        # Verify data was inserted
        assert len(fake_store.inserted_chunks) > 0
        assert len(fake_store.inserted_embeddings) > 0

        # Verify alignment
        assert len(fake_store.inserted_chunks) == len(fake_store.inserted_embeddings)

    def test_indexing_result_document_count_correct(self, dataset_csv, dataset_mapping):
        """Verify IndexingResult.document_count is correct."""
        loader = SymptomDiseaseDatasetLoader(dataset_csv, dataset_mapping)
        fake_embed = FakeEmbeddingModel()
        fake_store = FakeVectorStore()
        chunker = TextChunker(chunk_size=150, overlap=10)

        indexer = BiomedicalIndexer(
            document_loader=loader,
            chunker=chunker,
            embedding_model=fake_embed,
            vector_store=fake_store,
        )

        result = indexer.index()

        # Should have 5 unique documents (6 rows - 1 duplicate)
        assert result.document_count == 5

    def test_indexing_result_chunk_count_correct(self, dataset_csv, dataset_mapping):
        """Verify IndexingResult.chunk_count is correct."""
        loader = SymptomDiseaseDatasetLoader(dataset_csv, dataset_mapping)
        fake_embed = FakeEmbeddingModel()
        fake_store = FakeVectorStore()
        chunker = TextChunker(chunk_size=150, overlap=10)

        indexer = BiomedicalIndexer(
            document_loader=loader,
            chunker=chunker,
            embedding_model=fake_embed,
            vector_store=fake_store,
        )

        result = indexer.index()

        # Should have multiple chunks from the 5 documents
        assert result.chunk_count > 0
        assert result.chunk_count >= result.document_count

    def test_indexing_result_embedding_count_correct(self, dataset_csv, dataset_mapping):
        """Verify IndexingResult.embedding_count is correct."""
        loader = SymptomDiseaseDatasetLoader(dataset_csv, dataset_mapping)
        fake_embed = FakeEmbeddingModel()
        fake_store = FakeVectorStore()
        chunker = TextChunker(chunk_size=150, overlap=10)

        indexer = BiomedicalIndexer(
            document_loader=loader,
            chunker=chunker,
            embedding_model=fake_embed,
            vector_store=fake_store,
        )

        result = indexer.index()

        # Number of embeddings should equal number of chunks
        assert result.embedding_count == result.chunk_count

    def test_collection_name_passed_through(self, dataset_csv, dataset_mapping):
        """Verify collection_name is correctly passed through to IndexingResult."""
        loader = SymptomDiseaseDatasetLoader(dataset_csv, dataset_mapping)
        fake_embed = FakeEmbeddingModel()
        fake_store = FakeVectorStore(collection_name="custom_collection")
        chunker = TextChunker(chunk_size=150, overlap=10)

        indexer = BiomedicalIndexer(
            document_loader=loader,
            chunker=chunker,
            embedding_model=fake_embed,
            vector_store=fake_store,
            collection_name="custom_collection",
        )

        result = indexer.index()

        assert result.collection_name == "custom_collection"

    def test_full_pipeline_integration_end_to_end(self, dataset_csv, dataset_mapping):
        """Complete end-to-end integration test of the entire pipeline."""
        # Setup
        loader = SymptomDiseaseDatasetLoader(dataset_csv, dataset_mapping)
        fake_embed = FakeEmbeddingModel(embedding_dim=768)
        fake_store = FakeVectorStore(collection_name="symptom_chunks", vector_size=768)
        chunker = TextChunker(chunk_size=200, overlap=20)

        # Create indexer with all DI components
        indexer = BiomedicalIndexer(
            document_loader=loader,
            chunker=chunker,
            embedding_model=fake_embed,
            vector_store=fake_store,
            collection_name="symptom_chunks",
            vector_size=768,
        )

        # Execute pipeline
        result = indexer.index()

        # Verify all stages completed
        assert result.document_count == 5  # Deduped from 6 rows
        assert result.chunk_count > 0
        assert result.embedding_count == result.chunk_count
        assert result.collection_name == "symptom_chunks"

        # Verify vector store received data
        assert len(fake_store.inserted_chunks) == result.chunk_count
        assert len(fake_store.inserted_embeddings) == result.chunk_count

        # Verify embeddings were generated
        assert fake_embed.embed_batch_call_count == 1
        assert len(fake_embed.embed_batch_calls[0]) == result.chunk_count

        # Verify chunks reference original documents
        for chunk in fake_store.inserted_chunks:
            assert chunk.document is not None
            assert chunk.document.source_type == "symptom-disease-dataset"
            assert chunk.document.metadata["dataset_split"] == "train"
