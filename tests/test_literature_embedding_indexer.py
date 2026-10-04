from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from qdrant_client import QdrantClient

from symptom_rag_analyzer.data.literature_chunks import LiteratureChunk
from symptom_rag_analyzer.data.literature_models import LiteratureArticle
from symptom_rag_analyzer.embeddings.literature_embedding_windows import (
    LiteratureEmbeddingWindow,
)
from symptom_rag_analyzer.ingestion.literature_embedding_indexer import (
    LITERATURE_COLLECTION,
    LiteratureEmbeddingIndexer,
)
from symptom_rag_analyzer.vector_store.qdrant_store import QdrantVectorStore


def make_article():
    return LiteratureArticle(
        pmid="12345678",
        pmcid="PMC1234567",
        doi="10.1234/example",
        title="Example article",
        journal="Example Journal",
        publication_date="2024-01-02",
        publication_year=2024,
        record_type="article",
        article_type="research-article",
        publication_types=["Journal Article"],
        authors=["A. Author"],
        mesh_terms=["Example Term"],
        source_file="PMC1234567.xml",
    )


def make_chunk(text="one two three", *, chunk_index=0, section_path=None):
    return LiteratureChunk(
        text=text,
        pmid="12345678",
        pmcid="PMC1234567",
        doi="10.1234/example",
        section_title="Results",
        normalized_section_title="results",
        section_path=section_path if section_path is not None else ["Results", "Findings"],
        category="results",
        source="body",
        chunk_index=chunk_index,
        metadata={"data_source": "unit-test"},
    )


def make_window(
    *,
    text="one two",
    chunk_index=0,
    window_index=0,
    window_count=1,
    section_path=None,
):
    return LiteratureEmbeddingWindow(
        text=text,
        pmid="12345678",
        pmcid="PMC1234567",
        doi="10.1234/example",
        section_title="Results",
        normalized_section_title="results",
        section_path=section_path if section_path is not None else ["Results", "Findings"],
        category="results",
        source="body",
        literature_chunk_index=chunk_index,
        embedding_window_index=window_index,
        embedding_window_count=window_count,
        metadata={"data_source": "unit-test"},
    )


class FakeEmbeddingModel:
    embedding_dim = 768

    def __init__(self):
        self.batches = []

    def embed_batch(self, texts):
        self.batches.append(list(texts))
        return np.ones((len(texts), self.embedding_dim), dtype=np.float32)


class FakeWindowSplitter:
    def split_chunk(self, chunk):
        if chunk.text == "first second":
            return [
                make_window(text="first", chunk_index=chunk.chunk_index, window_index=0, window_count=2),
                make_window(text="second", chunk_index=chunk.chunk_index, window_index=1, window_count=2),
            ]
        return [make_window(text=chunk.text, chunk_index=chunk.chunk_index)]


class FakeVectorStore:
    collection_name = LITERATURE_COLLECTION
    vector_size = 768

    def __init__(self):
        self.points = {}
        self.upsert_batches = []

    def upsert_vectors(self, point_ids, vectors, payloads):
        self.upsert_batches.append(list(point_ids))
        for point_id, vector, payload in zip(point_ids, vectors, payloads):
            self.points[point_id] = (np.asarray(vector).copy(), dict(payload))
        return len(point_ids)


def make_indexer(*, batch_size=32, store=None, model=None):
    return LiteratureEmbeddingIndexer(
        embedding_batch_size=batch_size,
        progress_every_batches=100,
        embedding_model=model or FakeEmbeddingModel(),
        window_splitter=FakeWindowSplitter(),
        vector_store=store or FakeVectorStore(),
    )


def test_deterministic_id_is_stable_for_same_window():
    window = make_window()

    assert LiteratureEmbeddingIndexer.deterministic_window_id(window) == (
        LiteratureEmbeddingIndexer.deterministic_window_id(window)
    )


def test_different_windows_and_sections_have_different_ids():
    window = make_window()

    assert LiteratureEmbeddingIndexer.deterministic_window_id(window) != (
        LiteratureEmbeddingIndexer.deterministic_window_id(
            replace(window, embedding_window_index=1)
        )
    )
    assert LiteratureEmbeddingIndexer.deterministic_window_id(window) != (
        LiteratureEmbeddingIndexer.deterministic_window_id(
            replace(window, section_path=["Discussion"])
        )
    )


def test_payload_preserves_article_and_window_provenance():
    article = make_article()
    window = make_window()

    payload = LiteratureEmbeddingIndexer.make_payload(article, window)

    assert payload["text"] == window.text
    assert payload["pmid"] == article.pmid
    assert payload["pmcid"] == article.pmcid
    assert payload["doi"] == article.doi
    assert payload["title"] == article.title
    assert payload["journal"] == article.journal
    assert payload["publication_date"] == article.publication_date
    assert payload["publication_year"] == article.publication_year
    assert payload["record_type"] == article.record_type
    assert payload["article_type"] == article.article_type
    assert payload["publication_types"] == article.publication_types
    assert payload["authors"] == article.authors
    assert payload["mesh_terms"] == article.mesh_terms
    assert payload["section_path"] == window.section_path
    assert payload["literature_chunk_index"] == window.literature_chunk_index
    assert payload["embedding_window_index"] == window.embedding_window_index
    assert payload["embedding_window_count"] == window.embedding_window_count
    assert payload["source_file"] == article.source_file
    assert payload["source_type"] == article.source_type
    assert payload["chunk_metadata"] == window.metadata
    assert payload["window_metadata"]["data_source"] == "unit-test"
    assert payload["window_metadata"]["embedding_window_index"] == 0


def test_batch_processing_and_count_tracking():
    model = FakeEmbeddingModel()
    store = FakeVectorStore()
    indexer = make_indexer(batch_size=2, model=model, store=store)
    article = make_article()
    records = [
        (article, make_chunk("first second", chunk_index=0)),
        (article, make_chunk("third", chunk_index=1)),
        (article, make_chunk("fourth", chunk_index=2)),
    ]

    result = indexer.index_chunk_records(records, articles_processed=1)

    assert [len(batch) for batch in model.batches] == [2, 2]
    assert result.articles_processed == 1
    assert result.chunks_processed == 3
    assert result.embedding_windows == 4
    assert result.embeddings_generated == 4
    assert result.vectors_upserted == 4
    assert result.failures == 0
    assert result.is_successful


def test_reingestion_upserts_existing_deterministic_ids():
    store = FakeVectorStore()
    indexer = make_indexer(store=store)
    article = make_article()
    records = [(article, make_chunk("same source text", chunk_index=4))]

    first = indexer.index_chunk_records(records, articles_processed=1)
    first_ids = set(store.points)
    second = indexer.index_chunk_records(records, articles_processed=1)

    assert first.is_successful and second.is_successful
    assert first_ids == set(store.points)
    assert len(store.points) == 1
    assert len(store.upsert_batches) == 2


def test_indexer_rejects_symptom_collection():
    store = FakeVectorStore()
    store.collection_name = "symptom_chunks"

    with pytest.raises(ValueError, match="cannot target symptom_chunks"):
        make_indexer(store=store)


def test_indexer_can_target_a_dedicated_benchmark_collection():
    store = FakeVectorStore()
    store.collection_name = "literature_chunks_benchmark"

    indexer = LiteratureEmbeddingIndexer(
        collection_name="literature_chunks_benchmark",
        embedding_model=FakeEmbeddingModel(),
        window_splitter=FakeWindowSplitter(),
        vector_store=store,
    )

    assert indexer.vector_store.collection_name == "literature_chunks_benchmark"


def test_qdrant_explicit_upsert_is_idempotent_and_collection_isolated():
    client = QdrantClient(":memory:")
    symptom_store = QdrantVectorStore(
        collection_name="symptom_chunks",
        vector_size=768,
        client=client,
    )
    literature_store = QdrantVectorStore(
        collection_name="literature_chunks",
        vector_size=768,
        client=client,
    )
    vector = np.ones(768, dtype=np.float32)
    payload = {"text": "literature", "pmid": "12345678"}

    stable_id = "123e4567-e89b-12d3-a456-426614174000"
    assert literature_store.upsert_vectors([stable_id], [vector], [payload]) == 1
    assert literature_store.upsert_vectors([stable_id], [vector], [payload]) == 1

    assert client.count("literature_chunks", exact=True).count == 1
    assert client.count("symptom_chunks", exact=True).count == 0
    stored = literature_store.retrieve_by_ids([stable_id])
    assert len(stored) == 1
    assert len(stored[0].vector) == 768
    assert stored[0].payload == payload
    client.close()


def test_qdrant_explicit_upsert_validates_vector_dimensions_and_ids():
    store = FakeVectorStore()

    with pytest.raises(ValueError, match="unique within an upsert batch"):
        QdrantVectorStore.upsert_vectors(
            store,
            ["same", "same"],
            [np.ones(768), np.ones(768)],
            [{}, {}],
        )
    with pytest.raises(ValueError, match="dimension 768"):
        QdrantVectorStore.upsert_vectors(
            store,
            ["id"],
            [np.ones(3)],
            [{}],
        )