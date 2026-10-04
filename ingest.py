#!/usr/bin/env python
"""Ingest the real Symptom-Disease Dataset into persistent Qdrant."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src/ to Python path so symptom_rag_analyzer can be imported
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from symptom_rag_analyzer.data.dataset_loaders import (
    SymptomDiseaseDatasetLoader,
)
from symptom_rag_analyzer.retrieval.indexer import BiomedicalIndexer


def ingest_symptom_dataset() -> None:
    """Load, embed, and index the real biomedical training dataset."""

    dataset_dir = Path("data/raw/symptom-disease-dataset")
    csv_path = dataset_dir / "symptom-disease-train-dataset.csv"
    mapping_path = dataset_dir / "mapping.json"

    qdrant_path = Path("data/qdrant")
    collection_name = "symptom_chunks"

    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    if not mapping_path.is_file():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    print(f"Dataset source: {dataset_dir.resolve()}")
    print(f"Qdrant storage: {qdrant_path.resolve()}")
    print(f"Collection name: {collection_name}")
    print("\nIngesting dataset...")

    loader = SymptomDiseaseDatasetLoader(
        training_csv_path=csv_path,
        mapping_json_path=mapping_path,
    )

    indexer = BiomedicalIndexer(
        document_loader=loader,
        collection_name=collection_name,
        qdrant_path=qdrant_path,
    )

    result = indexer.index()

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Documents:  {result.document_count:,}")
    print(f"Chunks:     {result.chunk_count:,}")
    print(f"Embeddings: {result.embedding_count:,}")
    print(f"Collection: {result.collection_name}")
    print("=" * 60)


if __name__ == "__main__":
    ingest_symptom_dataset()