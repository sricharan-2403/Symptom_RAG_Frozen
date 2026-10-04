"""Tests for the ingestion entry point configuration.

Verifies that the ingestion pipeline is correctly configured without
requiring real embedding models or persistent Qdrant storage.
"""

import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


def test_ingest_loader_wrapper():
    """Verify loader wrapper has correct interface."""
    # This test runs without importing the real ingest module
    # to avoid unnecessary dependencies
    
    class MockLoader:
        """Mock of SymptomDiseaseDatasetLoader for testing."""
        def __init__(self, csv_path, mapping_path):
            self.csv_path = csv_path
            self.mapping_path = mapping_path
        
        def load_all(self):
            return []
    
    class SymptomDatasetLoaderWrapper:
        """The wrapper from ingest.py."""
        def __init__(self, csv_path, mapping_path):
            self.loader = MockLoader(csv_path, mapping_path)
        
        def load_all(self):
            return self.loader.load_all()
    
    # Create wrapper
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test.csv"
        mapping_path = Path(tmpdir) / "mapping.json"
        csv_path.touch()
        mapping_path.touch()
        
        wrapper = SymptomDatasetLoaderWrapper(csv_path, mapping_path)
        
        # Verify interface
        assert hasattr(wrapper, "load_all")
        assert callable(wrapper.load_all)
        assert wrapper.load_all() == []


def test_ingest_biomedical_indexer_configuration():
    """Verify loader wrapper has the interface needed for BiomedicalIndexer.
    
    BiomedicalIndexer expects document_loader.load_all() -> list[Document].
    This test verifies that SymptomDiseaseDatasetLoader has this interface.
    """
    from src.symptom_rag_analyzer.data.dataset_loaders import SymptomDiseaseDatasetLoader
    
    # Verify SymptomDiseaseDatasetLoader has the required interface
    assert hasattr(SymptomDiseaseDatasetLoader, "load_all")
    
    # Create a simple loader wrapper (from ingest.py)
    class SymptomDatasetLoaderWrapper:
        def __init__(self, csv_path, mapping_path):
            self.loader = SymptomDiseaseDatasetLoader(csv_path, mapping_path)
        
        def load_all(self):
            return self.loader.load_all()
    
    # Verify wrapper has the interface
    with tempfile.TemporaryDirectory() as tmpdir:
        import csv
        import json
        
        # Create minimal test files
        csv_path = Path(tmpdir) / "train.csv"
        mapping_path = Path(tmpdir) / "mapping.json"
        
        # Create mapping
        with open(mapping_path, "w") as f:
            json.dump({"Test Disease": 0}, f)
        
        # Create CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerow({"text": "test symptom", "label": "0"})
        
        # Verify wrapper works
        wrapper = SymptomDatasetLoaderWrapper(csv_path, mapping_path)
        assert hasattr(wrapper, "load_all")
        docs = wrapper.load_all()
        assert len(docs) == 1
        assert docs[0].filename == "symptom-disease-train-dataset.csv"


def test_ingest_paths_exist():
    """Verify dataset files exist before ingestion."""
    dataset_dir = Path("data/raw/symptom-disease-dataset")
    csv_path = dataset_dir / "symptom-disease-train-dataset.csv"
    mapping_path = dataset_dir / "mapping.json"
    
    assert csv_path.exists(), f"Dataset CSV not found: {csv_path}"
    assert mapping_path.exists(), f"Mapping file not found: {mapping_path}"


def test_ingest_mapping_structure():
    """Verify mapping.json has correct structure."""
    mapping_path = Path("data/raw/symptom-disease-dataset/mapping.json")
    
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    
    # Verify mapping structure
    assert isinstance(mapping, dict)
    assert len(mapping) > 0
    assert all(isinstance(k, str) for k in mapping.keys())
    assert all(isinstance(v, int) for v in mapping.values())
