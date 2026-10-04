"""Tests for dataset loaders.

Uses temporary CSV and mapping.json fixtures created in-memory.
Does not download real datasets or use embedding models.
"""

import csv
import json
import tempfile
from pathlib import Path

import pytest

from symptom_rag_analyzer.data.dataset_loaders import SymptomDiseaseDatasetLoader
from symptom_rag_analyzer.data.documents import Document


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def valid_mapping(temp_dir):
    """Create a valid mapping.json file in temp directory.

    Structure:
        {
          "Disease Name (str)": disease_id (int),
          ...
        }
    """
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
def valid_csv(temp_dir):
    """Create a valid training CSV file in temp directory.

    Columns: text, label
    """
    csv_file = temp_dir / "train.csv"
    rows = [
        {"text": "chest pain, shortness of breath, diaphoresis", "label": "447"},
        {"text": "polyuria, polydipsia, fatigue", "label": "1024"},
        {"text": "wheezing, breathing problems", "label": "76"},
        {"text": "sadness, loss of interest, sleep disturbance", "label": "278"},
        {"text": "cough, fever, sputum", "label": "800"},
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_file


@pytest.fixture
def csv_with_duplicates(temp_dir):
    """Create a CSV with exact duplicate (text, label) pairs."""
    csv_file = temp_dir / "train_dups.csv"
    rows = [
        {"text": "chest pain, shortness of breath, diaphoresis", "label": "447"},
        {"text": "polyuria, polydipsia, fatigue", "label": "1024"},
        # Exact duplicate of first row
        {"text": "chest pain, shortness of breath, diaphoresis", "label": "447"},
        {"text": "wheezing, breathing problems", "label": "76"},
        # Exact duplicate of second row
        {"text": "polyuria, polydipsia, fatigue", "label": "1024"},
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_file


@pytest.fixture
def csv_with_empty_rows(temp_dir):
    """Create a CSV with empty/whitespace symptom text."""
    csv_file = temp_dir / "train_empty.csv"
    rows = [
        {"text": "chest pain, shortness of breath", "label": "447"},
        {"text": "", "label": "1024"},  # Empty
        {"text": "   ", "label": "76"},  # Whitespace only
        {"text": "wheezing, breathing problems", "label": "278"},
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_file


@pytest.fixture
def csv_with_invalid_labels(temp_dir):
    """Create a CSV with invalid disease labels."""
    csv_file = temp_dir / "train_invalid.csv"
    rows = [
        {"text": "chest pain, shortness of breath", "label": "447"},
        {"text": "not a number", "label": "invalid_id"},  # Non-numeric
        {"text": "wheezing, breathing problems", "label": "9999"},  # ID not in mapping
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_file


@pytest.fixture
def csv_missing_columns(temp_dir):
    """Create a CSV missing required 'text' or 'label' columns."""
    csv_file = temp_dir / "train_bad_cols.csv"
    rows = [
        {"symptom": "chest pain", "disease_id": "447"},
        {"symptom": "fever", "disease_id": "800"},
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["symptom", "disease_id"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_file


@pytest.fixture
def invalid_mapping(temp_dir):
    """Create an invalid mapping.json (contains non-integer values)."""
    mapping_file = temp_dir / "bad_mapping.json"
    mapping = {
        "Heart Attack": "not_an_int",  # Invalid
        "Diabetes": 1024,
    }
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f)
    return mapping_file


class TestSymptomDiseaseDatasetLoader:
    """Test suite for SymptomDiseaseDatasetLoader."""

    def test_constructor_validates_csv_exists(self, temp_dir, valid_mapping):
        """Constructor should raise FileNotFoundError if CSV does not exist."""
        nonexistent_csv = temp_dir / "nonexistent.csv"
        with pytest.raises(FileNotFoundError, match="Training CSV file does not exist"):
            SymptomDiseaseDatasetLoader(nonexistent_csv, valid_mapping)

    def test_constructor_validates_mapping_exists(self, temp_dir, valid_csv):
        """Constructor should raise FileNotFoundError if mapping does not exist."""
        nonexistent_mapping = temp_dir / "nonexistent.json"
        with pytest.raises(FileNotFoundError, match="Mapping JSON file does not exist"):
            SymptomDiseaseDatasetLoader(valid_csv, nonexistent_mapping)

    def test_constructor_validates_csv_is_file(self, temp_dir, valid_mapping):
        """Constructor should raise ValueError if CSV path is a directory."""
        csv_dir = temp_dir / "csv_dir"
        csv_dir.mkdir()
        with pytest.raises(ValueError, match="Training CSV path is not a file"):
            SymptomDiseaseDatasetLoader(csv_dir, valid_mapping)

    def test_constructor_validates_mapping_is_file(self, temp_dir, valid_csv):
        """Constructor should raise ValueError if mapping path is a directory."""
        mapping_dir = temp_dir / "mapping_dir"
        mapping_dir.mkdir()
        with pytest.raises(ValueError, match="Mapping JSON path is not a file"):
            SymptomDiseaseDatasetLoader(valid_csv, mapping_dir)

    def test_load_all_basic_valid_data(self, valid_csv, valid_mapping):
        """load_all() should successfully load valid CSV and create Documents."""
        loader = SymptomDiseaseDatasetLoader(valid_csv, valid_mapping)
        documents = loader.load_all()

        # Should have 5 documents (5 valid rows)
        assert len(documents) == 5

        # All should be Document instances
        assert all(isinstance(d, Document) for d in documents)

        # Check first document
        doc = documents[0]
        assert doc.filename == "symptom-disease-train-dataset.csv"
        assert doc.page_number is None
        assert "Heart Attack" in doc.text
        assert "chest pain, shortness of breath, diaphoresis" in doc.text
        assert doc.source_type == "symptom-disease-dataset"

    def test_document_text_format(self, valid_csv, valid_mapping):
        """Document.text should have format: 'Disease: <name>\\nSymptoms: <text>'."""
        loader = SymptomDiseaseDatasetLoader(valid_csv, valid_mapping)
        documents = loader.load_all()

        doc = documents[0]
        expected_format = "Disease: Heart Attack\nSymptoms: chest pain, shortness of breath, diaphoresis"
        assert doc.text == expected_format

    def test_document_source_type(self, valid_csv, valid_mapping):
        """Document.source_type should be 'symptom-disease-dataset'."""
        loader = SymptomDiseaseDatasetLoader(valid_csv, valid_mapping)
        documents = loader.load_all()

        assert all(d.source_type == "symptom-disease-dataset" for d in documents)

    def test_document_metadata_present(self, valid_csv, valid_mapping):
        """Document metadata should contain all required fields."""
        loader = SymptomDiseaseDatasetLoader(valid_csv, valid_mapping)
        documents = loader.load_all()

        doc = documents[0]
        metadata = doc.metadata

        assert metadata["disease_id"] == 447
        assert metadata["disease_name"] == "Heart Attack"
        assert metadata["raw_symptom_text"] == "chest pain, shortness of breath, diaphoresis"
        assert metadata["data_source"] == "Hugging Face: dux-tecblic/symptom-disease-dataset"
        assert metadata["dataset_split"] == "train"

    def test_disease_label_resolution(self, valid_csv, valid_mapping):
        """Disease labels should be correctly resolved to disease names."""
        loader = SymptomDiseaseDatasetLoader(valid_csv, valid_mapping)
        documents = loader.load_all()

        # Verify each document has correct disease name
        disease_mapping = {
            0: "Heart Attack",
            1: "Type 2 Diabetes",
            2: "Asthma",
            3: "Depression",
            4: "Pneumonia",
        }

        for i, doc in enumerate(documents):
            expected_disease = disease_mapping[i]
            assert doc.metadata["disease_name"] == expected_disease

    def test_deduplicate_exact_records(self, csv_with_duplicates, valid_mapping):
        """Exact duplicate (text, label) pairs should be deduplicated."""
        loader = SymptomDiseaseDatasetLoader(csv_with_duplicates, valid_mapping)
        documents = loader.load_all()

        # CSV has 5 rows, but 2 duplicates (rows 1 and 3 are same, rows 2 and 5 are same)
        # So should have 3 unique documents
        assert len(documents) == 3

    def test_deterministic_ordering(self, valid_csv, valid_mapping):
        """Document order should be deterministic (first occurrence of each unique record)."""
        loader = SymptomDiseaseDatasetLoader(valid_csv, valid_mapping)
        documents1 = loader.load_all()

        # Load again
        loader2 = SymptomDiseaseDatasetLoader(valid_csv, valid_mapping)
        documents2 = loader2.load_all()

        # Should have identical order
        assert len(documents1) == len(documents2)
        for d1, d2 in zip(documents1, documents2):
            assert d1.text == d2.text
            assert d1.metadata == d2.metadata

    def test_ignore_empty_rows(self, csv_with_empty_rows, valid_mapping):
        """Rows with empty/whitespace symptom text should be ignored."""
        loader = SymptomDiseaseDatasetLoader(csv_with_empty_rows, valid_mapping)
        documents = loader.load_all()

        # CSV has 4 rows, but rows 2 and 3 are empty/whitespace
        # So should have 2 documents
        assert len(documents) == 2

        # All documents should have non-empty text
        assert all(d.text.strip() for d in documents)

    def test_strip_whitespace_from_fields(self, temp_dir, valid_mapping):
        """Whitespace should be stripped from text and label fields."""
        csv_file = temp_dir / "train_spaces.csv"
        rows = [
            {"text": "  chest pain, shortness of breath  ", "label": "  447  "},
            {"text": "\tfever\t", "label": "\t800\t"},
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerows(rows)

        loader = SymptomDiseaseDatasetLoader(csv_file, valid_mapping)
        documents = loader.load_all()

        assert len(documents) == 2
        # Check that whitespace was stripped
        assert "  chest pain" not in documents[0].text
        assert "\tfever" not in documents[1].text

    def test_reject_non_numeric_label(self, csv_with_invalid_labels, valid_mapping):
        """Non-numeric disease labels should raise ValueError."""
        loader = SymptomDiseaseDatasetLoader(csv_with_invalid_labels, valid_mapping)
        with pytest.raises(ValueError, match="Invalid disease label.*not an integer"):
            loader.load_all()

    def test_reject_missing_disease_id(self, temp_dir, valid_mapping):
        """Disease IDs not in mapping should raise ValueError."""
        csv_file = temp_dir / "train_missing_id.csv"
        rows = [
            {"text": "chest pain, shortness of breath", "label": "447"},
            {"text": "wheezing, breathing problems", "label": "9999"},  # ID not in mapping
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerows(rows)
        
        loader = SymptomDiseaseDatasetLoader(csv_file, valid_mapping)
        with pytest.raises(ValueError, match="Disease ID.*not found in mapping"):
            loader.load_all()

    def test_reject_missing_required_columns(self, csv_missing_columns, valid_mapping):
        """CSV missing 'text' or 'label' columns should raise ValueError."""
        loader = SymptomDiseaseDatasetLoader(csv_missing_columns, valid_mapping)
        with pytest.raises(ValueError, match="CSV missing required"):
            loader.load_all()

    def test_reject_invalid_mapping_json(self, valid_csv, invalid_mapping):
        """Invalid mapping.json (non-integer values) should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid disease ID"):
            SymptomDiseaseDatasetLoader(valid_csv, invalid_mapping)

    def test_reject_malformed_json(self, temp_dir, valid_csv):
        """Malformed JSON should raise ValueError."""
        bad_json = temp_dir / "bad.json"
        with open(bad_json, "w") as f:
            f.write("{invalid json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            SymptomDiseaseDatasetLoader(valid_csv, bad_json)

    def test_preserve_original_label_in_metadata(self, valid_csv, valid_mapping):
        """Original disease label (ID) should be preserved in metadata."""
        loader = SymptomDiseaseDatasetLoader(valid_csv, valid_mapping)
        documents = loader.load_all()

        # Verify disease_id is in metadata for each document
        for doc in documents:
            assert "disease_id" in doc.metadata
            assert isinstance(doc.metadata["disease_id"], int)

    def test_preserve_raw_symptom_text(self, valid_csv, valid_mapping):
        """Original symptom text should be preserved in metadata."""
        loader = SymptomDiseaseDatasetLoader(valid_csv, valid_mapping)
        documents = loader.load_all()

        doc = documents[0]
        assert doc.metadata["raw_symptom_text"] == "chest pain, shortness of breath, diaphoresis"

    def test_no_text_modification(self, valid_csv, valid_mapping):
        """Symptom text should not be modified (no lowercasing, expanding, etc)."""
        csv_file = valid_csv.parent / "train_case.csv"
        rows = [
            {"text": "Chest Pain, FEVER, Short_of_Breath", "label": "447"},
            {"text": "COVID-19 symptoms", "label": "800"},
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerows(rows)

        loader = SymptomDiseaseDatasetLoader(csv_file, valid_mapping)
        documents = loader.load_all()

        # Check that original case and abbreviations are preserved
        assert "Chest Pain, FEVER" in documents[0].text
        assert "Short_of_Breath" in documents[0].text
        assert "COVID-19" in documents[1].text

    def test_dataset_split_in_metadata(self, valid_csv, valid_mapping):
        """Metadata should record dataset_split as 'train'."""
        loader = SymptomDiseaseDatasetLoader(valid_csv, valid_mapping)
        documents = loader.load_all()

        assert all(d.metadata["dataset_split"] == "train" for d in documents)

    def test_filename_constant_across_documents(self, valid_csv, valid_mapping):
        """All documents should have filename='symptom-disease-train-dataset.csv'."""
        loader = SymptomDiseaseDatasetLoader(valid_csv, valid_mapping)
        documents = loader.load_all()

        assert all(d.filename == "symptom-disease-train-dataset.csv" for d in documents)

    def test_page_number_is_none(self, valid_csv, valid_mapping):
        """Page number should be None (not applicable to CSV)."""
        loader = SymptomDiseaseDatasetLoader(valid_csv, valid_mapping)
        documents = loader.load_all()

        assert all(d.page_number is None for d in documents)

    def test_empty_csv(self, temp_dir, valid_mapping):
        """Empty CSV (header only) should return empty list of documents."""
        csv_file = temp_dir / "empty.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            # No data rows

        loader = SymptomDiseaseDatasetLoader(csv_file, valid_mapping)
        documents = loader.load_all()

        assert documents == []

    def test_all_empty_rows_csv(self, csv_with_empty_rows, valid_mapping):
        """CSV with all empty/whitespace rows should return empty list."""
        csv_file = csv_with_empty_rows.parent / "all_empty.csv"
        rows = [
            {"text": "", "label": "447"},
            {"text": "   ", "label": "800"},
            {"text": "\t\n", "label": "76"},
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerows(rows)

        loader = SymptomDiseaseDatasetLoader(csv_file, valid_mapping)
        documents = loader.load_all()

        assert documents == []

    def test_large_mapping_all_ids_sequential(self, temp_dir, valid_csv):
        """Loader should handle large mapping with all sequential IDs."""
        # Create mapping with IDs 0-1081
        large_mapping = {f"Disease {i}": i for i in range(1082)}
        mapping_file = temp_dir / "large_mapping.json"
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(large_mapping, f)

        # Create CSV that uses these IDs
        csv_file = temp_dir / "train_large.csv"
        rows = [
            {"text": "symptom 1", "label": "0"},
            {"text": "symptom 2", "label": "500"},
            {"text": "symptom 3", "label": "1081"},
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerows(rows)

        loader = SymptomDiseaseDatasetLoader(csv_file, mapping_file)
        documents = loader.load_all()

        assert len(documents) == 3
        assert documents[0].metadata["disease_name"] == "Disease 0"
        assert documents[1].metadata["disease_name"] == "Disease 500"
        assert documents[2].metadata["disease_name"] == "Disease 1081"

    def test_handles_special_characters_in_text(self, temp_dir, valid_mapping):
        """Loader should preserve special characters in symptom text."""
        csv_file = temp_dir / "train_special.csv"
        rows = [
            {"text": "pain & aching (bilateral)", "label": "447"},
            {"text": "fever > 101°F, chills", "label": "800"},
            {"text": "\"acute\" symptoms; worsening", "label": "76"},
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerows(rows)

        loader = SymptomDiseaseDatasetLoader(csv_file, valid_mapping)
        documents = loader.load_all()

        assert len(documents) == 3
        assert "pain & aching (bilateral)" in documents[0].text
        assert "fever > 101°F" in documents[1].text
        assert '"acute"' in documents[2].text
