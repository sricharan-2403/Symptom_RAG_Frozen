"""Loaders for biomedical knowledge base datasets.

This module converts structured dataset files (CSV + mapping) into
Document objects for ingestion into the RAG pipeline.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from symptom_rag_analyzer.data.documents import Document


class SymptomDiseaseDatasetLoader:
    """Load Hugging Face Symptom-Disease Dataset training CSV into Documents.

    Converts each unique training record (text, label pair) into a Document,
    resolving disease labels through a mapping file and deduplicating exact
    duplicate records.

    The loader reads only the training split and preserves data provenance
    in metadata for traceability.
    """

    def __init__(self, training_csv_path: str | Path, mapping_json_path: str | Path):
        """Initialize the loader with dataset file paths.

        Args:
            training_csv_path: Path to symptom-disease-train-dataset.csv
            mapping_json_path: Path to mapping.json (disease name → ID mapping)

        Raises:
            FileNotFoundError: If either file does not exist.
            ValueError: If files are not accessible or paths are invalid.
        """
        self.training_csv_path = Path(training_csv_path)
        self.mapping_json_path = Path(mapping_json_path)

        # Validate paths exist and are files
        if not self.training_csv_path.exists():
            raise FileNotFoundError(
                f"Training CSV file does not exist: {self.training_csv_path}"
            )
        if not self.training_csv_path.is_file():
            raise ValueError(
                f"Training CSV path is not a file: {self.training_csv_path}"
            )

        if not self.mapping_json_path.exists():
            raise FileNotFoundError(
                f"Mapping JSON file does not exist: {self.mapping_json_path}"
            )
        if not self.mapping_json_path.is_file():
            raise ValueError(
                f"Mapping JSON path is not a file: {self.mapping_json_path}"
            )

        # Load and validate mapping
        self._disease_id_to_name = self._load_mapping()

    def _load_mapping(self) -> dict[int, str]:
        """Load disease mapping from mapping.json.

        The mapping.json has structure:
            {
              "Disease Name (string)": disease_id (int),
              ...
            }

        This method inverts it to:
            {
              disease_id (int): "Disease Name (string)",
              ...
            }

        Returns:
            Dictionary mapping disease ID (int) to disease name (str).

        Raises:
            ValueError: If mapping.json cannot be parsed or is invalid.
        """
        try:
            with open(self.mapping_json_path, "r", encoding="utf-8") as f:
                mapping_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in mapping file: {self.mapping_json_path}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Could not read mapping file: {self.mapping_json_path}"
            ) from e

        # Invert: disease_name -> id becomes id -> disease_name
        inverted = {}
        for disease_name, disease_id in mapping_dict.items():
            if not isinstance(disease_id, int):
                raise ValueError(
                    f"Invalid disease ID (expected int, got {type(disease_id).__name__}): "
                    f"'{disease_name}' -> {disease_id}"
                )
            inverted[disease_id] = disease_name

        return inverted

    def load_all(self) -> list[Document]:
        """Load training dataset and convert to Documents.

        Reads the training CSV, validates rows, deduplicates exact (text, label)
        pairs, resolves disease labels via mapping, and creates Document objects.

        Returns:
            List of Document objects in deterministic order. Each Document
            represents one unique training record.

        Raises:
            ValueError: If CSV is malformed, missing required columns, or contains
                       invalid disease labels.
        """
        seen_records: set[tuple[str, int]] = set()
        documents: list[Document] = []

        try:
            with open(self.training_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                # Validate required columns exist
                if reader.fieldnames is None or "text" not in reader.fieldnames:
                    raise ValueError("CSV missing required 'text' column")
                if "label" not in reader.fieldnames:
                    raise ValueError("CSV missing required 'label' column")

                for row_num, row in enumerate(reader, start=2):  # start=2 (after header)
                    # Extract and clean fields
                    symptom_text = row.get("text", "").strip()
                    label_str = row.get("label", "").strip()

                    # Skip empty symptom text
                    if not symptom_text:
                        continue

                    # Validate and parse disease label
                    try:
                        disease_id = int(label_str)
                    except ValueError:
                        raise ValueError(
                            f"Row {row_num}: Invalid disease label (not an integer): '{label_str}'"
                        )

                    # Resolve disease name from ID
                    if disease_id not in self._disease_id_to_name:
                        raise ValueError(
                            f"Row {row_num}: Disease ID {disease_id} not found in mapping"
                        )
                    disease_name = self._disease_id_to_name[disease_id]

                    # Deduplicate exact (text, label) pairs
                    record_key = (symptom_text, disease_id)
                    if record_key in seen_records:
                        continue
                    seen_records.add(record_key)

                    # Create Document
                    document = Document(
                        filename="symptom-disease-train-dataset.csv",
                        page_number=None,
                        text=f"Disease: {disease_name}\nSymptoms: {symptom_text}",
                        source_type="symptom-disease-dataset",
                        metadata={
                            "disease_id": disease_id,
                            "disease_name": disease_name,
                            "raw_symptom_text": symptom_text,
                            "data_source": "Hugging Face: dux-tecblic/symptom-disease-dataset",
                            "dataset_split": "train",
                        },
                    )
                    documents.append(document)

        except ValueError:
            # Re-raise ValueError (validation errors)
            raise
        except Exception as e:
            raise ValueError(
                f"Could not read training CSV: {self.training_csv_path}"
            ) from e

        return documents
