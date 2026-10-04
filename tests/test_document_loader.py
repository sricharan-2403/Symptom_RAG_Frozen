from pathlib import Path

import fitz
import pytest

from symptom_rag_analyzer.data.documents import Document
from symptom_rag_analyzer.data.loaders import DocumentLoader


def _create_test_pdf(path: Path, text: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_load_all_returns_one_document_per_pdf_page(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _create_test_pdf(pdf_path, "Clinical note for symptom analysis.")

    loader = DocumentLoader(str(tmp_path))
    documents = loader.load_all()

    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].filename == "sample.pdf"
    assert documents[0].page_number == 1
    assert documents[0].source_type == "PDF"
    assert "Clinical note" in documents[0].text


def test_load_single_pdf_raises_for_unreadable_pdf(tmp_path):
    bad_pdf = tmp_path / "broken.pdf"
    bad_pdf.write_bytes(b"not a valid pdf file")

    loader = DocumentLoader(str(tmp_path))

    with pytest.raises(ValueError, match="Could not open PDF"):
        loader._load_single_pdf(bad_pdf)
