"""Loaders for source documents used by the biomedical RAG pipeline.

This module intentionally contains only ingestion logic for source files.
It reads PDFs and converts each page into a reusable Document object.
"""

from __future__ import annotations

from pathlib import Path

import fitz

from symptom_rag_analyzer.data.documents import Document


class DocumentLoader:
    """Load all PDF files from a folder into Document objects.

    Each page of each PDF becomes a separate Document instance. The loader
    does not perform cleaning, chunking, embeddings, or retrieval.
    """

    def __init__(self, folder_path: str | Path):
        """Initialize the loader with a folder containing PDF files.

        Args:
            folder_path: Path to a directory containing PDF source files.
        """
        self.folder_path = Path(folder_path)

    def load_all(self) -> list[Document]:
        """Load every PDF in the configured folder.

        Returns:
            A list of Document objects, one per page across all PDFs.

        Raises:
            FileNotFoundError: If the configured folder does not exist.
            ValueError: If a PDF cannot be opened.
        """
        if not self.folder_path.exists():
            raise FileNotFoundError(f"PDF folder does not exist: {self.folder_path}")

        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"Expected a directory for PDF files: {self.folder_path}")

        documents: list[Document] = []
        for pdf_path in sorted(self.folder_path.glob("*.pdf")):
            documents.extend(self._load_single_pdf(pdf_path))

        return documents

    def _load_single_pdf(self, pdf_path: str | Path) -> list[Document]:
        """Read one PDF file and convert each page into a Document.

        Args:
            pdf_path: Path to a single PDF file.

        Returns:
            A list of Document objects, one per page.

        Raises:
            ValueError: If the PDF cannot be opened or read successfully.
        """
        pdf_file = Path(pdf_path)

        try:
            doc = fitz.open(str(pdf_file))
        except Exception as exc:  # pragma: no cover - thin wrapper around fitz exceptions
            raise ValueError(f"Could not open PDF: {pdf_file}") from exc

        try:
            pages: list[Document] = []
            for page_number in range(len(doc)):
                page = doc.load_page(page_number)
                text = page.get_text("text")

                pages.append(
                    Document(
                        filename=pdf_file.name,
                        page_number=page_number + 1,
                        text=text,
                        source_type="PDF",
                        metadata={"file_path": str(pdf_file)},
                    )
                )
            return pages
        except Exception as exc:
            raise ValueError(f"Could not read PDF content from: {pdf_file}") from exc
        finally:
            doc.close()
