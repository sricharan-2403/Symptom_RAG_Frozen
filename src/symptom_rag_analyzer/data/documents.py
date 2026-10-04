"""Core document model for extracted biomedical content.

This module defines the reusable data structure used throughout the RAG pipeline.
It intentionally contains only the data model and no document loading or
processing logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """Represents a single extracted document or page from a source file.

    The object is intentionally lightweight and reusable across ingestion,
    retrieval, and reasoning stages. It stores the source file identity,
    the page number when applicable, the extracted text, the source format,
    and a dictionary for future extensibility.
    """

    filename: str
    """Name of the source file (for example, a PDF or text document)."""

    page_number: int | None = None
    """Page number within the source document, or None for non-paged content."""

    text: str = ""
    """The extracted textual content for this document or page."""

    source_type: str = "unknown"
    """Type of source file, such as PDF, DOCX, TXT, or HTML."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Dictionary for additional source-specific or future metadata.

    This field is intentionally flexible so the model can evolve without
    changing the dataclass definition.
    """
