"""Core chunk model for biomedical retrieval content.

This module defines the reusable data structure used throughout the RAG
pipeline. It intentionally contains only the data model and no chunking,
embedding, or vector-database logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .documents import Document


@dataclass
class Chunk:
    """Represents a single text chunk derived from a source document.

    Each chunk maintains a reference to the original document object so the
    chunk can reuse source metadata without duplicating document identity data.
    """

    document: Document
    """Original document instance from which this chunk was created."""

    chunk_index: int
    """Position of this chunk within the containing document or section."""

    text: str
    """Text content for this chunk."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional chunk-specific metadata for retrieval and downstream tasks.

    This field is intentionally flexible so metadata can evolve without
    changing the dataclass definition.
    """
