"""Simple text chunking utilities for biomedical documents.

This module intentionally contains only the character-based chunking logic used
for preparing document text into reusable Chunk objects. It does not perform
embedding generation, vector storage, or external model calls.
"""

from __future__ import annotations

from .chunks import Chunk
from .documents import Document


class TextChunker:
    """Split documents into smaller overlapping character-based text chunks.

    The implementation is intentionally simple and readable for the initial RAG
    pipeline. It uses a sliding window over the document text and preserves the
    original Document instance on every generated Chunk.
    """

    def __init__(self, chunk_size: int, overlap: int = 0) -> None:
        """Initialize the chunker with the target chunk size and overlap."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Split each document into overlapping chunks and return Chunk objects.

        Empty or whitespace-only document text is ignored. Chunk indices start at
        zero for each individual document.
        """
        chunks: list[Chunk] = []

        for document in documents:
            if document.text is None or not document.text.strip():
                continue

            chunk_index = 0
            start = 0

            while start < len(document.text):
                end = start + self.chunk_size
                chunk_text = document.text[start:end]

                chunks.append(
                    Chunk(
                        document=document,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        metadata={},
                    )
                )

                if end >= len(document.text):
                    break

                start += self.chunk_size - self.overlap
                chunk_index += 1

        return chunks
