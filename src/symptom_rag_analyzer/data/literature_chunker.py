import re
from typing import List

from .literature_chunks import LiteratureChunk
from .literature_models import LiteratureArticle, LiteratureSection


class LiteratureSectionChunker:
    """Split literature sections into paragraph-aware, section-isolated chunks."""

    def __init__(self, target_size: int = 1000, overlap: int = 150) -> None:
        if target_size <= 0:
            raise ValueError("target_size must be greater than 0")
        if overlap < 0 or overlap >= target_size:
            raise ValueError("overlap must satisfy 0 <= overlap < target_size")

        self.target_size = target_size
        self.overlap = overlap

    def chunk_article(self, article: LiteratureArticle) -> List[LiteratureChunk]:
        """Chunk abstract sections followed by body sections without mixing them."""
        chunks = []
        for section in article.abstract_sections + article.body_sections:
            if not section.text or not section.text.strip():
                continue
            section_texts = self._chunk_section_text(section.text)
            for chunk_index, text in enumerate(section_texts):
                chunks.append(self._make_chunk(article, section, text, chunk_index))
        return chunks

    def _chunk_section_text(self, text: str) -> List[str]:
        paragraphs = self._split_paragraphs(text)
        chunks = []
        paragraph_group = []
        group_length = 0

        for paragraph in paragraphs:
            if len(paragraph) > self.target_size:
                if paragraph_group:
                    chunks.append("\n\n".join(paragraph_group))
                    paragraph_group = []
                    group_length = 0
                chunks.extend(self._split_oversized_paragraph(paragraph))
                continue

            proposed_length = group_length + (2 if paragraph_group else 0) + len(paragraph)
            if paragraph_group and proposed_length > self.target_size:
                chunks.append("\n\n".join(paragraph_group))
                paragraph_group = [paragraph]
                group_length = len(paragraph)
            else:
                paragraph_group.append(paragraph)
                group_length = proposed_length

        if paragraph_group:
            chunks.append("\n\n".join(paragraph_group))
        return chunks

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
        return [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n+", normalized_newlines.strip())
            if paragraph.strip()
        ]

    def _split_oversized_paragraph(self, paragraph: str) -> List[str]:
        step = self.target_size - self.overlap
        chunks = []
        for start in range(0, len(paragraph), step):
            end = start + self.target_size
            chunks.append(paragraph[start:end])
            if end >= len(paragraph):
                break
        return chunks

    @staticmethod
    def _make_chunk(
        article: LiteratureArticle,
        section: LiteratureSection,
        text: str,
        chunk_index: int,
    ) -> LiteratureChunk:
        return LiteratureChunk(
            text=text,
            pmid=article.pmid,
            pmcid=article.pmcid,
            doi=article.doi,
            section_title=section.title,
            normalized_section_title=section.normalized_title,
            section_path=list(section.section_path),
            category=section.category,
            source=section.source,
            level=section.level,
            chunk_index=chunk_index,
        )
