"""Create tokenizer-bounded model-input windows from literature chunks."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from symptom_rag_analyzer.data.literature_chunks import LiteratureChunk
from symptom_rag_analyzer.embeddings.model import BiomedicalEmbeddingModel


@dataclass
class LiteratureEmbeddingWindow:
    """One model-input text window with provenance to its source chunk."""

    text: str
    pmid: str
    pmcid: Optional[str]
    doi: Optional[str]
    section_title: str
    normalized_section_title: str
    section_path: List[str]
    category: str
    source: str
    literature_chunk_index: int
    embedding_window_index: int
    embedding_window_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class LiteratureEmbeddingWindowSplitter:
    """Split LiteratureChunks on tokenizer boundaries under the model limit."""

    def __init__(self, embedding_model: BiomedicalEmbeddingModel) -> None:
        self.tokenizer = embedding_model.model.tokenizer
        self.max_tokens = embedding_model.model.max_seq_length
        self.special_token_count = self.tokenizer.num_special_tokens_to_add(pair=False)
        self.content_token_capacity = self.max_tokens - self.special_token_count
        if self.content_token_capacity <= 0:
            raise ValueError(
                "model max_seq_length must exceed the tokenizer's special-token count"
            )

    def split_chunk(self, chunk: LiteratureChunk) -> List[LiteratureEmbeddingWindow]:
        """Return contiguous text windows without losing or repeating tokens."""
        if not isinstance(chunk.text, str):
            raise ValueError("LiteratureChunk.text must be a string")
        if not chunk.text.strip():
            raise ValueError("LiteratureChunk.text cannot be empty or whitespace")

        original_encoding = self.tokenizer(
            chunk.text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )
        token_ids = original_encoding["input_ids"]
        offsets = original_encoding["offset_mapping"]
        word_ids = original_encoding.word_ids()
        if not token_ids:
            raise ValueError("LiteratureChunk.text produced no tokenizer tokens")
        if (
            len(token_ids) != len(offsets)
            or len(token_ids) != len(word_ids)
            or any(start >= end for start, end in offsets)
            or any(word_id is None for word_id in word_ids)
        ):
            raise ValueError(
                "Tokenizer must provide valid offsets and word IDs for every content token"
            )

        window_texts = []
        token_start = 0
        char_start = 0
        while token_start < len(token_ids):
            token_end = min(
                token_start + self.content_token_capacity,
                len(token_ids),
            )
            while (
                token_end > token_start
                and token_end < len(token_ids)
                and word_ids[token_end - 1] == word_ids[token_end]
            ):
                token_end -= 1
            if token_end == token_start:
                raise ValueError(
                    "A single tokenizer word exceeds the model's content-token capacity"
                )

            char_end = (
                len(chunk.text)
                if token_end == len(token_ids)
                else offsets[token_end][0]
            )
            window_text = chunk.text[char_start:char_end]
            window_encoding = self.tokenizer(
                window_text,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                verbose=False,
            )
            if window_encoding["input_ids"] != token_ids[token_start:token_end]:
                raise ValueError(
                    "Tokenizer could not preserve the original token sequence at a window boundary"
                )

            model_input = self.tokenizer(
                window_text,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                verbose=False,
            )["input_ids"]
            if len(model_input) > self.max_tokens:
                raise ValueError(
                    "A generated text window exceeds the model's maximum input length"
                )
            window_texts.append(window_text)
            token_start = token_end
            char_start = char_end

        if "".join(window_texts) != chunk.text:
            raise ValueError("Generated windows do not preserve all source text")

        window_count = len(window_texts)
        return [
            LiteratureEmbeddingWindow(
                text=window_text,
                pmid=chunk.pmid,
                pmcid=chunk.pmcid,
                doi=chunk.doi,
                section_title=chunk.section_title,
                normalized_section_title=chunk.normalized_section_title,
                section_path=deepcopy(chunk.section_path),
                category=chunk.category,
                source=chunk.source,
                literature_chunk_index=chunk.chunk_index,
                embedding_window_index=window_index,
                embedding_window_count=window_count,
                metadata=deepcopy(chunk.metadata),
            )
            for window_index, window_text in enumerate(window_texts)
        ]

    def split_chunks(
        self, chunks: List[LiteratureChunk]
    ) -> List[LiteratureEmbeddingWindow]:
        """Split chunks independently and return windows in input order."""
        return [
            window
            for chunk in chunks
            for window in self.split_chunk(chunk)
        ]