import re
from types import SimpleNamespace

import pytest

from symptom_rag_analyzer.data.literature_chunks import LiteratureChunk
from symptom_rag_analyzer.embeddings.literature_embedding_windows import (
    LiteratureEmbeddingWindowSplitter,
)


class TokenizerEncoding(dict):
    def __init__(self, values, word_ids):
        super().__init__(values)
        self._word_ids = word_ids

    def word_ids(self):
        return self._word_ids


class WhitespaceTokenizer:
    is_fast = True

    def __init__(self):
        self.token_ids = {}

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        return_offsets_mapping=False,
        **_kwargs,
    ):
        matches = list(re.finditer(r"\S+", text))
        content_ids = []
        content_offsets = []
        content_word_ids = []
        for word_index, match in enumerate(matches):
            if match.group() == "compound":
                content_ids.extend([100001, 100002])
                content_offsets.extend([
                    (match.start(), match.start() + 4),
                    (match.start() + 4, match.end()),
                ])
                content_word_ids.extend([word_index, word_index])
            else:
                content_ids.append(
                    self.token_ids.setdefault(match.group(), len(self.token_ids) + 1)
                )
                content_offsets.append((match.start(), match.end()))
                content_word_ids.append(word_index)
        input_ids = [101, *content_ids, 102] if add_special_tokens else content_ids
        word_ids = [None, *content_word_ids, None] if add_special_tokens else content_word_ids
        result = TokenizerEncoding({"input_ids": input_ids}, word_ids)
        if return_offsets_mapping:
            result["offset_mapping"] = (
                [(0, 0), *content_offsets, (0, 0)]
                if add_special_tokens
                else content_offsets
            )
        return result


def make_splitter():
    tokenizer = WhitespaceTokenizer()
    model = SimpleNamespace(tokenizer=tokenizer, max_seq_length=100)
    return LiteratureEmbeddingWindowSplitter(SimpleNamespace(model=model)), tokenizer


def make_chunk(text, *, chunk_index=7, metadata=None, section_path=None):
    return LiteratureChunk(
        text=text,
        pmid="12345678",
        pmcid="PMC1234567",
        doi="10.1234/example",
        section_title="Results",
        normalized_section_title="results",
        section_path=section_path if section_path is not None else ["Results", "Findings"],
        category="results",
        source="body",
        chunk_index=chunk_index,
        metadata=metadata if metadata is not None else {"nested": {"labels": ["source"]}},
    )


def text_with_model_token_count(count):
    content_token_count = count - 2
    assert content_token_count > 0
    return " ".join(f"token{index}" for index in range(content_token_count))


def content_token_ids(tokenizer, text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def model_input_token_count(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def test_short_chunk_requires_one_window():
    splitter, tokenizer = make_splitter()
    chunk = make_chunk("brief finding here")

    windows = splitter.split_chunk(chunk)

    assert len(windows) == 1
    assert windows[0].text == chunk.text
    assert model_input_token_count(tokenizer, windows[0].text) <= 100


@pytest.mark.parametrize(
    ("model_token_count", "expected_window_count"),
    [(100, 1), (101, 2), (200, 3), (201, 3)],
)
def test_window_counts_at_model_input_boundaries(model_token_count, expected_window_count):
    splitter, tokenizer = make_splitter()
    chunk = make_chunk(text_with_model_token_count(model_token_count))

    windows = splitter.split_chunk(chunk)

    assert len(windows) == expected_window_count
    assert all(model_input_token_count(tokenizer, window.text) <= 100 for window in windows)


def test_long_chunk_requires_multiple_windows_and_preserves_order():
    splitter, _tokenizer = make_splitter()
    chunk = make_chunk(text_with_model_token_count(805))

    windows = splitter.split_chunk(chunk)

    assert len(windows) == 9
    assert [window.embedding_window_index for window in windows] == list(range(9))
    assert all(window.embedding_window_count == 9 for window in windows)


def test_windows_do_not_overlap_and_reconstruct_original_token_sequence():
    splitter, tokenizer = make_splitter()
    chunk = make_chunk(text_with_model_token_count(201))

    windows = splitter.split_chunk(chunk)

    original_ids = content_token_ids(tokenizer, chunk.text)
    reconstructed_ids = [
        token_id
        for window in windows
        for token_id in content_token_ids(tokenizer, window.text)
    ]
    assert reconstructed_ids == original_ids
    assert "".join(window.text for window in windows) == chunk.text


def test_window_boundary_does_not_split_a_wordpiece_group():
    tokenizer = WhitespaceTokenizer()
    splitter = LiteratureEmbeddingWindowSplitter(
        SimpleNamespace(model=SimpleNamespace(tokenizer=tokenizer, max_seq_length=100))
    )
    text = " ".join([f"word{index}" for index in range(97)] + ["compound", "tail"])
    chunk = make_chunk(text)

    windows = splitter.split_chunk(chunk)

    original_ids = content_token_ids(tokenizer, text)
    reconstructed_ids = [
        token_id
        for window in windows
        for token_id in content_token_ids(tokenizer, window.text)
    ]
    assert len(windows) == 2
    assert reconstructed_ids == original_ids
    assert all(model_input_token_count(tokenizer, window.text) <= 100 for window in windows)


def test_no_window_exceeds_model_input_limit():
    splitter, tokenizer = make_splitter()
    chunk = make_chunk(text_with_model_token_count(856))

    windows = splitter.split_chunk(chunk)

    assert all(model_input_token_count(tokenizer, window.text) <= 100 for window in windows)


def test_repeated_splitting_is_deterministic():
    splitter, _tokenizer = make_splitter()
    chunk = make_chunk(text_with_model_token_count(305))

    assert splitter.split_chunk(chunk) == splitter.split_chunk(chunk)


def test_provenance_is_preserved_on_every_window():
    splitter, _tokenizer = make_splitter()
    chunk = make_chunk(text_with_model_token_count(201), chunk_index=12)

    windows = splitter.split_chunk(chunk)

    for index, window in enumerate(windows):
        assert window.pmid == chunk.pmid
        assert window.pmcid == chunk.pmcid
        assert window.doi == chunk.doi
        assert window.section_title == chunk.section_title
        assert window.normalized_section_title == chunk.normalized_section_title
        assert window.section_path == chunk.section_path
        assert window.category == chunk.category
        assert window.source == chunk.source
        assert window.literature_chunk_index == 12
        assert window.embedding_window_index == index
        assert window.embedding_window_count == len(windows)


def test_section_paths_are_independent_between_windows_and_source():
    splitter, _tokenizer = make_splitter()
    chunk = make_chunk(text_with_model_token_count(201))

    windows = splitter.split_chunk(chunk)
    windows[0].section_path.append("changed")

    assert windows[1].section_path == ["Results", "Findings"]
    assert chunk.section_path == ["Results", "Findings"]


def test_metadata_is_independent_between_windows_and_source():
    splitter, _tokenizer = make_splitter()
    chunk = make_chunk(text_with_model_token_count(201))

    windows = splitter.split_chunk(chunk)
    windows[0].metadata["nested"]["labels"].append("changed")

    assert windows[1].metadata == {"nested": {"labels": ["source"]}}
    assert chunk.metadata == {"nested": {"labels": ["source"]}}


@pytest.mark.parametrize("text", ["", " \n\t "])
def test_empty_or_whitespace_chunk_raises_value_error(text):
    splitter, _tokenizer = make_splitter()

    with pytest.raises(ValueError, match="cannot be empty or whitespace"):
        splitter.split_chunk(make_chunk(text))


def test_multiple_chunks_remain_isolated_and_in_input_order():
    splitter, _tokenizer = make_splitter()
    first = make_chunk("alpha beta", chunk_index=0)
    second = make_chunk("gamma delta", chunk_index=1)

    windows = splitter.split_chunks([first, second])

    assert [window.text for window in windows] == ["alpha beta", "gamma delta"]
    assert [window.literature_chunk_index for window in windows] == [0, 1]