import pytest

from symptom_rag_analyzer.data.chunker import TextChunker
from symptom_rag_analyzer.data.documents import Document


def test_chunk_documents_splits_text_with_overlap_and_preserves_document_reference():
    document = Document(
        filename="symptom_note.txt",
        page_number=1,
        text="abcdefghij",
        source_type="TXT",
    )
    chunker = TextChunker(chunk_size=4, overlap=2)

    chunks = chunker.chunk_documents([document])

    assert len(chunks) == 4
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3]
    assert [chunk.text for chunk in chunks] == ["abcd", "cdef", "efgh", "ghij"]
    assert all(chunk.document is document for chunk in chunks)


def test_chunk_documents_ignores_empty_or_whitespace_only_documents():
    documents = [
        Document(filename="empty.txt", page_number=1, text="   \n\t  ", source_type="TXT"),
        Document(filename="real.txt", page_number=1, text="hello world", source_type="TXT"),
    ]
    chunker = TextChunker(chunk_size=5, overlap=1)

    chunks = chunker.chunk_documents(documents)

    assert len(chunks) == 3
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [chunk.document.filename for chunk in chunks] == ["real.txt", "real.txt", "real.txt"]
    assert [chunk.text for chunk in chunks] == ["hello", "o wor", "rld"]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (5, 5),
        (5, -1),
    ],
)
def test_invalid_chunk_settings_raise_value_error(chunk_size, overlap):
    with pytest.raises(ValueError):
        TextChunker(chunk_size=chunk_size, overlap=overlap)
