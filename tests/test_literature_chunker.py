import pytest

from symptom_rag_analyzer.data.literature_chunker import LiteratureSectionChunker
from symptom_rag_analyzer.data.literature_models import LiteratureArticle, LiteratureSection


def make_section(text, title="Introduction", source="body", level=1, section_path=None):
    return LiteratureSection(
        title=title,
        text=text,
        normalized_title=title.casefold(),
        section_path=section_path if section_path is not None else [title],
        category="introduction",
        source=source,
        level=level,
    )


def make_article(*, abstract_sections=None, body_sections=None):
    return LiteratureArticle(
        pmid="12345678",
        pmcid="PMC1234567",
        doi="10.1234/example",
        abstract_sections=abstract_sections or [],
        body_sections=body_sections or [],
    )


def test_short_section_produces_exactly_one_chunk():
    article = make_article(body_sections=[make_section("Short clinical section.")])

    chunks = LiteratureSectionChunker().chunk_article(article)

    assert len(chunks) == 1
    assert chunks[0].text == "Short clinical section."
    assert chunks[0].chunk_index == 0


def test_small_paragraphs_are_grouped_without_exceeding_target_size():
    paragraphs = ["a" * 400, "b" * 450, "c" * 300]
    article = make_article(body_sections=[make_section("\n\n".join(paragraphs))])

    chunks = LiteratureSectionChunker(target_size=1000, overlap=150).chunk_article(article)

    assert [len(chunk.text) for chunk in chunks] == [852, 300]
    assert chunks[0].text == "a" * 400 + "\n\n" + "b" * 450
    assert chunks[1].text == "c" * 300


def test_oversized_paragraph_uses_configured_character_overlap():
    text = "0123456789" * 280
    article = make_article(body_sections=[make_section(text)])

    chunks = LiteratureSectionChunker(target_size=1000, overlap=150).chunk_article(article)

    assert [len(chunk.text) for chunk in chunks] == [1000, 1000, 1000, 250]
    assert chunks[0].text[850:] == chunks[1].text[:150]
    assert chunks[1].text[850:] == chunks[2].text[:150]
    assert chunks[2].text[850:] == chunks[3].text[:150]
    assert "".join([chunks[0].text] + [chunk.text[150:] for chunk in chunks[1:]]) == text


@pytest.mark.parametrize(
    ("paragraph_length", "expected_slices"),
    [
        (1000, [(0, 1000)]),
        (1001, [(0, 1000), (850, 1001)]),
        (1701, [(0, 1000), (850, 1701)]),
        (1850, [(0, 1000), (850, 1850)]),
    ],
)
def test_oversized_paragraph_windows_do_not_emit_redundant_tails(
    paragraph_length, expected_slices
):
    text = "x" * paragraph_length
    article = make_article(body_sections=[make_section(text)])

    chunks = LiteratureSectionChunker(target_size=1000, overlap=150).chunk_article(article)

    assert [chunk.text for chunk in chunks] == [
        text[start:end] for start, end in expected_slices
    ]


def test_different_sections_never_share_a_chunk():
    article = make_article(body_sections=[
        make_section("First section text.", title="First"),
        make_section("Second section text.", title="Second"),
    ])

    chunks = LiteratureSectionChunker().chunk_article(article)

    assert [chunk.text for chunk in chunks] == ["First section text.", "Second section text."]
    assert [chunk.section_title for chunk in chunks] == ["First", "Second"]


def test_section_metadata_and_hierarchy_are_preserved():
    section = LiteratureSection(
        title="Clinical Presentation",
        text="Relevant text.",
        normalized_title="clinical presentation",
        section_path=["Results", "Clinical Presentation"],
        category="clinical_presentation",
        source="abstract",
        level=2,
    )
    article = make_article(abstract_sections=[section])

    chunk = LiteratureSectionChunker().chunk_article(article)[0]

    assert chunk.pmid == "12345678"
    assert chunk.pmcid == "PMC1234567"
    assert chunk.doi == "10.1234/example"
    assert chunk.section_title == "Clinical Presentation"
    assert chunk.normalized_section_title == "clinical presentation"
    assert chunk.section_path == ["Results", "Clinical Presentation"]
    assert chunk.category == "clinical_presentation"
    assert chunk.source == "abstract"
    assert chunk.level == 2


def test_empty_section_produces_zero_chunks():
    article = make_article(body_sections=[make_section(" \n  ")])

    assert LiteratureSectionChunker().chunk_article(article) == []


def test_abstract_sections_are_processed_before_body_sections():
    article = make_article(
        abstract_sections=[make_section("Abstract text.", "Abstract", "abstract")],
        body_sections=[make_section("Body text.", "Introduction", "body")],
    )

    chunks = LiteratureSectionChunker().chunk_article(article)

    assert [chunk.text for chunk in chunks] == ["Abstract text.", "Body text."]
    assert [chunk.source for chunk in chunks] == ["abstract", "body"]


def test_chunk_indexes_restart_for_each_section():
    article = make_article(body_sections=[
        make_section("a" * 1200, title="First"),
        make_section("b" * 1200, title="Second"),
    ])

    chunks = LiteratureSectionChunker(target_size=500, overlap=100).chunk_article(article)

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 0, 1, 2]
    assert [chunk.section_title for chunk in chunks] == ["First"] * 3 + ["Second"] * 3


def test_repeated_execution_produces_identical_chunks():
    article = make_article(body_sections=[
        make_section("alpha\n\nbeta\n\n" + "x" * 1400, title="Results")
    ])
    chunker = LiteratureSectionChunker(target_size=500, overlap=75)

    first = chunker.chunk_article(article)
    second = chunker.chunk_article(article)

    assert first == second


@pytest.mark.parametrize(
    ("target_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (5, -1),
        (5, 5),
        (5, 6),
    ],
)
def test_invalid_chunker_settings_raise_value_error(target_size, overlap):
    with pytest.raises(ValueError):
        LiteratureSectionChunker(target_size=target_size, overlap=overlap)
