from symptom_rag_analyzer.data.literature_chunks import LiteratureChunk


def test_literature_chunk_minimal_construction_uses_defaults():
    chunk = LiteratureChunk(text="Clinical text.", pmid="12345678")

    assert chunk.text == "Clinical text."
    assert chunk.pmid == "12345678"
    assert chunk.pmcid is None
    assert chunk.doi is None
    assert chunk.section_title == ""
    assert chunk.normalized_section_title == ""
    assert chunk.section_path == []
    assert chunk.category == "other"
    assert chunk.source == "body"
    assert chunk.level == 1
    assert chunk.chunk_index == 0
    assert chunk.metadata == {}


def test_literature_chunk_full_construction_preserves_fields():
    metadata = {"journal": "Example Journal", "publication_year": 2025}
    chunk = LiteratureChunk(
        text="A chunk of biomedical literature.",
        pmid="12345678",
        pmcid="PMC1234567",
        doi="10.1234/example",
        section_title="Clinical Presentation",
        normalized_section_title="clinical presentation",
        section_path=["Results", "Clinical Presentation"],
        category="clinical_presentation",
        source="body",
        level=2,
        chunk_index=3,
        metadata=metadata,
    )

    assert chunk.pmcid == "PMC1234567"
    assert chunk.doi == "10.1234/example"
    assert chunk.section_title == "Clinical Presentation"
    assert chunk.normalized_section_title == "clinical presentation"
    assert chunk.section_path == ["Results", "Clinical Presentation"]
    assert chunk.category == "clinical_presentation"
    assert chunk.source == "body"
    assert chunk.level == 2
    assert chunk.chunk_index == 3
    assert chunk.metadata == metadata


def test_metadata_default_is_independent_between_instances():
    first = LiteratureChunk(text="First.", pmid="1")
    second = LiteratureChunk(text="Second.", pmid="2")

    first.metadata["source"] = "first-only"

    assert first.metadata == {"source": "first-only"}
    assert second.metadata == {}


def test_section_path_default_is_independent_between_instances():
    first = LiteratureChunk(text="First.", pmid="1")
    second = LiteratureChunk(text="Second.", pmid="2")

    first.section_path.append("Introduction")

    assert first.section_path == ["Introduction"]
    assert second.section_path == []
