from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LiteratureSection:
    """A section of abstract or body text from a literature article."""

    title: str
    text: str
    normalized_title: str = ""
    section_path: List[str] = field(default_factory=list)
    category: str = "other"
    source: str = "body"
    level: int = 1


@dataclass
class LiteratureArticle:
    """Metadata and extracted sections for a biomedical literature article."""

    pmid: str
    pmcid: Optional[str] = None
    doi: Optional[str] = None
    title: str = ""
    journal: str = ""
    publication_date: str = ""
    publication_year: Optional[int] = None
    record_type: str = ""
    article_type: str = ""
    publication_types: List[str] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    mesh_terms: List[str] = field(default_factory=list)
    abstract_sections: List[LiteratureSection] = field(default_factory=list)
    body_sections: List[LiteratureSection] = field(default_factory=list)
    language: str = ""
    license: str = ""
    source_file: str = ""
    source_type: str = "pubmed-pmc-literature"

    def all_sections(self) -> List[LiteratureSection]:
        """Return abstract sections followed by body sections."""
        return self.abstract_sections + self.body_sections

    def has_body(self) -> bool:
        """Return whether any body section contains non-whitespace text."""
        return any(section.text.strip() for section in self.body_sections)

    def has_abstract(self) -> bool:
        """Return whether any abstract section contains non-whitespace text."""
        return any(section.text.strip() for section in self.abstract_sections)
