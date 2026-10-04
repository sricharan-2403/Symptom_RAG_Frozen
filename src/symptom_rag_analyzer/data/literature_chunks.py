from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LiteratureChunk:
    """A retrieval-ready chunk derived from a biomedical literature section."""

    text: str
    pmid: str
    pmcid: Optional[str] = None
    doi: Optional[str] = None
    section_title: str = ""
    normalized_section_title: str = ""
    section_path: List[str] = field(default_factory=list)
    category: str = "other"
    source: str = "body"
    level: int = 1
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
