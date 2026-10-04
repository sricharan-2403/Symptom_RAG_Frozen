import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Union

from .literature_models import LiteratureArticle, LiteratureSection


_SECTION_TAGS = {"sec", "abstract-sec", "abstract-section"}
_IGNORED_TEXT_TAGS = {
    "back",
    "caption",
    "fig",
    "fig-group",
    "fn",
    "fn-group",
    "graphic",
    "inline-graphic",
    "media",
    "ref",
    "ref-list",
    "reference-list",
    "supplementary-material",
    "supplementary-material-group",
    "table",
    "table-wrap",
    "table-wrap-group",
}
_METADATA_STOP_TAGS = {
    "article",
    "back",
    "ref",
    "ref-list",
    "related-article",
    "related-object",
    "sub-article",
    "supplementary-material",
}
_BLOCK_TEXT_TAGS = {"disp-quote", "list", "list-item", "p"}
_CATEGORY_TITLES = {
    "introduction": "introduction",
    "background": "background",
    "methods": "methods",
    "materials and methods": "methods",
    "methodology": "methods",
    "results": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "diagnosis": "diagnosis",
    "treatment": "treatment",
    "management": "management",
    "prognosis": "prognosis",
    "clinical presentation": "clinical_presentation",
    "differential diagnosis": "differential_diagnosis",
    "pathophysiology": "pathophysiology",
}


def _local_name(name: str) -> str:
    """Return an XML element or attribute name without its namespace."""
    return name.rsplit("}", 1)[-1].casefold()


def _attribute(element: ET.Element, name: str) -> str:
    for key, value in element.attrib.items():
        if _local_name(key) == name.casefold():
            return value
    return ""


def _children(element: Optional[ET.Element], name: str) -> List[ET.Element]:
    if element is None:
        return []
    return [child for child in element if _local_name(child.tag) == name.casefold()]


def _first_child(element: Optional[ET.Element], name: str) -> Optional[ET.Element]:
    matches = _children(element, name)
    return matches[0] if matches else None


def _scoped_descendants(
    element: Optional[ET.Element],
    names: Set[str],
    stop_tags: Set[str],
) -> Iterable[ET.Element]:
    if element is None:
        return
    for child in element:
        child_name = _local_name(child.tag)
        if child_name in stop_tags:
            continue
        if child_name in names:
            yield child
        yield from _scoped_descendants(child, names, stop_tags)


def _normalize_text(value: str) -> str:
    """Collapse XML whitespace while preserving the source wording."""
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_section_title(title: str) -> str:
    """Remove obvious leading numbering and lowercase a section title."""
    normalized = _normalize_text(title)
    normalized = re.sub(r"^\d+(?:\.\d+)*[.)]?\s+", "", normalized)
    normalized = normalized.rstrip(":").strip()
    return normalized.casefold()


def categorize_section(title: str) -> str:
    """Assign a category only for an obvious exact section-title match."""
    return _CATEGORY_TITLES.get(normalize_section_title(title), "other")


class PMCLiteratureParser:
    """Parse one PMC JATS XML file into a LiteratureArticle."""

    def parse_file(self, xml_path: Union[str, Path]) -> LiteratureArticle:
        """Parse a PMC XML file, raising a clear error for malformed or missing articles."""
        path = Path(xml_path)
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError) as error:
            raise ValueError(f"Could not parse PMC XML file {path}: {error}") from error

        article = self._find_article(root)
        if article is None:
            raise ValueError(f"No article element found in PMC XML file {path}.")

        front = _first_child(article, "front")
        article_meta = _first_child(front, "article-meta")
        journal_meta = _first_child(front, "journal-meta")
        identifiers = self._extract_identifiers(article_meta)
        publication_date = self._extract_publication_date(article_meta, journal_meta)

        return LiteratureArticle(
            pmid=identifiers.get("pmid", ""),
            pmcid=identifiers.get("pmcid") or None,
            doi=identifiers.get("doi") or None,
            title=self._extract_article_title(article_meta),
            journal=self._extract_journal_title(journal_meta),
            publication_date=publication_date,
            publication_year=self._extract_year(publication_date),
            record_type=_attribute(article, "record-type"),
            article_type=_attribute(article, "article-type"),
            publication_types=self._extract_publication_types(article_meta),
            authors=self._extract_authors(article_meta),
            mesh_terms=self._extract_mesh_terms(article_meta),
            abstract_sections=self._extract_abstract_sections(article_meta),
            body_sections=self._extract_body_sections(article),
            language=self._extract_language(article, article_meta, root),
            license=self._extract_license(article_meta),
            source_file=str(path),
        )

    def _find_article(self, root: ET.Element) -> Optional[ET.Element]:
        if _local_name(root.tag) == "article":
            return root

        articles = [
            element for element in root.iter()
            if _local_name(element.tag) == "article"
        ]
        for article in articles:
            front = _first_child(article, "front")
            if _first_child(front, "article-meta") is not None:
                return article
        return articles[0] if articles else None

    def _extract_identifiers(self, article_meta: Optional[ET.Element]) -> dict[str, str]:
        identifiers = {"pmid": "", "pmcid": "", "doi": ""}
        id_tags = {"article-id", "pub-id", "pmid", "pmcid", "doi"}
        for element in _scoped_descendants(article_meta, id_tags, _METADATA_STOP_TAGS):
            tag = _local_name(element.tag)
            id_type = _attribute(element, "pub-id-type").casefold().replace("-", "")
            value = _normalize_text("".join(element.itertext()))
            if not value:
                continue

            if tag == "pmid" or (tag in {"article-id", "pub-id"} and id_type == "pmid"):
                identifiers["pmid"] = identifiers["pmid"] or value
            elif tag == "pmcid" or (
                tag in {"article-id", "pub-id"} and id_type in {"pmc", "pmcid"}
            ):
                identifiers["pmcid"] = identifiers["pmcid"] or self._normalize_pmcid(value)
            elif tag == "doi" or (
                tag in {"article-id", "pub-id"} and id_type == "doi"
            ):
                identifiers["doi"] = identifiers["doi"] or value

        for element in _scoped_descendants(article_meta, {"elocation-id"}, _METADATA_STOP_TAGS):
            if _attribute(element, "eidtype").casefold() == "doi" and not identifiers["doi"]:
                identifiers["doi"] = _normalize_text("".join(element.itertext()))
        return identifiers

    @staticmethod
    def _normalize_pmcid(value: str) -> str:
        value = value.strip()
        if value and not value.casefold().startswith("pmc"):
            return f"PMC{value}"
        return value

    def _extract_article_title(self, article_meta: Optional[ET.Element]) -> str:
        title_group = next(
            _scoped_descendants(article_meta, {"title-group"}, _METADATA_STOP_TAGS),
            None,
        )
        title = _first_child(title_group, "article-title")
        return self._collect_text(title, _IGNORED_TEXT_TAGS)

    def _extract_journal_title(self, journal_meta: Optional[ET.Element]) -> str:
        title_group = next(
            _scoped_descendants(journal_meta, {"journal-title-group"}, _METADATA_STOP_TAGS),
            None,
        )
        title = _first_child(title_group, "journal-title")
        if title is None:
            title = next(
                _scoped_descendants(journal_meta, {"journal-title"}, _METADATA_STOP_TAGS),
                None,
            )
        return self._collect_text(title, _IGNORED_TEXT_TAGS)

    def _extract_publication_date(
        self,
        article_meta: Optional[ET.Element],
        journal_meta: Optional[ET.Element],
    ) -> str:
        dates = list(_scoped_descendants(article_meta, {"pub-date", "date"}, _METADATA_STOP_TAGS))
        dates.extend(_scoped_descendants(journal_meta, {"pub-date", "date"}, _METADATA_STOP_TAGS))
        dates.extend(_scoped_descendants(article_meta, {"date-in-citation"}, _METADATA_STOP_TAGS))

        for date in dates:
            parts = []
            for child_name in ("year", "month", "day", "season", "string-date"):
                child = _first_child(date, child_name)
                text = self._collect_text(child, _IGNORED_TEXT_TAGS)
                if text:
                    parts.append(text)
            if parts:
                return " ".join(parts)
            if _local_name(date.tag) in {"date-in-citation", "string-date"}:
                text = self._collect_text(date, _IGNORED_TEXT_TAGS)
                if text:
                    return text
        return ""

    @staticmethod
    def _extract_year(publication_date: str) -> Optional[int]:
        match = re.search(r"\b(\d{4})\b", publication_date)
        return int(match.group(1)) if match else None

    def _extract_publication_types(self, article_meta: Optional[ET.Element]) -> List[str]:
        types = []
        for element in _scoped_descendants(
            article_meta,
            {"publication-type", "pub-type"},
            _METADATA_STOP_TAGS,
        ):
            value = self._collect_text(element, _IGNORED_TEXT_TAGS)
            if value and value not in types:
                types.append(value)
        for group in _scoped_descendants(article_meta, {"subj-group"}, _METADATA_STOP_TAGS):
            group_type = _attribute(group, "subj-group-type").casefold().replace("_", "-")
            if group_type not in {"article-type", "publication-type", "pub-type"}:
                continue
            for subject in _scoped_descendants(group, {"subject"}, _METADATA_STOP_TAGS):
                value = self._collect_text(subject, _IGNORED_TEXT_TAGS)
                if value and value not in types:
                    types.append(value)
        return types

    def _extract_authors(self, article_meta: Optional[ET.Element]) -> List[str]:
        authors = []
        contributors = _scoped_descendants(article_meta, {"contrib"}, _METADATA_STOP_TAGS)
        for contributor in contributors:
            contrib_type = _attribute(contributor, "contrib-type").casefold()
            if contrib_type and contrib_type != "author":
                continue
            name = self._contributor_name(contributor)
            if name and name not in authors:
                authors.append(name)

        if authors:
            return authors

        for author in _scoped_descendants(article_meta, {"author"}, _METADATA_STOP_TAGS):
            name = self._contributor_name(author)
            if name and name not in authors:
                authors.append(name)
        return authors

    def _contributor_name(self, contributor: ET.Element) -> str:
        collab = next(
            _scoped_descendants(contributor, {"collab", "collaboration"}, _METADATA_STOP_TAGS),
            None,
        )
        if collab is not None:
            return self._collect_text(collab, _IGNORED_TEXT_TAGS)

        name = next(
            _scoped_descendants(contributor, {"name", "string-name"}, _METADATA_STOP_TAGS),
            None,
        )
        if name is None:
            return ""

        given = self._collect_text(_first_child(name, "given-names"), _IGNORED_TEXT_TAGS)
        surname = self._collect_text(_first_child(name, "surname"), _IGNORED_TEXT_TAGS)
        suffix = self._collect_text(_first_child(name, "suffix"), _IGNORED_TEXT_TAGS)
        parts = [part for part in (given, surname, suffix) if part]
        return " ".join(parts) if parts else self._collect_text(name, _IGNORED_TEXT_TAGS)

    def _extract_mesh_terms(self, article_meta: Optional[ET.Element]) -> List[str]:
        terms = []
        for heading in _scoped_descendants(
            article_meta,
            {"mesh-heading", "meshheading"},
            _METADATA_STOP_TAGS,
        ):
            descriptor = next(
                _scoped_descendants(
                    heading,
                    {"descriptor-name", "mesh-term"},
                    _METADATA_STOP_TAGS,
                ),
                None,
            )
            value = self._collect_text(descriptor if descriptor is not None else heading, _IGNORED_TEXT_TAGS)
            if value and value not in terms:
                terms.append(value)

        for group in _scoped_descendants(article_meta, {"kwd-group"}, _METADATA_STOP_TAGS):
            if "mesh" not in _attribute(group, "kwd-group-type").casefold():
                continue
            for keyword in _scoped_descendants(group, {"kwd"}, _METADATA_STOP_TAGS):
                value = self._collect_text(keyword, _IGNORED_TEXT_TAGS)
                if value and value not in terms:
                    terms.append(value)
        return terms

    def _extract_language(
        self,
        article: ET.Element,
        article_meta: Optional[ET.Element],
        root: ET.Element,
    ) -> str:
        for element in (article, root):
            language = _attribute(element, "lang")
            if language:
                return language
        language_element = next(
            _scoped_descendants(article_meta, {"language"}, _METADATA_STOP_TAGS),
            None,
        )
        return self._collect_text(language_element, _IGNORED_TEXT_TAGS)

    def _extract_license(self, article_meta: Optional[ET.Element]) -> str:
        values = []
        for element in _scoped_descendants(
            article_meta,
            {"license", "license-p", "license-ref"},
            _METADATA_STOP_TAGS,
        ):
            text = self._collect_text(element, _IGNORED_TEXT_TAGS)
            href = _attribute(element, "href")
            for value in (text, href):
                if value and value not in values:
                    values.append(value)
        return " | ".join(values)

    def _extract_abstract_sections(
        self,
        article_meta: Optional[ET.Element],
    ) -> List[LiteratureSection]:
        abstracts = list(
            _scoped_descendants(
                article_meta,
                {"abstract", "trans-abstract"},
                _METADATA_STOP_TAGS,
            )
        )
        sections = []
        for abstract in abstracts:
            title_node = _first_child(abstract, "title")
            title = self._collect_text(title_node, _IGNORED_TEXT_TAGS)
            if not title:
                title = "Abstract"

            nested_sections = [
                child for child in abstract
                if _local_name(child.tag) in _SECTION_TAGS
            ]
            if nested_sections:
                own_text = self._collect_text(
                    abstract,
                    _IGNORED_TEXT_TAGS | {"title"} | _SECTION_TAGS,
                )
                if own_text:
                    sections.append(self._make_section(title, own_text, [title], "abstract", 1))
                parent_path = [] if title.casefold() == "abstract" else [title]
                for child in nested_sections:
                    sections.extend(
                        self._parse_section_tree(
                            child,
                            parent_path,
                            len(parent_path) + 1,
                            "abstract",
                        )
                    )
                continue

            text = self._collect_text(abstract, _IGNORED_TEXT_TAGS | {"title"})
            if text:
                sections.append(self._make_section(title, text, [title], "abstract", 1))
        return sections

    def _extract_body_sections(self, article: ET.Element) -> List[LiteratureSection]:
        body = next(
            _scoped_descendants(article, {"body"}, {"back", "ref-list", "sub-article"}),
            None,
        )
        if body is None:
            return []

        sections = []
        for child in body:
            if _local_name(child.tag) in _SECTION_TAGS:
                sections.extend(self._parse_section_tree(child, [], 1, "body"))
        return sections

    def _parse_section_tree(
        self,
        element: ET.Element,
        parent_path: Sequence[str],
        level: int,
        source: str,
    ) -> List[LiteratureSection]:
        title_node = _first_child(element, "title")
        title = self._collect_text(title_node, _IGNORED_TEXT_TAGS)
        if not title:
            title = self._collect_text(_first_child(element, "label"), _IGNORED_TEXT_TAGS)
        if not title:
            title = "Untitled section"

        section_path = list(parent_path) + [title]
        text = self._collect_text(
            element,
            _IGNORED_TEXT_TAGS | _SECTION_TAGS | {"title", "label"},
        )
        sections = []
        if text:
            sections.append(self._make_section(title, text, section_path, source, level))

        for child in element:
            if _local_name(child.tag) in _SECTION_TAGS:
                sections.extend(self._parse_section_tree(child, section_path, level + 1, source))
        return sections

    @staticmethod
    def _make_section(
        title: str,
        text: str,
        section_path: Sequence[str],
        source: str,
        level: int,
    ) -> LiteratureSection:
        return LiteratureSection(
            title=title,
            text=text,
            normalized_title=normalize_section_title(title),
            section_path=list(section_path),
            category=categorize_section(title),
            source=source,
            level=level,
        )

    def _collect_text(
        self,
        element: Optional[ET.Element],
        excluded_tags: Set[str],
        excluded_elements: Optional[Set[int]] = None,
    ) -> str:
        """Collect normalized mixed-content text without excluded subtrees."""
        if element is None:
            return ""
        return _normalize_text(
            self._collect_text_raw(element, excluded_tags, excluded_elements or set())
        )

    def _collect_text_raw(
        self,
        element: ET.Element,
        excluded_tags: Set[str],
        excluded_elements: Set[int],
    ) -> str:
        pieces = []

        def append_text(value: str, force_word_boundary: bool = False) -> None:
            if not value:
                return
            previous = next((piece for piece in reversed(pieces) if piece), "")
            if previous and not previous[-1].isspace() and not value[0].isspace():
                if force_word_boundary and previous[-1].isalnum() and value[0].isalnum():
                    pieces.append(" ")
                elif previous[-1].isalpha() and value[0].isalpha():
                    pieces.append(" ")
            pieces.append(value)

        append_text(element.text or "")
        for child in element:
            child_name = _local_name(child.tag)
            if child_name not in excluded_tags and id(child) not in excluded_elements:
                child_text = self._collect_text_raw(child, excluded_tags, excluded_elements)
                if child_name in _BLOCK_TEXT_TAGS:
                    append_text(" ")
                append_text(child_text, force_word_boundary=child_name == "xref")
                if child_name in _BLOCK_TEXT_TAGS:
                    append_text(" ")
            if child.tail:
                append_text(child.tail, force_word_boundary=child_name == "xref")
        return "".join(pieces)

