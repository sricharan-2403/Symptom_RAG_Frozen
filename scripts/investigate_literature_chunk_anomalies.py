import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PMC_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "pmc"
TARGET_SIZE = 1000
OVERLAP = 150

SHORT_CASES = (
    ("37065189", "PMC10102637", "Nervous system"),
    ("37197144", "PMC10187598", "Introduction"),
    ("37426954", "PMC10323732", "Strengths and Limitations"),
    ("37499081", "PMC10374188", "Abstract"),
    ("37389507", "PMC10784847", "Recommendation-Specific Supportive Text"),
)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].casefold()


def normalized_title(title: str) -> str:
    return re.sub(r"\s+", " ", title or "").strip().casefold().rstrip(":").strip()


def direct_child(element: ET.Element, name: str) -> Optional[ET.Element]:
    return next((child for child in element if local_name(child.tag) == name), None)


def find_article_element(root: ET.Element) -> Optional[ET.Element]:
    if local_name(root.tag) == "article":
        return root
    return next(
        (element for element in root.iter() if local_name(element.tag) == "article"),
        None,
    )


def print_empty_article_structure(xml_path: Path) -> None:
    root = ET.parse(xml_path).getroot()
    article = find_article_element(root)
    print("XML structure because no sections were extracted:")
    print(f"  root children: {[local_name(child.tag) for child in root]}")
    if article is None:
        print("  no <article> element found")
        print(f"  root content elements: {[local_name(node.tag) for node in root.iter()]}")
        return

    front = direct_child(article, "front")
    metadata = direct_child(front, "article-meta") if front is not None else None
    body = direct_child(article, "body")
    abstract_nodes = [node for node in article.iter() if local_name(node.tag) in {"abstract", "trans-abstract"}]
    section_nodes = [node for node in article.iter() if local_name(node.tag) in {"sec", "abstract-sec", "abstract-section"}]
    paragraph_nodes = [node for node in article.iter() if local_name(node.tag) == "p"]
    print(f"  article direct children: {[local_name(child.tag) for child in article]}")
    print(f"  article_type attribute: {article.attrib.get('article-type', '(missing)')}")
    print(f"  article-meta children: {[local_name(child.tag) for child in metadata] if metadata is not None else '(missing)'}")
    print(f"  abstract elements: {len(abstract_nodes)}")
    print(f"  body direct children: {[local_name(child.tag) for child in body] if body is not None else '(missing)'}")
    print(f"  section elements: {len(section_nodes)}")
    print(f"  paragraph elements: {len(paragraph_nodes)}")
    print(f"  present content element names: {sorted({local_name(node.tag) for node in article.iter()})}")


def raw_xml_title_and_length(xml_path: Path, source: str, title: str) -> List[Tuple[str, int, str]]:
    root = ET.parse(xml_path).getroot()
    article = find_article_element(root)
    if article is None:
        return []

    candidate_names = {"sec", "abstract-sec", "abstract-section"} if source == "body" else {"abstract", "trans-abstract"}
    evidence = []
    for node in article.iter():
        if local_name(node.tag) not in candidate_names:
            continue
        title_node = direct_child(node, "title")
        node_title = " ".join("".join(title_node.itertext()).split()) if title_node is not None else ""
        if not node_title and source == "abstract":
            node_title = node.attrib.get("abstract-type", "") or "Abstract"
        if normalized_title(node_title) != normalized_title(title):
            continue
        raw_text = " ".join("".join(node.itertext()).split())
        evidence.append((node_title, len(raw_text), raw_text[:300]))
    return evidence


def chunks_by_source_section(article, chunker, article_chunks):
    sections = article.abstract_sections + article.body_sections
    mapped = []
    offset = 0
    for section in sections:
        if section.source == "abstract":
            isolated_article = replace(article, abstract_sections=[section], body_sections=[])
        else:
            isolated_article = replace(article, abstract_sections=[], body_sections=[section])
        section_chunks = chunker.chunk_article(isolated_article)
        full_run_chunks = article_chunks[offset:offset + len(section_chunks)]
        mapped.append((section, full_run_chunks))
        offset += len(section_chunks)
    return mapped, offset == len(article_chunks)


def section_assessment(section, chunks) -> str:
    if not chunks:
        return "No corresponding chunk was produced."

    shortest_index = min(range(len(chunks)), key=lambda index: (len(chunks[index].text), index))
    chunk = chunks[shortest_index]
    if len(section.text) > TARGET_SIZE and shortest_index > 0:
        previous = chunks[shortest_index - 1].text
        if previous.endswith(chunk.text):
            if not any(character.isalnum() for character in chunk.text):
                return (
                    "D) chunker artifact: punctuation-only trailing window duplicates "
                    "the previous window's final character (the text itself is B)."
                )
            return "D) trailing window duplicates the previous window's final characters."
    if len(chunk.text) > 1:
        return "The corresponding section has no one-character chunk."
    if not any(character.isalnum() for character in chunk.text):
        return "B) punctuation-only chunk text; no duplicated-window evidence found."
    if len(section.text) == len(chunk.text):
        return "A) the extracted section itself contains one meaningful character."
    return "The short chunk is present in parser output; source XML evidence is printed for comparison."


def print_case1(parser) -> str:
    xml_path = PMC_DIRECTORY / "PMC13088060.xml"
    print("\n" + "=" * 78 + "\nCASE 1: ZERO-SECTION ARTICLE\n" + "=" * 78)
    root = ET.parse(xml_path).getroot()
    article = parser.parse_file(xml_path)
    print(f"XML filename: {xml_path.name}")
    print(f"XML root: {root.tag}")
    print(f"article_type: {article.article_type}")
    print(f"article title: {article.title}")
    print(f"PMID: {article.pmid}")
    print(f"PMCID: {article.pmcid}")
    print(f"number of abstract sections: {len(article.abstract_sections)}")
    print(f"number of body sections: {len(article.body_sections)}")
    print(f"article.has_abstract(): {article.has_abstract()}")
    print(f"article.has_body(): {article.has_body()}")
    sections = article.all_sections()
    if sections:
        for section in sections:
            print(f"section title={section.title!r}; text length={len(section.text)}")
        return "Sections were extracted."

    print("No sections were extracted.")
    print_empty_article_structure(xml_path)
    body = direct_child(find_article_element(root), "body")
    direct_paragraphs = sum(local_name(child.tag) == "p" for child in body) if body is not None else 0
    direct_sections = sum(local_name(child.tag) == "sec" for child in body) if body is not None else 0
    abstract_count = sum(
        local_name(node.tag) in {"abstract", "trans-abstract"}
        for node in (find_article_element(root).iter() if find_article_element(root) is not None else [])
    )
    return (
        f"No sections extracted; XML has {abstract_count} abstract element(s), "
        f"{direct_paragraphs} direct body paragraph(s), and {direct_sections} direct body <sec> element(s)."
    )


def print_short_case(parser, chunker, pmid: str, pmcid: str, requested_title: str) -> Optional[str]:
    xml_path = PMC_DIRECTORY / f"{pmcid}.xml"
    print("\n" + "=" * 78)
    print(f"CASE 2: PMID {pmid} / {pmcid} / {requested_title}")
    print("=" * 78)
    article = parser.parse_file(xml_path)
    print(f"XML filename: {xml_path.name}")
    print(f"Parsed PMID: {article.pmid}")
    print(f"Parsed PMCID: {article.pmcid}")
    if article.pmid and article.pmid != pmid:
        print(f"WARNING: parsed PMID does not match requested PMID {pmid}")
    if article.pmcid and article.pmcid != pmcid:
        print(f"WARNING: parsed PMCID does not match requested PMCID {pmcid}")

    article_chunks = chunker.chunk_article(article)
    section_chunks, mapping_complete = chunks_by_source_section(article, chunker, article_chunks)
    print(f"Section/chunk mapping complete: {mapping_complete}")

    matches = [
        (section, chunks)
        for section, chunks in section_chunks
        if normalized_title(section.title) == normalized_title(requested_title)
    ]
    if not matches:
        print(f"No extracted section matched title {requested_title!r}.")
        raw_evidence = raw_xml_title_and_length(xml_path, "body", requested_title)
        for raw_title, raw_length, raw_excerpt in raw_evidence:
            print(f"Raw XML section title={raw_title!r}, itertext length={raw_length}, excerpt={raw_excerpt!r}")
        return None

    anomalous_matches = [
        (section, chunks)
        for section, chunks in matches
        if any(len(chunk.text) <= 1 for chunk in chunks)
    ]
    if anomalous_matches:
        matches = anomalous_matches
    elif len(matches) > 1:
        print(
            f"Found {len(matches)} sections with this title; none produced a one-character chunk. "
            "Reporting the section with the shortest corresponding chunk."
        )
        matches = [
            min(matches, key=lambda item: min((len(chunk.text) for chunk in item[1]), default=0))
        ]

    classifications = []
    for match_index, (section, chunks) in enumerate(matches, start=1):
        print(f"Matched section {match_index}")
        print(f"section title: {section.title!r}")
        print(f"normalized title: {section.normalized_title!r}")
        print(f"category: {section.category!r}")
        print(f"source: {section.source!r}")
        print(f"section path: {section.section_path!r}")
        print(f"original extracted text length: {len(section.text)}")
        print(f"repr(section.text): {section.text!r}")
        print(f"Corresponding chunks in this section: {len(chunks)}")
        chunk = min(chunks, key=lambda item: (len(item.text), item.chunk_index)) if chunks else None
        if chunk is not None:
            print(
                f"Corresponding shortest chunk: chunk_index={chunk.chunk_index}; "
                f"length={len(chunk.text)}; repr(chunk.text)={chunk.text!r}"
            )
            if chunk.chunk_index > 0:
                previous = next(
                    (item for item in chunks if item.chunk_index == chunk.chunk_index - 1),
                    None,
                )
                if previous is not None:
                    print(f"Previous chunk final 30 characters: {previous.text[-30:]!r}")
                    print(
                        "Previous chunk ends with this chunk: "
                        f"{previous.text.endswith(chunk.text)}"
                    )
        assessment = section_assessment(section, chunks)
        print(f"Assessment: {assessment}")
        classifications.append(assessment)

    return " ".join(classifications)


def main() -> int:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    try:
        from symptom_rag_analyzer.data.literature_chunker import LiteratureSectionChunker
        from symptom_rag_analyzer.data.literature_parser import PMCLiteratureParser
    except ImportError as error:
        print(f"ERROR: Could not import existing parser/chunker: {error}", file=sys.stderr)
        return 1

    parser = PMCLiteratureParser()
    chunker = LiteratureSectionChunker(target_size=TARGET_SIZE, overlap=OVERLAP)
    try:
        zero_status = print_case1(parser)
        short_assessments = []
        for pmid, pmcid, title in SHORT_CASES:
            short_assessments.append(
                print_short_case(parser, chunker, pmid, pmcid, title)
            )
    except (ET.ParseError, OSError, ValueError) as error:
        print(f"ERROR during targeted anomaly investigation: {type(error).__name__}: {error}", file=sys.stderr)
        print("\nZERO_ARTICLE_STATUS:\nInvestigation failed before completion.")
        print("\nSHORT_CHUNK_STATUS:\nInvestigation failed before completion.")
        print("\nRECOMMENDATION:\nINVESTIGATE_ONLY")
        return 1

    joined_assessments = " ".join(value or "" for value in short_assessments)
    if "duplicates the previous window" in joined_assessments:
        recommendation = "CHUNKER_FIX_NEEDED"
        short_status = "At least one short chunk is a duplicated trailing window fragment."
    elif "source XML evidence" in joined_assessments:
        recommendation = "INVESTIGATE_ONLY"
        short_status = "Short extracted chunk(s) require source XML review; no implementation issue was established."
    elif any(value is None for value in short_assessments):
        recommendation = "INVESTIGATE_ONLY"
        short_status = "One or more requested sections were not found in parser output."
    else:
        recommendation = "INVESTIGATE_ONLY"
        short_status = "No chunker or parser defect was established by the targeted evidence."

    print(f"\nZERO_ARTICLE_STATUS:\n{zero_status}")
    print(f"\nSHORT_CHUNK_STATUS:\n{short_status}")
    print(f"\nRECOMMENDATION:\n{recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
