import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from symptom_rag_analyzer.data.literature_models import LiteratureArticle
from symptom_rag_analyzer.data.literature_parser import PMCLiteratureParser


MANIFEST_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_fulltext_manifest.csv"
PMC_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "pmc"
REQUIRED_COLUMNS = {"pmid", "pmcid", "publication_types"}


def load_local_manifest_records() -> List[Tuple[Dict[str, str], Path]]:
    with MANIFEST_FILE.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Manifest is missing required columns: " + ", ".join(sorted(missing))
            )

        records = []
        seen_pmcids = set()
        for row in reader:
            pmcid = (row.get("pmcid") or "").strip()
            if not pmcid or pmcid in seen_pmcids:
                continue
            xml_path = PMC_DIRECTORY / f"{pmcid}.xml"
            if not xml_path.is_file():
                continue
            seen_pmcids.add(pmcid)
            records.append((row, xml_path))
        return records


def publication_types(row: Dict[str, str]) -> set[str]:
    return {
        value.strip().casefold()
        for value in (row.get("publication_types") or "").split("|")
        if value.strip()
    }


def choose_example(
    candidates: List[Tuple[Dict[str, str], Path]],
    parser: PMCLiteratureParser,
    used_pmcids: set[str],
    matches: Callable[[Dict[str, str], LiteratureArticle], bool],
) -> Optional[Tuple[Dict[str, str], Path, LiteratureArticle]]:
    for row, xml_path in candidates:
        pmcid = (row.get("pmcid") or "").strip()
        if pmcid in used_pmcids:
            continue
        try:
            article = parser.parse_file(xml_path)
        except (OSError, ValueError):
            continue
        if matches(row, article):
            used_pmcids.add(pmcid)
            return row, xml_path, article
    return None


def section_excerpt(text: str) -> str:
    text = text or ""
    return text if len(text) <= 200 else text[:200] + "..."


def print_section(index: int, section, label: str) -> None:
    print(f"{label} {index}")
    print(f"  level: {section.level}")
    print(f"  title: {section.title}")
    print(f"  normalized_title: {section.normalized_title}")
    print(f"  section_path: {section.section_path}")
    print(f"  category: {section.category}")
    print(f"  text length: {len(section.text)}")
    print(f"  text: {section_excerpt(section.text)}")


def print_article(label: str, selection) -> None:
    if selection is None:
        print(f"\n{label}: NOT AVAILABLE among local manifest-backed XML files")
        return

    row, xml_path, article = selection
    print(f"\n{'=' * 76}\n{label}\n{'=' * 76}")
    print(f"XML filename: {xml_path.name}")
    print(f"PMID: {article.pmid or row.get('pmid', '')}")
    print(f"PMCID: {article.pmcid or row.get('pmcid', '')}")
    print(f"Title: {article.title}")
    print(f"article_type: {article.article_type}")
    print(f"record_type: {article.record_type}")
    print(f"Publication year: {article.publication_year}")
    print(f"Number of abstract sections: {len(article.abstract_sections)}")
    print(f"Number of body sections: {len(article.body_sections)}")
    print(f"has_abstract(): {article.has_abstract()}")
    print(f"has_body(): {article.has_body()}")

    print("Abstract sections:")
    if article.abstract_sections:
        for index, section in enumerate(article.abstract_sections, start=1):
            print_section(index, section, "Abstract")
    else:
        print("  (none)")

    print("First 10 body sections:")
    if article.body_sections:
        for index, section in enumerate(article.body_sections[:10], start=1):
            print_section(index, section, "Body")
    else:
        print("  (none)")

    sections = article.all_sections()
    path_counts = Counter(tuple(section.section_path) for section in sections)
    print(f"Maximum section level: {max((section.level for section in sections), default=0)}")
    print(
        "Maximum section_path depth: "
        f"{max((len(section.section_path) for section in sections), default=0)}"
    )
    print(f"Any section with empty text: {any(not section.text.strip() for section in sections)}")
    print(f"Any duplicated section_path values: {any(count > 1 for count in path_counts.values())}")


def main() -> int:
    try:
        candidates = load_local_manifest_records()
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"ERROR: Could not read manifest: {error}", file=sys.stderr)
        return 1

    parser = PMCLiteratureParser()
    used_pmcids: set[str] = set()

    research = choose_example(
        candidates,
        parser,
        used_pmcids,
        lambda row, article: article.article_type.casefold() in {"research-article", "research"},
    )
    case_report = choose_example(
        candidates,
        parser,
        used_pmcids,
        lambda row, article: article.article_type.casefold() in {"case-report", "case-report-article"},
    )
    review = choose_example(
        candidates,
        parser,
        used_pmcids,
        lambda row, article: "review" in article.article_type.casefold()
        or bool(
            {"review", "systematic review", "meta-analysis"}
            & publication_types(row)
        ),
    )
    abstract_only = choose_example(
        candidates,
        parser,
        used_pmcids,
        lambda row, article: article.has_abstract() and not article.has_body(),
    )

    print(f"Manifest: {MANIFEST_FILE.relative_to(PROJECT_ROOT)}")
    print(f"Local manifest-backed XML candidates: {len(candidates)}")
    print_article("RESEARCH ARTICLE", research)
    print_article("CASE REPORT", case_report)
    print_article("REVIEW ARTICLE", review)
    print_article("ABSTRACT WITH NO PARSED BODY", abstract_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
