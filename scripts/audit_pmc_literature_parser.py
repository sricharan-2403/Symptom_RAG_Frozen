import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_fulltext_manifest.csv"
PMC_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "pmc"
REQUIRED_COLUMNS = {"pmid", "pmcid"}


def load_manifest_pmids_by_pmcid() -> Dict[str, Set[str]]:
    if not MANIFEST_FILE.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}")

    pmids_by_pmcid = defaultdict(set)
    with MANIFEST_FILE.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Manifest is missing required columns: " + ", ".join(sorted(missing))
            )
        for row in reader:
            pmcid = (row.get("pmcid") or "").strip()
            pmid = (row.get("pmid") or "").strip()
            if pmcid:
                if pmid:
                    pmids_by_pmcid[pmcid].add(pmid)
                else:
                    pmids_by_pmcid.setdefault(pmcid, set())
    return pmids_by_pmcid


def discover_xml_files(manifest_pmcids: Set[str]) -> List[Path]:
    if not PMC_DIRECTORY.is_dir():
        raise FileNotFoundError(f"PMC XML directory not found: {PMC_DIRECTORY}")
    return sorted(
        (
            path for path in PMC_DIRECTORY.glob("*.xml")
            if path.stem in manifest_pmcids and path.is_file()
        ),
        key=lambda path: path.stem,
    )


def section_metrics(article) -> dict:
    sections = article.all_sections()
    path_counts = Counter(tuple(section.section_path) for section in sections)
    duplicate_path_count = sum(count - 1 for count in path_counts.values() if count > 1)
    return {
        "abstract_count": len(article.abstract_sections),
        "body_count": len(article.body_sections),
        "total_count": len(sections),
        "max_level": max((section.level for section in sections), default=0),
        "max_path_depth": max((len(section.section_path) for section in sections), default=0),
        "empty_count": sum(not section.text.strip() for section in sections),
        "duplicate_path_count": duplicate_path_count,
        "text_characters": sum(len(section.text) for section in sections),
        "sections": sections,
    }


def print_article_record(pmcid: str, article, metrics: dict) -> None:
    print(
        "PARSE=SUCCESS"
        f" | PMID={article.pmid or '(missing)'}"
        f" | PMCID={article.pmcid or pmcid}"
        f" | article_type={article.article_type or '(missing)'}"
        f" | publication_year={article.publication_year}"
        f" | has_abstract={article.has_abstract()}"
        f" | has_body={article.has_body()}"
        f" | abstract_sections={metrics['abstract_count']}"
        f" | body_sections={metrics['body_count']}"
        f" | total_sections={metrics['total_count']}"
        f" | max_level={metrics['max_level']}"
        f" | max_path_depth={metrics['max_path_depth']}"
        f" | empty_sections={metrics['empty_count']}"
        f" | duplicate_section_paths={metrics['duplicate_path_count']}"
        f" | extracted_text_characters={metrics['text_characters']}"
    )


def main() -> int:
    try:
        pmids_by_pmcid = load_manifest_pmids_by_pmcid()
        xml_files = discover_xml_files(set(pmids_by_pmcid))
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    try:
        from symptom_rag_analyzer.data.literature_parser import PMCLiteratureParser
    except ImportError as error:
        print(f"ERROR: Could not import PMCLiteratureParser: {error}", file=sys.stderr)
        return 1

    parser = PMCLiteratureParser()
    successful_records = []
    failures = []
    section_categories = Counter()
    normalized_titles = Counter()
    total_sections = 0
    total_empty_sections = 0
    total_duplicate_paths = 0
    articles_with_abstract = 0
    articles_with_body = 0
    abstract_only_articles = 0
    articles_with_neither = 0
    maximum_section_depth = 0

    print(f"Manifest: {MANIFEST_FILE.relative_to(PROJECT_ROOT)}")
    print(f"XML directory: {PMC_DIRECTORY.relative_to(PROJECT_ROOT)}")
    print("Per-article results:")
    for xml_path in xml_files:
        pmcid = xml_path.stem
        try:
            article = parser.parse_file(xml_path)
            metrics = section_metrics(article)
        except Exception as error:
            manifest_pmids = "|".join(sorted(pmids_by_pmcid.get(pmcid, set())))
            failures.append({
                "pmcid": pmcid,
                "pmid": manifest_pmids,
                "filename": xml_path.name,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            })
            print(
                f"PARSE=FAILURE | PMID={manifest_pmids or '(not available)'}"
                f" | PMCID={pmcid} | filename={xml_path.name}"
                f" | exception_type={type(error).__name__}"
                f" | exception_message={error}"
            )
            continue

        successful_records.append({
            "pmcid": pmcid,
            "pmid": article.pmid,
            "filename": xml_path.name,
            "text_characters": metrics["text_characters"],
            "article": article,
            "metrics": metrics,
        })
        print_article_record(pmcid, article, metrics)

        articles_with_abstract += article.has_abstract()
        articles_with_body += article.has_body()
        abstract_only_articles += article.has_abstract() and not article.has_body()
        articles_with_neither += not article.has_abstract() and not article.has_body()
        total_sections += metrics["total_count"]
        total_empty_sections += metrics["empty_count"]
        total_duplicate_paths += metrics["duplicate_path_count"]
        maximum_section_depth = max(maximum_section_depth, metrics["max_path_depth"])
        for section in metrics["sections"]:
            normalized_titles[section.normalized_title or "(empty)"] += 1
            section_categories[section.category or "other"] += 1

    successful_count = len(successful_records)
    failure_count = len(failures)
    average_sections = total_sections / successful_count if successful_count else 0.0
    parsed_text_records = sorted(
        successful_records,
        key=lambda row: (row["text_characters"], row["pmcid"]),
    )

    print("\nCORPUS SUMMARY")
    print(f"Total manifest-backed XMLs discovered: {len(xml_files)}")
    print(f"Successfully parsed: {successful_count}")
    print(f"Parse failures: {failure_count}")
    print(f"Articles with abstract: {articles_with_abstract}")
    print(f"Articles with body: {articles_with_body}")
    print(f"Abstract-only articles: {abstract_only_articles}")
    print(f"Articles with neither abstract nor body: {articles_with_neither}")
    print(f"Total sections: {total_sections}")
    print(f"Average sections/article: {average_sections:.2f}")
    print(f"Maximum section depth: {maximum_section_depth}")
    print(f"Empty-section total: {total_empty_sections}")
    print(f"Duplicate-path total: {total_duplicate_paths}")

    print("\n30 MOST COMMON NORMALIZED SECTION TITLES")
    for title, count in normalized_titles.most_common(30):
        print(f"{count}\t{title}")

    print("\nCATEGORY COUNTS")
    for category, count in sorted(section_categories.items()):
        print(f"{category}\t{count}")

    print("\n20 ARTICLES WITH THE SMALLEST EXTRACTED TEXT LENGTH")
    for record in parsed_text_records[:20]:
        print(
            f"{record['text_characters']}\t{record['pmid'] or '(missing)'}"
            f"\t{record['pmcid']}\t{record['filename']}"
        )

    print("\n20 ARTICLES WITH THE LARGEST EXTRACTED TEXT LENGTH")
    for record in parsed_text_records[-20:][::-1]:
        print(
            f"{record['text_characters']}\t{record['pmid'] or '(missing)'}"
            f"\t{record['pmcid']}\t{record['filename']}"
        )

    print("\nPARSE FAILURES")
    if failures:
        for failure in failures:
            print(
                f"PMCID={failure['pmcid']} | PMID={failure['pmid'] or '(not available)'}"
                f" | filename={failure['filename']}"
                f" | exception_type={failure['exception_type']}"
                f" | exception_message={failure['exception_message']}"
            )
    else:
        print("None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
