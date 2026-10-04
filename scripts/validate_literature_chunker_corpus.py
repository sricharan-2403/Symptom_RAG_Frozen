import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_fulltext_manifest.csv"
PMC_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "pmc"
TARGET_SIZE = 1000
OVERLAP = 150
REQUIRED_COLUMNS = {"pmid", "pmcid", "acquisition_route"}


def load_manifest_targets() -> Tuple[int, Dict[str, Set[str]]]:
    if not MANIFEST_FILE.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}")

    pmids_by_pmcid = defaultdict(set)
    target_rows = 0
    with MANIFEST_FILE.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Manifest is missing required columns: " + ", ".join(sorted(missing))
            )
        for row in reader:
            if (row.get("acquisition_route") or "").strip().casefold() != "pmc":
                continue
            pmcid = (row.get("pmcid") or "").strip()
            if not pmcid:
                continue
            target_rows += 1
            pmid = (row.get("pmid") or "").strip()
            if pmid:
                pmids_by_pmcid[pmcid].add(pmid)
            else:
                pmids_by_pmcid.setdefault(pmcid, set())
    return target_rows, pmids_by_pmcid


def discover_local_xml(pmcids: Set[str]) -> List[Path]:
    if not PMC_DIRECTORY.is_dir():
        raise FileNotFoundError(f"PMC XML directory not found: {PMC_DIRECTORY}")
    return sorted(
        (PMC_DIRECTORY / f"{pmcid}.xml" for pmcid in pmcids
         if (PMC_DIRECTORY / f"{pmcid}.xml").is_file()),
        key=lambda path: path.stem,
    )


def article_section_chunks(article, chunker):
    """Chunk each existing section independently using the production chunker."""
    expected = []
    section_chunks = []
    sections = list(article.abstract_sections) + list(article.body_sections)
    abstract_count = len(article.abstract_sections)
    for section_index, section in enumerate(sections):
        if section_index < abstract_count:
            section_article = replace(article, abstract_sections=[section], body_sections=[])
        else:
            section_article = replace(article, abstract_sections=[], body_sections=[section])
        chunks = chunker.chunk_article(section_article)
        section_chunks.append((section_index, section, chunks))
        expected.extend(chunks)
    return sections, section_chunks, expected


def check_article_integrity(article, chunks, chunker) -> Dict[str, bool]:
    sections, section_chunks, expected_chunks = article_section_chunks(article, chunker)
    repeated_chunks = chunker.chunk_article(article)

    unique_keys = set()
    no_duplicate_records = True
    for section_index, _section, section_chunk_list in section_chunks:
        for chunk in section_chunk_list:
            key = (section_index, chunk.chunk_index)
            if key in unique_keys:
                no_duplicate_records = False
            unique_keys.add(key)

    indexes_valid = all(
        [chunk.chunk_index for chunk in section_chunk_list]
        == list(range(len(section_chunk_list)))
        for _section_index, _section, section_chunk_list in section_chunks
    )
    sources_in_order = [chunk.source for chunk in chunks] == [
        chunk.source for chunk in expected_chunks
    ]
    paths_preserved = all(
        chunk.section_path == section.section_path
        for _section_index, section, section_chunk_list in section_chunks
        for chunk in section_chunk_list
    )
    section_ownership = chunks == expected_chunks
    within_limit = all(len(chunk.text) <= chunker.target_size for chunk in chunks)

    return {
        "no_empty_text": all(bool(chunk.text.strip()) for chunk in chunks),
        "every_chunk_has_pmid": all(bool(chunk.pmid.strip()) for chunk in chunks),
        "one_source_section_per_chunk": section_ownership,
        "section_path_preserved": paths_preserved,
        "indexes_start_at_zero_and_are_contiguous": indexes_valid,
        "abstracts_precede_body": sources_in_order,
        "size_limit_or_oversized_fallback": within_limit,
        "deterministic": chunks == repeated_chunks,
        "no_duplicate_section_chunk_indexes": no_duplicate_records,
    }


def make_article_record(xml_path: Path, article, chunks, sections, checks) -> dict:
    lengths = [len(chunk.text) for chunk in chunks]
    return {
        "pmcid": article.pmcid or xml_path.stem,
        "pmid": article.pmid,
        "title": article.title,
        "article_type": article.article_type,
        "section_count": len(sections),
        "nonempty_section_count": sum(bool(section.text.strip()) for section in sections),
        "chunks": chunks,
        "chunk_count": len(chunks),
        "chunk_text_characters": sum(lengths),
        "chunk_lengths": lengths,
        "checks": checks,
    }


def excerpt(text: str, limit: int = 300) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    return normalized if len(normalized) <= limit else normalized[:limit] + "..."


def print_article_example(label: str, record: dict, relevant_length: int) -> None:
    print(
        f"{label}: PMID={record['pmid'] or '(missing)'}"
        f" | PMCID={record['pmcid']}"
        f" | title={record['title']}"
        f" | article_type={record['article_type'] or '(missing)'}"
        f" | sections={record['section_count']}"
        f" | chunks={record['chunk_count']}"
        f" | relevant_chunk_length={relevant_length}"
    )


def print_chunk_example(label: str, record: dict, chunk) -> None:
    print(
        f"{label}: PMID={record['pmid'] or '(missing)'}"
        f" | PMCID={record['pmcid']}"
        f" | title={record['title']}"
        f" | article_type={record['article_type'] or '(missing)'}"
        f" | sections={record['section_count']}"
        f" | chunks={record['chunk_count']}"
        f" | relevant_chunk_length={len(chunk.text)}"
        f" | section_title={chunk.section_title or '(untitled)'}"
        f" | category={chunk.category}"
        f" | source={chunk.source}"
        f" | excerpt={excerpt(chunk.text)}"
    )


def print_statistics(records, target_rows, unique_target_count, xml_files, failures) -> bool:
    parsed_count = len(records)
    chunks = [chunk for record in records for chunk in record["chunks"]]
    chunk_lengths = [len(chunk.text) for chunk in chunks]
    chunks_per_article = [record["chunk_count"] for record in records]
    article_chunks = sum(count > 0 for count in chunks_per_article)
    article_zero_chunks = sum(count == 0 for count in chunks_per_article)
    sections_processed = sum(record["section_count"] for record in records)
    nonempty_sections = sum(record["nonempty_section_count"] for record in records)
    lengths = Counter()
    lengths["<= 200 chars"] = sum(length <= 200 for length in chunk_lengths)
    lengths["201-500 chars"] = sum(201 <= length <= 500 for length in chunk_lengths)
    lengths["501-999 chars"] = sum(501 <= length <= 999 for length in chunk_lengths)
    lengths["exactly 1000 chars"] = sum(length == TARGET_SIZE for length in chunk_lengths)
    lengths["> 1000 chars"] = sum(length > TARGET_SIZE for length in chunk_lengths)

    print("\n" + "=" * 76 + "\nCORPUS-LEVEL STATISTICS\n" + "=" * 76)
    print(f"Manifest PMC targets (rows): {target_rows}")
    print(f"Manifest PMC targets (unique PMCIDs): {unique_target_count}")
    print(f"Local XML files found: {len(xml_files)}")
    print(f"Articles successfully parsed: {parsed_count}")
    print(f"Parse failures: {len(failures)}")
    print(f"Articles producing chunks: {article_chunks}")
    print(f"Articles producing zero chunks: {article_zero_chunks}")
    print(f"Total sections processed: {sections_processed}")
    print(f"Total non-empty sections: {nonempty_sections}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average chunks/article: {mean(chunks_per_article):.2f}" if chunks_per_article else "Average chunks/article: 0.00")
    print(f"Median chunks/article: {median(chunks_per_article):.2f}" if chunks_per_article else "Median chunks/article: 0.00")
    print(f"Minimum chunks/article: {min(chunks_per_article, default=0)}")
    print(f"Maximum chunks/article: {max(chunks_per_article, default=0)}")
    print(f"Average chunk length: {mean(chunk_lengths):.2f}" if chunk_lengths else "Average chunk length: 0.00")
    print(f"Median chunk length: {median(chunk_lengths):.2f}" if chunk_lengths else "Median chunk length: 0.00")
    print(f"Minimum chunk length: {min(chunk_lengths, default=0)}")
    print(f"Maximum chunk length: {max(chunk_lengths, default=0)}")

    print("\nCHUNK-SIZE DIAGNOSTICS")
    for label, count in lengths.items():
        print(f"{label}: {count}")
    total_chunks = len(chunks)
    short_percent = 100 * lengths["<= 200 chars"] / total_chunks if total_chunks else 0.0
    oversized_percent = 100 * lengths["> 1000 chars"] / total_chunks if total_chunks else 0.0
    print(f"Percentage <= 200 chars: {short_percent:.2f}%")
    print(f"Percentage > 1000 chars: {oversized_percent:.2f}%")

    abstract_chunks = sum(chunk.source == "abstract" for chunk in chunks)
    body_chunks = sum(chunk.source == "body" for chunk in chunks)
    category_counts = Counter(chunk.category for chunk in chunks)
    article_type_counts = Counter(
        record["article_type"] or "(missing)"
        for record in records
        for _chunk in record["chunks"]
    )
    source_counts = Counter(chunk.source for chunk in chunks)
    print("\nSECTION DIAGNOSTICS")
    print(f"Abstract chunks: {abstract_chunks}")
    print(f"Body chunks: {body_chunks}")
    print("Chunks by section category:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")
    print("Chunks by article type:")
    for article_type, count in sorted(article_type_counts.items()):
        print(f"  {article_type}: {count}")
    print("Chunks by source:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")

    checks = Counter(check for _record in records for check, passed in _record["checks"].items() if not passed)
    print("\nINTEGRITY CHECKS")
    check_names = (
        "no_empty_text",
        "every_chunk_has_pmid",
        "one_source_section_per_chunk",
        "section_path_preserved",
        "indexes_start_at_zero_and_are_contiguous",
        "abstracts_precede_body",
        "size_limit_or_oversized_fallback",
        "deterministic",
        "no_duplicate_section_chunk_indexes",
    )
    for name in check_names:
        failed = checks[name]
        print(f"{name}: {'FAIL' if failed else 'PASS'}" + (f" ({failed} article(s))" if failed else ""))
    print(f"Parse failures considered corpus validation failures: {len(failures)}")
    print(f"Integrity check failures: {sum(checks.values())}")

    print("\nFIVE ARTICLES WITH SMALLEST TOTAL EXTRACTED CHUNK TEXT")
    for record in sorted(records, key=lambda item: (item["chunk_text_characters"], item["pmcid"]))[:5]:
        print_article_example("Smallest", record, record["chunk_text_characters"])

    print("\nFIVE ARTICLES WITH LARGEST TOTAL EXTRACTED CHUNK TEXT")
    for record in sorted(records, key=lambda item: (item["chunk_text_characters"], item["pmcid"]), reverse=True)[:5]:
        print_article_example("Largest", record, record["chunk_text_characters"])

    print("\nFIVE ARTICLES PRODUCING THE MOST CHUNKS")
    for record in sorted(records, key=lambda item: (item["chunk_count"], item["pmcid"]), reverse=True)[:5]:
        print_article_example("Most chunks", record, record["chunk_text_characters"])

    chunk_examples = [
        (record, chunk)
        for record in records
        for chunk in record["chunks"]
    ]
    print("\nFIVE SHORTEST CHUNKS")
    for record, chunk in sorted(chunk_examples, key=lambda item: (len(item[1].text), item[0]["pmcid"], item[1].section_path))[:5]:
        print_chunk_example("Shortest", record, chunk)

    print("\nFIVE LONGEST CHUNKS")
    for record, chunk in sorted(chunk_examples, key=lambda item: (len(item[1].text), item[0]["pmcid"], item[1].section_path), reverse=True)[:5]:
        print_chunk_example("Longest", record, chunk)

    if failures:
        print("\nPARSE FAILURES")
        for failure in failures:
            print(
                f"PMCID={failure['pmcid']} | PMID={failure['pmid'] or '(not available)'}"
                f" | filename={failure['filename']}"
                f" | {failure['exception_type']}: {failure['exception_message']}"
            )

    return not failures and not sum(checks.values())


def main() -> int:
    try:
        target_rows, pmids_by_pmcid = load_manifest_targets()
        xml_files = discover_local_xml(set(pmids_by_pmcid))
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"ERROR: Could not discover manifest-backed PMC files: {error}", file=sys.stderr)
        print("\nFAIL")
        return 1

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    try:
        from symptom_rag_analyzer.data.literature_chunker import LiteratureSectionChunker
        from symptom_rag_analyzer.data.literature_parser import PMCLiteratureParser
    except ImportError as error:
        print(f"ERROR: Could not import parser/chunker: {error}", file=sys.stderr)
        print("\nFAIL")
        return 1

    parser = PMCLiteratureParser()
    chunker = LiteratureSectionChunker(target_size=TARGET_SIZE, overlap=OVERLAP)
    records = []
    failures = []
    for xml_path in xml_files:
        try:
            article = parser.parse_file(xml_path)
            chunks = chunker.chunk_article(article)
            checks = check_article_integrity(article, chunks, chunker)
            sections = list(article.abstract_sections) + list(article.body_sections)
            records.append(make_article_record(xml_path, article, chunks, sections, checks))
        except Exception as error:
            failures.append({
                "pmcid": xml_path.stem,
                "pmid": "|".join(sorted(pmids_by_pmcid.get(xml_path.stem, set()))),
                "filename": xml_path.name,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            })

    checks_passed = print_statistics(
        records,
        target_rows,
        len(pmids_by_pmcid),
        xml_files,
        failures,
    )
    if checks_passed:
        print("\nPASS")
        return 0
    print("\nFAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
