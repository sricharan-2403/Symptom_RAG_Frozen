import csv
import os
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTICLE_FILE = DATA_DIR / "pubmed_article_metadata.csv"
RECOVERED_FILE = DATA_DIR / "pubmed_metadata_recovered.csv"
OUTPUT_FILE = DATA_DIR / "pubmed_complete_metadata.csv"

CANONICAL_COLUMNS = (
    "pmid",
    "pmcid",
    "doi",
    "title",
    "abstract",
    "journal",
    "publication_date",
    "publication_year",
    "publication_types",
    "language",
    "authors",
    "mesh_terms",
    "book_title",
    "publisher",
    "record_type",
    "disease_ids",
    "disease_names",
    "query_categories",
    "queries",
    "best_rank",
    "match_counts",
    "has_pmc",
)
EXPECTED_TOTAL = 35999
EXPECTED_RECORD_TYPES = {
    "pubmed_article": 34829,
    "pubmed_book_article": 1170,
}


def load_records(path, default_record_type=None):
    records = []
    seen_pmids = set()
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if "pmid" not in (reader.fieldnames or []):
            raise ValueError(f"{path.name} is missing the required 'pmid' column.")

        for row_number, source_row in enumerate(reader, start=2):
            pmid = (source_row.get("pmid") or "").strip()
            if not pmid:
                raise ValueError(f"{path.name} has an empty PMID at row {row_number}.")
            if pmid in seen_pmids:
                raise ValueError(f"Duplicate PMID {pmid} in {path.name}.")
            seen_pmids.add(pmid)

            record = {
                column: source_row.get(column) or ""
                for column in CANONICAL_COLUMNS
            }
            record["pmid"] = pmid
            if not record["record_type"] and default_record_type:
                record["record_type"] = default_record_type
            records.append(record)
    return records


def validate_records(records):
    pmids = [record["pmid"] for record in records]
    duplicates = len(pmids) - len(set(pmids))
    record_type_counts = Counter(record["record_type"] for record in records)

    errors = []
    if duplicates:
        errors.append(f"found {duplicates} duplicate PMID(s)")
    if len(records) != EXPECTED_TOTAL:
        errors.append(f"expected {EXPECTED_TOTAL} rows, found {len(records)}")
    if record_type_counts != Counter(EXPECTED_RECORD_TYPES):
        errors.append(
            "unexpected record_type distribution: "
            + ", ".join(
                f"{record_type}={record_type_counts.get(record_type, 0)}"
                for record_type in EXPECTED_RECORD_TYPES
            )
        )
    return record_type_counts, errors


def write_output(records):
    with OUTPUT_FILE.open("x", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
        output_file.flush()
        os.fsync(output_file.fileno())


def main():
    if OUTPUT_FILE.exists():
        print(f"ERROR: Refusing to overwrite existing output: {OUTPUT_FILE}", file=sys.stderr)
        return 1

    try:
        article_records = load_records(ARTICLE_FILE, default_record_type="pubmed_article")
        recovered_records = load_records(RECOVERED_FILE)
        records = article_records + recovered_records
        record_type_counts, validation_errors = validate_records(records)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"ERROR: Merge validation failed: {error}", file=sys.stderr)
        return 1

    print("PUBMED METADATA MERGE")
    print(f"Existing article records: {len(article_records)}")
    print(f"Recovered records: {len(recovered_records)}")
    print(f"Combined rows: {len(records)}")
    print(f"Unique PMIDs: {len({record['pmid'] for record in records})}")
    print(f"Duplicate PMIDs: {len(records) - len({record['pmid'] for record in records})}")
    print("record_type distribution:")
    for record_type in EXPECTED_RECORD_TYPES:
        print(f"  {record_type}: {record_type_counts.get(record_type, 0)}")

    if validation_errors:
        print("Validation: FAIL")
        for error in validation_errors:
            print(f"  - {error}")
        return 1

    try:
        write_output(records)
    except OSError as error:
        print(f"ERROR: Could not create output without overwriting: {error}", file=sys.stderr)
        return 1

    print("Validation: PASS")
    print(f"Output: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())