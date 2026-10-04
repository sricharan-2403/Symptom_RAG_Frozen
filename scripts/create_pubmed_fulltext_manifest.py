import csv
import os
import re
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_FILE = DATA_DIR / "pubmed_selected_literature.csv"
OUTPUT_FILE = DATA_DIR / "pubmed_fulltext_manifest.csv"

METADATA_COLUMNS = (
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
NEW_COLUMNS = (
    "acquisition_route",
    "acquisition_status",
    "source_url",
    "local_path",
    "license",
    "retrieved_at",
    "error_type",
    "error_message",
)
OUTPUT_COLUMNS = METADATA_COLUMNS + NEW_COLUMNS
ROUTES = {"pmc", "doi_candidate", "metadata_only"}


def read_selected_records():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Selected literature input does not exist: {INPUT_FILE}")
    with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        missing_columns = set(METADATA_COLUMNS) - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Selected literature is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
        return list(reader)


def pmid_sort_key(pmid):
    try:
        return 0, int(pmid)
    except ValueError:
        return 1, pmid


def publication_year_sort_value(value):
    match = re.search(r"\b(\d{4})\b", value or "")
    return int(match.group(1)) if match else None


def route_record(source_row):
    record = {column: source_row.get(column) or "" for column in METADATA_COLUMNS}
    record["pmid"] = (source_row.get("pmid") or "").strip()
    pmcid = (source_row.get("pmcid") or "").strip()
    doi = (source_row.get("doi") or "").strip()

    if pmcid:
        route = "pmc"
        source_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    elif doi:
        route = "doi_candidate"
        source_url = f"https://doi.org/{doi}"
    else:
        route = "metadata_only"
        source_url = ""

    record.update({
        "acquisition_route": route,
        "acquisition_status": "pending",
        "source_url": source_url,
        "local_path": "",
        "license": "",
        "retrieved_at": "",
        "error_type": "",
        "error_message": "",
    })
    return record


def sort_key(record):
    year = publication_year_sort_value(record.get("publication_year", ""))
    return (
        record["acquisition_route"],
        -(year if year is not None else -1),
        pmid_sort_key(record["pmid"]),
    )


def validate_records(source_rows, manifest_rows):
    source_pmids = [(row.get("pmid") or "").strip() for row in source_rows]
    nonempty_source_pmids = [pmid for pmid in source_pmids if pmid]
    source_counts = Counter(nonempty_source_pmids)
    duplicate_count = sum(count - 1 for count in source_counts.values() if count > 1)
    unique_count = len(source_counts)

    manifest_pmids = [row["pmid"] for row in manifest_rows]
    manifest_counts = Counter(manifest_pmids)
    route_counts = Counter(row["acquisition_route"] for row in manifest_rows)

    route_consistent = True
    for row in manifest_rows:
        has_pmcid = bool((row.get("pmcid") or "").strip())
        has_doi = bool((row.get("doi") or "").strip())
        route = row.get("acquisition_route")
        if route not in ROUTES:
            route_consistent = False
        elif route == "pmc" and not has_pmcid:
            route_consistent = False
        elif route == "doi_candidate" and (has_pmcid or not has_doi):
            route_consistent = False
        elif route == "metadata_only" and (has_pmcid or has_doi):
            route_consistent = False

    every_source_once = (
        len(manifest_rows) == len(source_rows)
        and manifest_counts == Counter(nonempty_source_pmids)
    )
    deterministic_order = manifest_rows == sorted(manifest_rows, key=sort_key)
    no_missing_pmid = not any(not pmid for pmid in source_pmids) and all(manifest_pmids)

    checks = {
        "unique_pmids": duplicate_count == 0 and unique_count == len(source_rows),
        "route_consistency": route_consistent,
        "no_missing_pmid": no_missing_pmid and every_source_once,
        "deterministic_order": deterministic_order,
    }
    return {
        "source_total": len(source_rows),
        "unique_count": unique_count,
        "duplicate_count": duplicate_count,
        "empty_source_pmids": sum(not pmid for pmid in source_pmids),
        "route_counts": route_counts,
        "checks": checks,
    }


def print_summary(stats):
    print("PUBMED FULL-TEXT ACQUISITION MANIFEST")
    print("======================================")
    print(f"\nTotal selected PMIDs: {stats['source_total']}")
    print(f"PMC route: {stats['route_counts']['pmc']}")
    print(f"DOI candidate route: {stats['route_counts']['doi_candidate']}")
    print(f"Metadata-only: {stats['route_counts']['metadata_only']}")
    print(f"Duplicate PMIDs: {stats['duplicate_count']}")
    print(f"Empty PMIDs: {stats['empty_source_pmids']}")
    print("\nValidation:")
    print(f"Unique PMIDs: {'PASS' if stats['checks']['unique_pmids'] else 'FAIL'}")
    print(f"Route consistency: {'PASS' if stats['checks']['route_consistency'] else 'FAIL'}")
    print(f"No missing PMID: {'PASS' if stats['checks']['no_missing_pmid'] else 'FAIL'}")
    print(f"Deterministic ordering: {'PASS' if stats['checks']['deterministic_order'] else 'FAIL'}")
    print(f"\nOutput: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")


def write_manifest(records):
    with OUTPUT_FILE.open("x", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(records)
        output_file.flush()
        os.fsync(output_file.fileno())


def main():
    if OUTPUT_FILE.exists():
        print(f"ERROR: Refusing to overwrite existing output: {OUTPUT_FILE}", file=sys.stderr)
        return 1

    try:
        source_rows = read_selected_records()
        records = [route_record(row) for row in source_rows]
        records.sort(key=sort_key)
        stats = validate_records(source_rows, records)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"ERROR: Could not validate acquisition manifest: {error}", file=sys.stderr)
        return 1

    print_summary(stats)
    if not all(stats["checks"].values()):
        print("ERROR: Manifest validation failed; output was not written.", file=sys.stderr)
        return 1

    try:
        write_manifest(records)
    except OSError as error:
        print(f"ERROR: Could not write manifest without overwriting: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())