import csv
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FAILURE_FILE = DATA_DIR / "pubmed_metadata_failures.csv"
RECOVERED_FILE = DATA_DIR / "pubmed_metadata_recovered.csv"
CANDIDATE_FILE = DATA_DIR / "pubmed_candidate_articles.csv"
RECOVERY_FAILURE_FILE = DATA_DIR / "pubmed_metadata_recovery_failures.csv"

EXPECTED_TARGET_COUNT = 1170
PROVENANCE_FIELDS = (
    "disease_ids",
    "disease_names",
    "query_categories",
    "queries",
    "best_rank",
    "match_counts",
)
METADATA_FIELDS = (
    "title",
    "abstract",
    "publication_date",
    "authors",
)
RECOVERED_REQUIRED_COLUMNS = ("pmid", "record_type") + METADATA_FIELDS + PROVENANCE_FIELDS
CANDIDATE_REQUIRED_COLUMNS = ("pmid",) + PROVENANCE_FIELDS


def read_csv(path, required_columns):
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = set(required_columns) - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"{path.name} is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
        return list(reader)


def read_target_pmids():
    failure_rows = read_csv(FAILURE_FILE, ("pmids",))
    pmid_entries = []
    for row in failure_rows:
        pmid_entries.extend(
            value.strip()
            for value in (row.get("pmids") or "").split(",")
            if value.strip()
        )
    return pmid_entries, set(pmid_entries)


def count_missing_values(rows, fields):
    return {
        field: sum(not (row.get(field) or "").strip() for row in rows)
        for field in fields
    }


def read_candidate_provenance():
    candidate_rows = read_csv(CANDIDATE_FILE, CANDIDATE_REQUIRED_COLUMNS)
    candidates = {}
    duplicate_candidates = 0
    for row in candidate_rows:
        pmid = (row.get("pmid") or "").strip()
        if not pmid:
            continue
        if pmid in candidates:
            duplicate_candidates += 1
            continue
        candidates[pmid] = row
    return candidates, duplicate_candidates


def read_recovery_failure_count():
    if not RECOVERY_FAILURE_FILE.exists():
        return None
    with RECOVERY_FAILURE_FILE.open("r", encoding="utf-8-sig", newline="") as failure_file:
        return sum(1 for _ in csv.DictReader(failure_file))


def main():
    try:
        target_entries, target_pmids = read_target_pmids()
        recovered_rows = read_csv(RECOVERED_FILE, RECOVERED_REQUIRED_COLUMNS)
        candidates, duplicate_candidate_rows = read_candidate_provenance()
        recovery_failure_count = read_recovery_failure_count()
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"ERROR: Validation could not complete: {error}", file=sys.stderr)
        return 1

    recovered_pmids = [(row.get("pmid") or "").strip() for row in recovered_rows]
    nonempty_recovered_pmids = [pmid for pmid in recovered_pmids if pmid]
    unique_recovered_pmids = set(nonempty_recovered_pmids)
    missing_target_pmids = target_pmids - unique_recovered_pmids
    unexpected_pmids = unique_recovered_pmids - target_pmids
    duplicate_pmid_count = len(nonempty_recovered_pmids) - len(unique_recovered_pmids)
    empty_recovered_pmid_count = len(recovered_rows) - len(nonempty_recovered_pmids)

    record_type_counts = Counter(
        (row.get("record_type") or "").strip() or "(empty)"
        for row in recovered_rows
    )
    missing_metadata = count_missing_values(recovered_rows, METADATA_FIELDS)
    missing_provenance = count_missing_values(recovered_rows, PROVENANCE_FIELDS)

    missing_candidate_rows = 0
    provenance_mismatches = Counter()
    for row in recovered_rows:
        pmid = (row.get("pmid") or "").strip()
        candidate = candidates.get(pmid)
        if candidate is None:
            missing_candidate_rows += 1
            continue
        for field in PROVENANCE_FIELDS:
            if (row.get(field) or "").strip() != (candidate.get(field) or "").strip():
                provenance_mismatches[field] += 1

    print("=" * 72)
    print("PUBMED RECOVERY VALIDATION")
    print("=" * 72)
    print(f"Target PMIDs: {len(target_pmids)}")
    print(f"Original failure PMID entries: {len(target_entries)}")
    print(f"Recovered rows: {len(recovered_rows)}")
    print(f"Unique recovered PMIDs: {len(unique_recovered_pmids)}")
    print(f"Missing target PMIDs: {len(missing_target_pmids)}")
    print(f"Unexpected PMIDs: {len(unexpected_pmids)}")
    print(f"Duplicate PMIDs: {duplicate_pmid_count}")
    print(f"Empty recovered PMID rows: {empty_recovered_pmid_count}")

    print("\nrecord_type distribution:")
    for record_type in sorted(record_type_counts):
        print(f"  {record_type}: {record_type_counts[record_type]}")

    print("\nMissing metadata values:")
    for field in METADATA_FIELDS:
        print(f"  {field}: {missing_metadata[field]}")

    print("\nMissing provenance values:")
    for field in PROVENANCE_FIELDS:
        print(f"  {field}: {missing_provenance[field]}")

    print("\nProvenance cross-check:")
    print(f"  Recovered PMIDs without candidate row: {missing_candidate_rows}")
    print(f"  Duplicate candidate PMID rows: {duplicate_candidate_rows}")
    print(f"  Provenance mismatches: {sum(provenance_mismatches.values())}")
    for field in PROVENANCE_FIELDS:
        if provenance_mismatches[field]:
            print(f"    {field}: {provenance_mismatches[field]}")

    if recovery_failure_count is None:
        print("\nRecovery failure file rows: file not found")
    else:
        print(f"\nRecovery failure file rows: {recovery_failure_count}")

    validation_errors = []
    if len(target_pmids) != EXPECTED_TARGET_COUNT:
        validation_errors.append(f"expected {EXPECTED_TARGET_COUNT} target PMIDs")
    if missing_target_pmids:
        validation_errors.append("one or more target PMIDs are missing")
    if unexpected_pmids:
        validation_errors.append("unexpected recovered PMIDs were found")
    if duplicate_pmid_count or empty_recovered_pmid_count:
        validation_errors.append("recovered PMID rows are not unique and complete")
    if record_type_counts != Counter({"pubmed_book_article": len(recovered_rows)}):
        validation_errors.append("record_type distribution differs from the expected book-article set")
    if missing_candidate_rows or duplicate_candidate_rows or sum(provenance_mismatches.values()):
        validation_errors.append("candidate provenance validation failed")
    if recovery_failure_count not in (None, 0):
        validation_errors.append("recovery failure file is not empty")

    if validation_errors:
        print("\nValidation result: FAIL")
        for error in validation_errors:
            print(f"  - {error}")
        return 1

    print("\nValidation result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())