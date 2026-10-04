import csv
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_complete_metadata.csv"
RECOVERY_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_metadata_recovered.csv"
EXPECTED_TOTAL_ROWS = 35_999
EXPECTED_RECOVERY_ROWS = 1_170
EXPECTED_ARTICLE_ROWS = 34_829
PROVENANCE_COLUMNS = (
    "disease_ids",
    "disease_names",
    "query_categories",
    "queries",
    "best_rank",
    "match_counts",
)


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        print(f"FAIL: {message}")
        raise ValidationError(message)
    print(f"PASS: {message}")


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = reader.fieldnames or []
        if not columns:
            raise ValidationError(f"{path.name} has no header row.")
        if len(columns) != len(set(columns)):
            raise ValidationError(f"{path.name} has duplicate column names.")

        rows = []
        for row_number, raw_row in enumerate(reader, start=2):
            if None in raw_row or any(value is None for value in raw_row.values()):
                raise ValidationError(
                    f"{path.name} has an inconsistent field count at row {row_number}."
                )
            rows.append({column: raw_row[column] for column in columns})
    return columns, rows


def index_unique_pmids(rows, path_label):
    records = {}
    for row_number, row in enumerate(rows, start=2):
        pmid = (row.get("pmid") or "").strip()
        if not pmid:
            raise ValidationError(f"{path_label} has an empty PMID at row {row_number}.")
        if pmid in records:
            raise ValidationError(f"{path_label} has duplicate PMID {pmid}.")
        records[pmid] = row
    return records


def validate_input(canonical_columns, canonical_rows, recovery_columns, recovery_rows):
    required_columns = {"pmid", "record_type", *PROVENANCE_COLUMNS}
    missing_canonical = required_columns - set(canonical_columns)
    missing_recovery = required_columns - set(recovery_columns)
    require(
        not missing_canonical and not missing_recovery,
        "canonical and recovery files contain PMID, record_type, and provenance columns",
    )
    require(
        set(canonical_columns) == set(recovery_columns),
        "canonical and recovery files have matching column sets",
    )
    require(
        len(canonical_rows) == EXPECTED_TOTAL_ROWS,
        f"canonical row count is {EXPECTED_TOTAL_ROWS}",
    )
    canonical_by_pmid = index_unique_pmids(canonical_rows, "Canonical file")
    require(
        len(canonical_by_pmid) == EXPECTED_TOTAL_ROWS,
        "canonical PMIDs are unique",
    )
    require(
        len(recovery_rows) == EXPECTED_RECOVERY_ROWS,
        f"recovery row count is {EXPECTED_RECOVERY_ROWS}",
    )
    recovery_by_pmid = index_unique_pmids(recovery_rows, "Recovery file")
    require(
        len(recovery_by_pmid) == EXPECTED_RECOVERY_ROWS,
        "recovery PMIDs are unique",
    )
    require(
        all(row.get("record_type", "").strip() == "pubmed_book_article" for row in recovery_rows),
        "every recovery record_type is pubmed_book_article",
    )
    require(
        set(recovery_by_pmid).issubset(canonical_by_pmid),
        "all recovery PMIDs exist in the canonical file",
    )
    require(
        all(
            all((row.get(column) or "").strip() for column in PROVENANCE_COLUMNS)
            for row in recovery_rows
        ),
        "all recovery provenance columns are populated",
    )

    canonical_types = Counter(row.get("record_type", "").strip() for row in canonical_rows)
    require(
        canonical_types == Counter({
            "pubmed_article": EXPECTED_ARTICLE_ROWS,
            "pubmed_book_article": EXPECTED_RECOVERY_ROWS,
        }),
        "canonical input contains 34,829 PubmedArticle and 1,170 PubmedBookArticle records",
    )
    require(
        all(
            canonical_by_pmid[pmid].get("record_type", "").strip() == "pubmed_book_article"
            for pmid in recovery_by_pmid
        ),
        "every recovery PMID currently identifies a PubmedBookArticle row",
    )
    return canonical_by_pmid, recovery_by_pmid


def validate_temporary(
    temporary_path,
    canonical_columns,
    canonical_by_pmid,
    recovery_by_pmid,
):
    temporary_columns, temporary_rows = read_csv(temporary_path)
    require(
        temporary_columns == canonical_columns,
        "temporary output preserves canonical column order",
    )
    require(
        len(temporary_rows) == EXPECTED_TOTAL_ROWS,
        "temporary output has 35,999 total rows",
    )
    temporary_by_pmid = index_unique_pmids(temporary_rows, "Temporary output")
    duplicate_count = len(temporary_rows) - len(temporary_by_pmid)
    require(
        len(temporary_by_pmid) == EXPECTED_TOTAL_ROWS and duplicate_count == 0,
        "temporary output has 35,999 unique PMIDs and zero duplicates",
    )

    temporary_types = Counter(row.get("record_type", "").strip() for row in temporary_rows)
    require(
        temporary_types.get("pubmed_article", 0) == EXPECTED_ARTICLE_ROWS,
        "temporary output contains 34,829 PubmedArticle records",
    )
    require(
        temporary_types.get("pubmed_book_article", 0) == EXPECTED_RECOVERY_ROWS,
        "temporary output contains 1,170 PubmedBookArticle records",
    )
    require(
        all(pmid in temporary_by_pmid for pmid in recovery_by_pmid),
        "every recovered PMID appears exactly once",
    )

    original_non_recovery = {
        pmid: row for pmid, row in canonical_by_pmid.items() if pmid not in recovery_by_pmid
    }
    require(
        len(original_non_recovery) == EXPECTED_ARTICLE_ROWS
        and all(pmid in temporary_by_pmid for pmid in original_non_recovery),
        "all 34,829 original non-recovery PMIDs remain exactly once",
    )
    require(
        all(
            temporary_by_pmid[pmid] == original_row
            for pmid, original_row in original_non_recovery.items()
        ),
        "all original PubmedArticle records remain unchanged",
    )
    require(
        all(
            all(
                temporary_by_pmid[pmid].get(column, "") == recovered_row.get(column, "")
                for column in canonical_columns
            )
            for pmid, recovered_row in recovery_by_pmid.items()
        ),
        "recovered records and provenance match the recovery file unchanged",
    )
    require(
        set(temporary_by_pmid)
        == set(original_non_recovery) | set(recovery_by_pmid),
        "temporary output contains exactly the original non-recovery and refreshed PMID sets",
    )
    return temporary_rows


def main():
    temporary_path = None
    try:
        canonical_columns, canonical_rows = read_csv(CANONICAL_FILE)
        recovery_columns, recovery_rows = read_csv(RECOVERY_FILE)
        canonical_by_pmid, recovery_by_pmid = validate_input(
            canonical_columns,
            canonical_rows,
            recovery_columns,
            recovery_rows,
        )

        merged_rows = [
            recovery_by_pmid.get((row.get("pmid") or "").strip(), row)
            for row in canonical_rows
        ]

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=CANONICAL_FILE.parent,
            prefix=f"{CANONICAL_FILE.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.DictWriter(temporary_file, fieldnames=canonical_columns)
            writer.writeheader()
            writer.writerows(merged_rows)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        validate_temporary(
            temporary_path,
            canonical_columns,
            canonical_by_pmid,
            recovery_by_pmid,
        )
        os.replace(temporary_path, CANONICAL_FILE)
        temporary_path = None
        print(f"\nCanonical file atomically replaced: {CANONICAL_FILE.relative_to(PROJECT_ROOT)}")
        return 0
    except (OSError, UnicodeError, csv.Error, ValidationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        print("Canonical file was not replaced.", file=sys.stderr)
        return 1
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
