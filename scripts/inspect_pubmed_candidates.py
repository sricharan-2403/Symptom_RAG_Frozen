import csv
import math
import statistics
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_FILE = (
    PROJECT_ROOT / "data" / "processed" / "pubmed_candidate_articles.csv"
)
MANIFEST_FILE = (
    PROJECT_ROOT / "data" / "processed" / "disease_literature_manifest.csv"
)

CANDIDATE_COLUMNS = {
    "pmid",
    "disease_ids",
    "disease_names",
    "query_categories",
    "queries",
    "best_rank",
    "match_counts",
}
QUERY_CATEGORIES = ("diagnosis", "clinical_presentation", "symptoms")


def read_csv(path):
    warnings = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file)
            raw_headers = next(reader, [])
            headers = [header.strip() for header in raw_headers]
            if raw_headers and raw_headers[0].startswith("\ufeff"):
                warnings.append("Removed a byte-order mark from the first column name.")
            if any(header != cleaned for header, cleaned in zip(raw_headers, headers)):
                warnings.append("Trimmed surrounding whitespace from column names.")

            duplicate_headers = sorted(
                header for header, count in Counter(headers).items() if count > 1
            )
            if duplicate_headers:
                warnings.append(
                    "Duplicate column names were found: "
                    + ", ".join(duplicate_headers)
                )

            records = []
            for row_number, values in enumerate(reader, start=2):
                if len(values) != len(headers):
                    warnings.append(
                        f"Row {row_number} has {len(values)} fields; "
                        f"expected {len(headers)}."
                    )
                values = (values + [""] * len(headers))[: len(headers)]
                records.append(dict(zip(headers, values)))
    except (OSError, UnicodeError, csv.Error) as error:
        raise RuntimeError(f"Could not safely read {path}: {error}") from error

    return headers, records, warnings


def split_values(value):
    return {part.strip() for part in value.split("|") if part.strip()}


def disease_key(name):
    return " ".join(name.split()).casefold()


def percentile(values, percentage):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentage
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def format_number(value):
    if value is None:
        return "N/A"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def main():
    try:
        candidate_headers, candidates, candidate_warnings = read_csv(CANDIDATE_FILE)
        manifest_headers, manifest, manifest_warnings = read_csv(MANIFEST_FILE)
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    warnings = [
        f"Candidate CSV: {warning}" for warning in candidate_warnings
    ] + [f"Disease manifest: {warning}" for warning in manifest_warnings]

    print("=" * 72)
    print("PUBMED CANDIDATE INSPECTION")
    print("=" * 72)
    print("\n[Candidate file]")
    print(f"Path: {CANDIDATE_FILE}")
    print(f"Rows: {len(candidates)}")
    print(f"Columns ({len(candidate_headers)}): {', '.join(candidate_headers) or '(none)'}")

    missing_candidate_columns = sorted(CANDIDATE_COLUMNS - set(candidate_headers))
    if missing_candidate_columns:
        warnings.append(
            "Candidate CSV is missing required columns: "
            + ", ".join(missing_candidate_columns)
        )
    if "disease_name" not in manifest_headers:
        warnings.append("Disease manifest is missing required column: disease_name")

    pmids = [row.get("pmid", "").strip() for row in candidates]
    empty_pmids = sum(not pmid for pmid in pmids)
    pmid_counts = Counter(pmid for pmid in pmids if pmid)
    duplicate_pmids = sum(count > 1 for count in pmid_counts.values())
    duplicate_pmid_rows = sum(count - 1 for count in pmid_counts.values() if count > 1)
    print(f"Unique non-empty PMIDs: {len(pmid_counts)}")
    print(f"Duplicate PMID values: {duplicate_pmids}")
    print(f"Extra rows from duplicate PMIDs: {duplicate_pmid_rows}")

    disease_counts = Counter()
    disease_labels = {}
    represented_diseases = set()
    empty_disease_rows = 0
    category_counts = Counter()
    category_overlap = Counter()
    unknown_categories = set()
    best_rank_counts = Counter()
    invalid_best_rank_rows = 0

    for row in candidates:
        diseases = split_values(row.get("disease_names", ""))
        if not diseases:
            empty_disease_rows += 1
        for disease in diseases:
            key = disease_key(disease)
            represented_diseases.add(key)
            disease_labels.setdefault(key, disease)
            disease_counts[key] += 1

        categories = split_values(row.get("query_categories", ""))
        if categories:
            category_overlap[len(categories)] += 1
        for category in categories:
            category_counts[category] += 1
            if category not in QUERY_CATEGORIES:
                unknown_categories.add(category)

        best_rank = row.get("best_rank", "").strip()
        if best_rank:
            try:
                best_rank_counts[int(best_rank)] += 1
            except ValueError:
                invalid_best_rank_rows += 1
        else:
            invalid_best_rank_rows += 1

    if empty_pmids:
        warnings.append(f"{empty_pmids} candidate row(s) have an empty PMID.")
    if empty_disease_rows:
        warnings.append(
            f"{empty_disease_rows} candidate row(s) have no non-empty disease name."
        )
    if invalid_best_rank_rows:
        warnings.append(
            f"{invalid_best_rank_rows} candidate row(s) have a missing or invalid best_rank."
        )
    if unknown_categories:
        warnings.append(
            "Unrecognized query categories found: "
            + ", ".join(sorted(unknown_categories))
        )

    expected_diseases = {}
    empty_manifest_disease_rows = 0
    for row in manifest:
        disease = row.get("disease_name", "").strip()
        if not disease:
            empty_manifest_disease_rows += 1
            continue
        expected_diseases.setdefault(disease_key(disease), disease)
    if empty_manifest_disease_rows:
        warnings.append(
            f"{empty_manifest_disease_rows} manifest row(s) have an empty disease_name."
        )

    zero_candidate_diseases = [
        expected_diseases[key]
        for key in expected_diseases
        if key not in represented_diseases
    ]
    candidate_only_diseases = represented_diseases - set(expected_diseases)
    if candidate_only_diseases:
        warnings.append(
            f"{len(candidate_only_diseases)} candidate disease name(s) are absent "
            "from the expected manifest."
        )

    print("\n[Disease coverage]")
    print(f"Diseases represented in candidates: {len(represented_diseases)}")
    print(f"Expected diseases in manifest: {len(expected_diseases)}")
    print(f"Diseases with at least one candidate: {len(expected_diseases) - len(zero_candidate_diseases)}")
    print(f"Diseases with zero candidates: {len(zero_candidate_diseases)}")
    if zero_candidate_diseases:
        print("Zero-candidate disease names:")
        for disease in sorted(zero_candidate_diseases, key=str.casefold):
            print(f"  - {disease}")
    else:
        print("Zero-candidate disease names: (none)")

    counts_for_expected = [
        disease_counts.get(key, 0) for key in expected_diseases
    ]
    print("\n[Candidate count per expected disease]")
    if counts_for_expected:
        print(f"Minimum: {format_number(min(counts_for_expected))}")
        print(f"Maximum: {format_number(max(counts_for_expected))}")
        print(f"Mean: {format_number(statistics.mean(counts_for_expected))}")
        print(f"Median: {format_number(statistics.median(counts_for_expected))}")
        print(f"25th percentile: {format_number(percentile(counts_for_expected, 0.25))}")
        print(f"75th percentile: {format_number(percentile(counts_for_expected, 0.75))}")
    else:
        print("No valid expected disease names; statistics are unavailable.")

    print("\n[Top 20 diseases by candidate count]")
    ranked_diseases = sorted(
        expected_diseases,
        key=lambda key: (-disease_counts.get(key, 0), expected_diseases[key].casefold()),
    )
    if ranked_diseases:
        for position, key in enumerate(ranked_diseases[:20], start=1):
            print(f"{position:>2}. {expected_diseases[key]}: {disease_counts.get(key, 0)}")
    else:
        print("(no expected diseases)")

    print("\n[Query-category coverage]")
    for category in QUERY_CATEGORIES:
        print(f"{category}: {category_counts[category]}")

    print("\n[Query-category overlap per candidate row]")
    print(f"Exactly 1 category: {category_overlap[1]}")
    print(f"Exactly 2 categories: {category_overlap[2]}")
    print(f"All 3 categories: {category_overlap[3]}")
    print(f"Other category counts: {sum(count for size, count in category_overlap.items() if size not in (1, 2, 3))}")

    print("\n[best_rank distribution]")
    if best_rank_counts:
        for rank in sorted(best_rank_counts):
            print(f"Rank {rank}: {best_rank_counts[rank]}")
    else:
        print("(no valid best_rank values)")

    print("\n[Schema validation]")
    if missing_candidate_columns:
        print("Missing required candidate columns: " + ", ".join(missing_candidate_columns))
    else:
        print("Required candidate columns: present")
    print(f"Non-empty PMID values: {len(pmid_counts)} / {len(candidates)}")
    print(f"Non-empty disease names: {len(candidates) - empty_disease_rows} / {len(candidates)} candidate rows")
    print("Manifest disease_name column: " + ("present" if "disease_name" in manifest_headers else "missing"))

    print("\n[Warnings]")
    if warnings:
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("None")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())