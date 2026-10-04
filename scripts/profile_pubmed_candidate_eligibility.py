import csv
import math
import re
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_FILE = DATA_DIR / "pubmed_complete_metadata.csv"
MANIFEST_FILE = DATA_DIR / "disease_literature_manifest.csv"
REPORT_FILE = DATA_DIR / "pubmed_candidate_eligibility_profile.txt"

EXCLUDED_PUBLICATION_TYPES = {
    "retracted publication",
    "retraction notice",
    "newspaper article",
    "news",
    "biography",
    "portrait",
    "interview",
    "legal case",
    "directory",
    "address",
}
QUERY_CATEGORIES = ("diagnosis", "symptoms", "clinical_presentation")
DISTANCE_BUCKETS = (
    ("0", 0, 0),
    ("1-5", 1, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    (">100", 101, None),
)
REQUIRED_METADATA_COLUMNS = (
    "pmid",
    "language",
    "abstract",
    "publication_types",
    "publication_year",
    "publication_date",
    "pmcid",
    "doi",
    "record_type",
    "disease_ids",
    "disease_names",
    "query_categories",
)


def read_csv(path, required_columns):
    if not path.is_file():
        raise FileNotFoundError(f"Required input CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = set(required_columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path.name} is missing columns: " + ", ".join(sorted(missing))
            )
        return list(reader)


def split_values(value):
    return {part.strip() for part in (value or "").split("|") if part.strip()}


def has_value(row, field):
    return bool((row.get(field) or "").strip())


def has_english(row):
    return "eng" in {language.casefold() for language in split_values(row.get("language"))}


def publication_types(row):
    return {value.casefold() for value in split_values(row.get("publication_types"))}


def is_excluded_publication_type(row):
    return bool(publication_types(row) & EXCLUDED_PUBLICATION_TYPES)


def has_publication_type(row, publication_type):
    return publication_type.casefold() in publication_types(row)


def parse_publication_year(row):
    for value in (row.get("publication_year"), row.get("publication_date")):
        if value:
            match = re.search(r"\b(\d{4})\b", value)
            if match:
                return int(match.group(1))
    return None


def disease_manifest(rows):
    diseases = {}
    for row_number, row in enumerate(rows, start=2):
        disease_id = (row.get("disease_id") or "").strip()
        disease_name = (row.get("disease_name") or "").strip()
        if not disease_id or not disease_name:
            raise ValueError(f"Disease manifest has an empty ID or name at row {row_number}.")
        if disease_id in diseases:
            raise ValueError(f"Duplicate disease ID in manifest: {disease_id}")
        diseases[disease_id] = disease_name
    return diseases


def disease_metrics(stage3_rows, diseases):
    counts = {
        disease_id: {
            "disease_name": name,
            "candidate_count": 0,
            "records_with_abstract": 0,
            "records_with_pmc": 0,
            "records_with_doi": 0,
            "record_types": Counter(),
        }
        for disease_id, name in diseases.items()
    }
    unknown_disease_ids = set()

    for row in stage3_rows:
        for disease_id in split_values(row.get("disease_ids")):
            stats = counts.get(disease_id)
            if stats is None:
                unknown_disease_ids.add(disease_id)
                continue
            stats["candidate_count"] += 1
            stats["records_with_abstract"] += int(has_value(row, "abstract"))
            stats["records_with_pmc"] += int(has_value(row, "pmcid"))
            stats["records_with_doi"] += int(has_value(row, "doi"))
            record_type = (row.get("record_type") or "").strip() or "(empty)"
            stats["record_types"][record_type] += 1

    return counts, unknown_disease_ids


def disease_association_counts(rows):
    counts = Counter()
    for row in rows:
        association_count = len(split_values(row.get("disease_ids")))
        bucket = str(association_count) if association_count <= 5 else "6+"
        counts[bucket] += 1
    return counts


def format_distribution(counter):
    if not counter:
        return "(none)"
    return ", ".join(f"{key}={counter[key]}" for key in sorted(counter))


def build_report(metadata_rows, manifest_rows):
    diseases = disease_manifest(manifest_rows)
    stage1_rows = [row for row in metadata_rows if has_english(row)]
    stage2_rows = [row for row in stage1_rows if has_value(row, "abstract")]
    stage3_rows = [row for row in stage2_rows if not is_excluded_publication_type(row)]

    retracted_publication_all = sum(
        has_publication_type(row, "Retracted Publication") for row in metadata_rows
    )
    retraction_notice_all = sum(
        has_publication_type(row, "Retraction Notice") for row in metadata_rows
    )
    either_retraction_all = sum(
        bool(publication_types(row) & {"retracted publication", "retraction notice"})
        for row in metadata_rows
    )
    stage3_retraction_count = sum(
        bool(publication_types(row) & {"retracted publication", "retraction notice"})
        for row in stage3_rows
    )
    stage4_rows = [
        row for row in stage3_rows
        if not publication_types(row) & {"retracted publication", "retraction notice"}
    ]

    disease_counts, unknown_disease_ids = disease_metrics(stage3_rows, diseases)
    per_disease_values = [stats["candidate_count"] for stats in disease_counts.values()]
    represented_counts = [count for count in per_disease_values if count > 0]
    disease_buckets = {
        label: sum(
            lower <= count and (upper is None or count <= upper)
            for count in per_disease_values
        )
        for label, lower, upper in DISTANCE_BUCKETS
    }

    query_category_counts = Counter()
    for row in stage3_rows:
        query_category_counts.update(split_values(row.get("query_categories")))

    with_pmcid = sum(has_value(row, "pmcid") for row in stage3_rows)
    with_doi = sum(has_value(row, "doi") for row in stage3_rows)
    with_both = sum(has_value(row, "pmcid") and has_value(row, "doi") for row in stage3_rows)
    with_neither = sum(not has_value(row, "pmcid") and not has_value(row, "doi") for row in stage3_rows)
    record_type_counts = Counter(
        (row.get("record_type") or "").strip() or "(empty)"
        for row in stage3_rows
    )

    year_counts = Counter()
    missing_years = 0
    for row in stage3_rows:
        year = parse_publication_year(row)
        if year is None:
            missing_years += 1
        else:
            year_counts[year] += 1
    year_buckets = {
        "before 2000": sum(count for year, count in year_counts.items() if year < 2000),
        "2000-2009": sum(count for year, count in year_counts.items() if 2000 <= year <= 2009),
        "2010-2014": sum(count for year, count in year_counts.items() if 2010 <= year <= 2014),
        "2015-2019": sum(count for year, count in year_counts.items() if 2015 <= year <= 2019),
        "2020-2024": sum(count for year, count in year_counts.items() if 2020 <= year <= 2024),
        "2025": year_counts[2025],
        "2026": year_counts[2026],
        "2027+": sum(count for year, count in year_counts.items() if year >= 2027),
        "missing/invalid": missing_years,
    }

    bottom_diseases = sorted(
        (
            (disease_id, stats)
            for disease_id, stats in disease_counts.items()
            if stats["candidate_count"] > 0
        ),
        key=lambda item: (item[1]["candidate_count"], item[0]),
    )[:30]
    top_diseases = sorted(
        disease_counts.items(),
        key=lambda item: (-item[1]["candidate_count"], item[0]),
    )[:20]
    zero_diseases = sorted(
        (
            (disease_id, stats)
            for disease_id, stats in disease_counts.items()
            if stats["candidate_count"] == 0
        ),
        key=lambda item: item[0],
    )

    total_records = len(metadata_rows)
    duplicate_pmids = total_records - len({(row.get("pmid") or "").strip() for row in metadata_rows})
    lines = []
    add = lines.append
    add("PUBMED CANDIDATE ELIGIBILITY PROFILE")
    add("=" * 80)
    add("")

    add("1. PROGRESSIVE ELIGIBILITY COUNTS")
    add(f"STAGE 0 — ALL RECORDS: {total_records}")
    add(f"STAGE 1 — ENGLISH PRESENT: {len(stage1_rows)}")
    add(f"STAGE 2 — ENGLISH + ABSTRACT: {len(stage2_rows)}")
    add(f"STAGE 3 — ENGLISH + ABSTRACT + ACCEPTABLE PUBLICATION TYPE: {len(stage3_rows)}")
    add(f"STAGE 4 — STAGE 3 AFTER EXPLICIT RETRACTION REMOVAL: {len(stage4_rows)}")
    add(f"Unique PMIDs: {total_records - duplicate_pmids}")
    add(f"Duplicate PMID rows: {duplicate_pmids}")
    add("")

    add("2. EXCLUDED PUBLICATION TYPES")
    for publication_type in sorted(EXCLUDED_PUBLICATION_TYPES):
        count = sum(has_publication_type(row, publication_type) for row in metadata_rows)
        add(f"{publication_type}: {count}")
    add("")

    add("3. RETRACTION CHECK")
    add(f"Retracted Publication (all records): {retracted_publication_all}")
    add(f"Retraction Notice (all records): {retraction_notice_all}")
    add(f"Either retraction type (all records): {either_retraction_all}")
    add(f"Either retraction type remaining after Stage 3: {stage3_retraction_count}")
    add("")

    add("4. DISEASE COVERAGE")
    add(f"Unique diseases represented after Stage 3: {len(represented_counts)}")
    add(f"Disease universe from manifest: {len(diseases)}")
    add(f"Disease IDs in metadata absent from manifest: {len(unknown_disease_ids)}")
    for label, _, _ in DISTANCE_BUCKETS:
        add(f"Diseases with {label} candidates: {disease_buckets[label]}")
    if represented_counts:
        add(f"Minimum candidates per represented disease: {min(represented_counts)}")
        add(f"Maximum candidates per represented disease: {max(represented_counts)}")
        add(f"Mean candidates per represented disease: {statistics.mean(represented_counts):.2f}")
        add(f"Median candidates per represented disease: {statistics.median(represented_counts):.2f}")
    else:
        add("Minimum/maximum/mean/median candidates per represented disease: N/A")
    add("")

    add("5. QUERY CATEGORY COVERAGE AFTER STAGE 3")
    for category in QUERY_CATEGORIES:
        add(f"{category}: {query_category_counts[category]}")
    add("Category counts can overlap and are not mutually exclusive.")
    add("")

    add("6. FULL-TEXT SIGNALS AFTER STAGE 3")
    add(f"Records with PMCID: {with_pmcid}")
    add(f"Records without PMCID: {len(stage3_rows) - with_pmcid}")
    add(f"Records with DOI: {with_doi}")
    add(f"Records without DOI: {len(stage3_rows) - with_doi}")
    add(f"Records with both PMCID and DOI: {with_both}")
    add(f"Records with neither PMCID nor DOI: {with_neither}")
    add("")

    add("7. RECORD TYPES AFTER STAGE 3")
    add(f"pubmed_article: {record_type_counts['pubmed_article']}")
    add(f"pubmed_book_article: {record_type_counts['pubmed_book_article']}")
    other_record_types = sorted(
        set(record_type_counts) - {"pubmed_article", "pubmed_book_article"}
    )
    for record_type in other_record_types:
        add(f"{record_type}: {record_type_counts[record_type]}")
    add("")

    add("8. PUBLICATION YEAR DISTRIBUTION AFTER STAGE 3")
    valid_years = list(year_counts)
    add(f"Minimum year: {min(valid_years) if valid_years else 'N/A'}")
    add(f"Maximum year: {max(valid_years) if valid_years else 'N/A'}")
    for label, count in year_buckets.items():
        add(f"{label}: {count}")
    add("")

    association_counts = disease_association_counts(stage3_rows)
    add("9. DISEASE ASSOCIATION DISTRIBUTION AFTER STAGE 3")
    for label in ("1", "2", "3", "4", "5", "6+", "0"):
        display_label = "missing/zero disease associations" if label == "0" else f"exactly {label} disease association(s)"
        add(f"{display_label}: {association_counts[label]}")
    add("")

    def add_disease_table(title, disease_rows):
        add(title)
        add("disease_id | disease_name | candidate_count")
        for disease_id, stats in disease_rows:
            add(f"{disease_id} | {stats['disease_name']} | {stats['candidate_count']}")
        add("")

    add_disease_table("10. TOP 20 DISEASES", top_diseases)
    add_disease_table("11. BOTTOM 30 REPRESENTED DISEASES", bottom_diseases)

    add("12. ZERO-CANDIDATE DISEASES")
    add(f"Count: {len(zero_diseases)}")
    add("disease_id | disease_name")
    for disease_id, stats in zero_diseases:
        add(f"{disease_id} | {stats['disease_name']}")

    return "\n".join(lines) + "\n"


def main():
    if REPORT_FILE.exists():
        print(f"ERROR: Refusing to overwrite existing report: {REPORT_FILE}", file=sys.stderr)
        return 1

    try:
        metadata_rows = read_csv(METADATA_FILE, REQUIRED_METADATA_COLUMNS)
        manifest_rows = read_csv(MANIFEST_FILE, ("disease_id", "disease_name"))
        report = build_report(metadata_rows, manifest_rows)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"ERROR: Could not profile PubMed metadata: {error}", file=sys.stderr)
        return 1

    try:
        with REPORT_FILE.open("x", encoding="utf-8", newline="") as report_file:
            report_file.write(report)
    except OSError as error:
        print(f"ERROR: Could not write report without overwriting: {error}", file=sys.stderr)
        return 1

    print(report, end="")
    print("\nPROFILE COMPLETE")
    print(f"Records inspected: {len(metadata_rows)}")
    print(f"Report: {REPORT_FILE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())