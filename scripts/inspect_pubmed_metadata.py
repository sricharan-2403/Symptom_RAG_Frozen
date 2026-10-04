import csv
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_FILE = DATA_DIR / "pubmed_complete_metadata.csv"
MANIFEST_FILE = DATA_DIR / "disease_literature_manifest.csv"
REPORT_FILE = DATA_DIR / "pubmed_metadata_inspection_report.txt"

EXPECTED_DISEASE_COUNT = 866
RECORD_TYPE_ORDER = ("pubmed_article", "pubmed_book_article")
QUERY_CATEGORY_ORDER = ("diagnosis", "clinical_presentation", "symptoms")
DISEASE_COUNT_BUCKETS = (
    ("zero", 0, 0),
    ("1-5", 1, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    (">50", 51, None),
)
REQUIRED_METADATA_COLUMNS = (
    "pmid",
    "pmcid",
    "doi",
    "title",
    "abstract",
    "publication_date",
    "publication_year",
    "authors",
    "mesh_terms",
    "record_type",
    "disease_ids",
    "disease_names",
    "query_categories",
    "has_pmc",
)


def read_csv(path, required_columns):
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = set(required_columns) - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"{path.name} is missing columns: " + ", ".join(sorted(missing_columns))
            )
        return list(reader)


def split_values(value):
    return {part.strip() for part in (value or "").split("|") if part.strip()}


def has_value(row, column):
    return bool((row.get(column) or "").strip())


def publication_year(row):
    value = (row.get("publication_year") or "").strip()
    if not value:
        value = (row.get("publication_date") or "").strip()
    match = re.search(r"\b(\d{4})\b", value)
    return int(match.group(1)) if match else None


def percentage(count, total):
    return 100.0 * count / total if total else 0.0


def format_percent(count, total):
    return f"{percentage(count, total):.1f}%"


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def manifest_diseases(manifest_rows):
    diseases = {}
    for row_number, row in enumerate(manifest_rows, start=2):
        disease_id = (row.get("disease_id") or "").strip()
        disease_name = (row.get("disease_name") or "").strip()
        if not disease_id or not disease_name:
            raise ValueError(f"Disease manifest has an empty ID or name at row {row_number}.")
        if disease_id in diseases:
            raise ValueError(f"Duplicate disease_id in manifest: {disease_id}")
        diseases[disease_id] = disease_name
    return diseases


def analyze_diseases(metadata_rows, diseases):
    disease_stats = {
        disease_id: {
            "disease_name": name,
            "candidate_record_count": 0,
            "records_with_abstract": 0,
            "records_with_pmc": 0,
            "records_with_doi": 0,
            "record_types": Counter(),
        }
        for disease_id, name in diseases.items()
    }
    disease_association_distribution = Counter()
    query_category_distribution = Counter()
    unknown_disease_ids = Counter()

    seen_pmids = set()
    for row in metadata_rows:
        pmid = (row.get("pmid") or "").strip()
        if not pmid or pmid in seen_pmids:
            continue
        seen_pmids.add(pmid)

        disease_ids = split_values(row.get("disease_ids"))
        disease_association_distribution[
            "exactly 1" if len(disease_ids) == 1
            else "2" if len(disease_ids) == 2
            else "3+" if len(disease_ids) >= 3
            else "0"
        ] += 1

        record_type = (row.get("record_type") or "").strip() or "(empty)"
        for disease_id in disease_ids:
            if disease_id not in disease_stats:
                unknown_disease_ids[disease_id] += 1
                continue
            stats = disease_stats[disease_id]
            stats["candidate_record_count"] += 1
            stats["records_with_abstract"] += bool(has_value(row, "abstract"))
            stats["records_with_pmc"] += bool(
                has_value(row, "pmcid") or (row.get("has_pmc") or "").strip().casefold() == "true"
            )
            stats["records_with_doi"] += bool(has_value(row, "doi"))
            stats["record_types"][record_type] += 1

        query_category_distribution.update(split_values(row.get("query_categories")))

    return disease_stats, disease_association_distribution, query_category_distribution, unknown_disease_ids


def build_report(metadata_rows, manifest_rows):
    diseases = manifest_diseases(manifest_rows)
    disease_stats, association_counts, query_categories, unknown_ids = analyze_diseases(
        metadata_rows, diseases
    )

    pmids = [(row.get("pmid") or "").strip() for row in metadata_rows]
    nonempty_pmids = [pmid for pmid in pmids if pmid]
    unique_pmids = set(nonempty_pmids)
    duplicate_pmids = len(nonempty_pmids) - len(unique_pmids)
    record_type_counts = Counter(
        (row.get("record_type") or "").strip() or "(empty)"
        for row in metadata_rows
    )
    completeness_fields = {
        "title": "title",
        "abstract": "abstract",
        "PMCID": "pmcid",
        "DOI": "doi",
        "MeSH terms": "mesh_terms",
        "authors": "authors",
        "publication date": "publication_date",
    }
    completeness = {
        label: sum(has_value(row, column) for row in metadata_rows)
        for label, column in completeness_fields.items()
    }

    years = Counter()
    missing_year_count = 0
    for row in metadata_rows:
        year = publication_year(row)
        if year is None:
            missing_year_count += 1
        else:
            years[year] += 1
    future_year_counts = {
        year: count for year, count in years.items() if year > date.today().year
    }

    disease_counts = [stats["candidate_record_count"] for stats in disease_stats.values()]
    represented_diseases = sum(count > 0 for count in disease_counts)
    bucket_counts = {
        label: sum(
            lower <= count and (upper is None or count <= upper)
            for count in disease_counts
        )
        for label, lower, upper in DISEASE_COUNT_BUCKETS
    }
    all_diseases = sorted(
        disease_stats.items(),
        key=lambda item: (
            item[1]["candidate_record_count"],
            item[1]["disease_name"].casefold(),
            item[0],
        ),
    )
    lowest_diseases = all_diseases[:30]
    highest_diseases = list(reversed(all_diseases[-30:]))
    disease_record_type_names = sorted({
        record_type
        for stats in disease_stats.values()
        for record_type in stats["record_types"]
    })

    total_disease_links = sum(disease_counts)
    top_decile_count = max(1, math.ceil(len(disease_counts) * 0.10)) if disease_counts else 0
    top_decile_links = sum(sorted(disease_counts, reverse=True)[:top_decile_count])
    median_disease_count = statistics.median(disease_counts) if disease_counts else 0
    maximum_disease_count = max(disease_counts, default=0)
    top_decile_share = percentage(top_decile_links, total_disease_links)
    max_median_ratio = (
        maximum_disease_count / median_disease_count
        if median_disease_count
        else float("inf") if maximum_disease_count else 0.0
    )
    heavily_skewed = top_decile_share > 25.0 or max_median_ratio > 3.0

    lines = []
    add = lines.append
    add("PUBMED METADATA QUALITY AND COVERAGE REPORT")
    add("=" * 88)
    add("")

    add("OVERALL RECORDS")
    add("-" * 88)
    add(f"Total metadata records: {len(metadata_rows)}")
    add(f"Unique PMIDs: {len(unique_pmids)}")
    add(f"Duplicate PMIDs: {duplicate_pmids}")
    add(f"Empty PMID rows: {len(metadata_rows) - len(nonempty_pmids)}")
    add("record_type distribution:")
    for record_type, count in sorted(record_type_counts.items()):
        add(f"  {record_type}: {count}")
    add("")

    add("METADATA COMPLETENESS")
    add("-" * 88)
    for label, count in completeness.items():
        add(f"Records with {label}: {count} ({format_percent(count, len(metadata_rows))})")
    add("")

    add("PUBLICATION YEAR DISTRIBUTION")
    add("-" * 88)
    add(f"Minimum year: {min(years) if years else 'N/A'}")
    add(f"Maximum year: {max(years) if years else 'N/A'}")
    add(f"Records without a parseable publication year: {missing_year_count}")
    add("Counts by year:")
    for year in sorted(years):
        add(f"  {year}: {years[year]}")
    add("")

    add("DISEASE COVERAGE (DISEASE ID IS THE DISEASE KEY)")
    add("-" * 88)
    add(f"Expected diseases from manifest: {EXPECTED_DISEASE_COUNT}")
    add(f"Disease IDs in manifest: {len(diseases)}")
    add(f"Unique diseases represented: {represented_diseases}")
    add(f"Disease IDs in metadata but absent from manifest: {sum(unknown_ids.values())}")
    add("Diseases by candidate-record count:")
    for label, _, _ in DISEASE_COUNT_BUCKETS:
        add(f"  {label}: {bucket_counts[label]}")
    add("")

    table_header = (
        "disease_name | disease_id | candidate_record_count | records_with_abstract | "
        "records_with_pmc | records_with_doi | record_type counts"
    )
    add("PER-DISEASE METRICS (ALL MANIFEST DISEASES)")
    add("-" * 88)
    add(table_header)
    for disease_id, stats in sorted(
        disease_stats.items(), key=lambda item: (item[1]["disease_name"].casefold(), item[0])
    ):
        type_counts = ", ".join(
            f"{record_type}={stats['record_types'].get(record_type, 0)}"
            for record_type in disease_record_type_names
        ) or "(none)"
        add(
            f"{stats['disease_name']} | {disease_id} | {stats['candidate_record_count']} | "
            f"{stats['records_with_abstract']} | {stats['records_with_pmc']} | "
            f"{stats['records_with_doi']} | {type_counts}"
        )
    add("")

    add("30 DISEASES WITH THE LOWEST METADATA RECORD COUNTS")
    add("-" * 88)
    add("disease_name | disease_id | candidate_record_count")
    for disease_id, stats in lowest_diseases:
        add(f"{stats['disease_name']} | {disease_id} | {stats['candidate_record_count']}")
    add("")

    add("30 DISEASES WITH THE HIGHEST METADATA RECORD COUNTS")
    add("-" * 88)
    add("disease_name | disease_id | candidate_record_count")
    for disease_id, stats in highest_diseases:
        add(f"{stats['disease_name']} | {disease_id} | {stats['candidate_record_count']}")
    add("")

    add("DISEASE ASSOCIATIONS PER RECORD")
    add("-" * 88)
    add(f"Exactly 1 disease: {association_counts['exactly 1']}")
    add(f"2 diseases: {association_counts['2']}")
    add(f"3+ diseases: {association_counts['3+']}")
    add(f"No disease ID: {association_counts['0']}")
    add("")

    add("QUERY-CATEGORY DISTRIBUTION")
    add("-" * 88)
    for category in QUERY_CATEGORY_ORDER:
        add(f"{category}: {query_categories[category]}")
    unknown_categories = sorted(set(query_categories) - set(QUERY_CATEGORY_ORDER))
    for category in unknown_categories:
        add(f"{category}: {query_categories[category]} (unrecognized category)")
    add("")

    add("INTERPRETATION FOR NEXT FILTERING STAGE")
    add("-" * 88)
    abstract_count = completeness["abstract"]
    pmcid_count = completeness["PMCID"]
    overall_completeness = percentage(
        sum(completeness.values()), len(metadata_rows) * len(completeness)
    )
    add(
        f"Overall completeness across {len(completeness)} tracked metadata fields is "
        f"{overall_completeness:.1f}% by field presence."
    )
    if percentage(abstract_count, len(metadata_rows)) >= 95:
        add(f"Overall metadata completeness is strong for abstracts ({format_percent(abstract_count, len(metadata_rows))}).")
    else:
        add(f"Abstract coverage is {format_percent(abstract_count, len(metadata_rows))}; account for missing abstracts during filtering.")
    add(
        f"PMCID is present for {pmcid_count}/{len(metadata_rows)} records "
        f"({format_percent(pmcid_count, len(metadata_rows))}); this indicates a PMC link, "
        "not guaranteed full-text access or licensing."
    )
    weak_disease_count = bucket_counts["zero"] + bucket_counts["1-5"]
    add(
        f"Weak literature coverage (0-5 records) affects {weak_disease_count} of "
        f"{len(diseases)} diseases. Zero-candidate diseases: {bucket_counts['zero']}; "
        f"1-5 records: {bucket_counts['1-5']}."
    )
    add(
        f"Candidate distribution is {'heavily skewed' if heavily_skewed else 'not heavily skewed'} "
        f"under the report rule (top 10% share={top_decile_share:.1f}%; "
        f"max/median={max_median_ratio:.2f}; heavy if >25% or >3x)."
    )
    if duplicate_pmids:
        add("Duplicate PMIDs exist; deduplicate before article-level filtering.")
    missing_title_count = len(metadata_rows) - completeness["title"]
    missing_abstract_count = len(metadata_rows) - completeness["abstract"]
    missing_author_count = len(metadata_rows) - completeness["authors"]
    if missing_title_count or missing_abstract_count:
        add("Some records lack a title or abstract; avoid treating missing text as negative evidence.")
    if missing_author_count:
        add(f"{missing_author_count} records lack authors; author-based filtering would exclude them unless handled explicitly.")
    add("Keep PubmedArticle and PubmedBookArticle types visible as separate source classes during downstream review.")
    add("Disease coverage is grouped by disease_id because some manifest disease names are shared by multiple IDs.")
    if future_year_counts:
        future_count = sum(future_year_counts.values())
        future_year_summary = ", ".join(
            f"{year}: {count}" for year, count in sorted(future_year_counts.items())
        )
        add(
            f"Review {future_count} record(s) dated after the current year "
            f"({future_year_summary}) before time-sensitive filtering."
        )
    add("This is a retrieval/data-quality report only; no disease is ranked by medical importance.")

    return "\n".join(lines) + "\n"


def main():
    overwrite = "--overwrite" in sys.argv[1:]
    if REPORT_FILE.exists() and not overwrite:
        print(f"ERROR: Refusing to overwrite existing report: {REPORT_FILE}", file=sys.stderr)
        return 1

    try:
        metadata_rows = read_csv(METADATA_FILE, REQUIRED_METADATA_COLUMNS)
        manifest_rows = read_csv(MANIFEST_FILE, ("disease_id", "disease_name"))
        report = build_report(metadata_rows, manifest_rows)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"ERROR: Could not create inspection report: {error}", file=sys.stderr)
        return 1

    try:
        mode = "w" if overwrite else "x"
        with REPORT_FILE.open(mode, encoding="utf-8", newline="") as report_file:
            report_file.write(report)
    except OSError as error:
        print(f"ERROR: Could not save report without overwriting: {error}", file=sys.stderr)
        return 1

    print(report, end="")
    print(f"\nReport saved to: {REPORT_FILE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())