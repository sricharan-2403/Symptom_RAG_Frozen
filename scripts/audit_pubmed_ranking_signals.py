import csv
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_FILE = DATA_DIR / "pubmed_complete_metadata.csv"
MANIFEST_FILE = DATA_DIR / "disease_literature_manifest.csv"
REPORT_FILE = DATA_DIR / "pubmed_ranking_signal_audit.txt"

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
EVIDENCE_PUBLICATION_TYPES = (
    "Review",
    "Systematic Review",
    "Meta-Analysis",
    "Case Reports",
    "Practice Guideline",
    "Consensus Statement",
    "Randomized Controlled Trial",
    "Clinical Trial",
    "Observational Study",
    "Journal Article",
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
    "title",
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
    return "eng" in {value.casefold() for value in split_values(row.get("language"))}


def normalized_publication_types(row):
    return tuple(sorted({value.casefold() for value in split_values(row.get("publication_types"))}))


def query_categories(row):
    available = {value.casefold() for value in split_values(row.get("query_categories"))}
    return tuple(category for category in QUERY_CATEGORIES if category in available)


def is_stage3_eligible(row):
    if not has_english(row) or not has_value(row, "abstract"):
        return False
    publication_types = set(normalized_publication_types(row))
    return not bool(publication_types & EXCLUDED_PUBLICATION_TYPES)


def parse_year(row):
    for value in (row.get("publication_year"), row.get("publication_date")):
        if value:
            match = re.search(r"\b(\d{4})\b", value)
            if match:
                return int(match.group(1))
    return None


def load_diseases(manifest_rows):
    diseases = {}
    for row_number, row in enumerate(manifest_rows, start=2):
        disease_id = (row.get("disease_id") or "").strip()
        disease_name = (row.get("disease_name") or "").strip()
        if not disease_id or not disease_name:
            raise ValueError(f"Disease manifest has an empty ID or name at row {row_number}.")
        if disease_id in diseases:
            raise ValueError(f"Duplicate disease ID in manifest: {disease_id}")
        diseases[disease_id] = disease_name
    return diseases


def disease_sort_key(disease_id):
    try:
        return 0, int(disease_id)
    except ValueError:
        return 1, disease_id.casefold()


def format_combo(values):
    return " + ".join(values) if values else "(none)"


def analyze_diseases(stage3_rows, diseases):
    metrics = {
        disease_id: {
            "disease_name": name,
            "total_candidates": 0,
            "diagnosis": 0,
            "symptoms": 0,
            "clinical_presentation": 0,
            "pmcid": 0,
            "doi": 0,
            "years": [],
            "year_2020_plus": 0,
        }
        for disease_id, name in diseases.items()
    }
    disease_coverage_categories = defaultdict(set)

    for row in stage3_rows:
        categories = set(query_categories(row))
        year = parse_year(row)
        for disease_id in split_values(row.get("disease_ids")):
            if disease_id not in metrics:
                continue
            stats = metrics[disease_id]
            stats["total_candidates"] += 1
            for category in categories:
                stats[category] += 1
                disease_coverage_categories[disease_id].add(category)
            stats["pmcid"] += int(has_value(row, "pmcid"))
            stats["doi"] += int(has_value(row, "doi"))
            if year is not None:
                stats["years"].append(year)
                stats["year_2020_plus"] += int(year >= 2020)

    return metrics, disease_coverage_categories


def build_report(metadata_rows, manifest_rows):
    diseases = load_diseases(manifest_rows)
    stage3_rows = [row for row in metadata_rows if is_stage3_eligible(row)]

    publication_combinations = Counter(
        format_combo(normalized_publication_types(row))
        for row in stage3_rows
    )
    category_combinations = Counter(
        format_combo(query_categories(row))
        for row in stage3_rows
    )

    category_coverage = Counter()
    for row in stage3_rows:
        categories = set(query_categories(row))
        if categories == set(QUERY_CATEGORIES):
            category_coverage["all three"] += 1
        elif len(categories) == 2:
            labels = {
                ("diagnosis", "symptoms"): "diagnosis + symptoms",
                ("diagnosis", "clinical_presentation"): "diagnosis + clinical_presentation",
                ("symptoms", "clinical_presentation"): "symptoms + clinical_presentation",
            }
            category_coverage[labels[tuple(category for category in QUERY_CATEGORIES if category in categories)]] += 1
        elif len(categories) == 1:
            category_coverage[f"{next(iter(categories))} only"] += 1

    pmcid_only = doi_only = both_access = neither_access = 0
    for row in stage3_rows:
        has_pmcid = has_value(row, "pmcid")
        has_doi = has_value(row, "doi")
        if has_pmcid and has_doi:
            both_access += 1
        elif has_pmcid:
            pmcid_only += 1
        elif has_doi:
            doi_only += 1
        else:
            neither_access += 1

    publication_types = [set(normalized_publication_types(row)) for row in stage3_rows]
    evidence_counts = {
        publication_type: sum(publication_type.casefold() in values for values in publication_types)
        for publication_type in EVIDENCE_PUBLICATION_TYPES
    }
    book_article_count = sum(
        (row.get("record_type") or "").strip().casefold() == "pubmed_book_article"
        for row in stage3_rows
    )

    metrics, disease_categories = analyze_diseases(stage3_rows, diseases)
    diseases_with_all_categories = sum(
        set(QUERY_CATEGORIES) <= disease_categories.get(disease_id, set())
        for disease_id in diseases
    )
    disease_category_gaps = {
        category: sum(
            category not in disease_categories.get(disease_id, set())
            for disease_id in diseases
        )
        for category in QUERY_CATEGORIES
    }
    diseases_with_pmcid = sum(stats["pmcid"] > 0 for stats in metrics.values())
    diseases_without_pmcid = len(diseases) - diseases_with_pmcid

    represented = [
        (disease_id, stats)
        for disease_id, stats in metrics.items()
        if stats["total_candidates"] > 0
    ]
    lowest_access = sorted(
        represented,
        key=lambda item: (item[1]["pmcid"], disease_sort_key(item[0])),
    )[:30]
    highest_access = sorted(
        represented,
        key=lambda item: (-item[1]["pmcid"], disease_sort_key(item[0])),
    )[:20]

    recency_rows = []
    for disease_id, stats in metrics.items():
        years = stats["years"]
        recency_rows.append((
            disease_id,
            stats["disease_name"],
            min(years) if years else None,
            max(years) if years else None,
            stats["year_2020_plus"],
        ))
    dated_recency = [row for row in recency_rows if row[2] is not None]
    recent_counts = [row[4] for row in recency_rows]
    lowest_recent = sorted(
        (row for row in recency_rows if metrics[row[0]]["total_candidates"] > 0),
        key=lambda row: (row[4], disease_sort_key(row[0])),
    )[:20]

    multi_disease_rows = []
    for row in stage3_rows:
        association_count = len(split_values(row.get("disease_ids")))
        if association_count > 1:
            multi_disease_rows.append((association_count, row))
    multi_disease_rows.sort(
        key=lambda item: (
            -item[0],
            (item[1].get("pmid") or "").strip(),
        )
    )

    lines = []
    add = lines.append
    add("PUBMED RANKING SIGNAL AUDIT")
    add("=" * 80)
    add(f"Stage 3 eligible records analyzed: {len(stage3_rows)} of {len(metadata_rows)}")
    add("")

    add("SECTION 1 — PUBLICATION TYPE COMBINATIONS")
    add("Count | Normalized publication types")
    for combination, count in publication_combinations.most_common(30):
        add(f"{count} | {combination}")
    add("")

    add("SECTION 2 — QUERY CATEGORY COMBINATIONS")
    add("Count | Query categories")
    for combination, count in category_combinations.most_common():
        add(f"{count} | {combination}")
    add("")

    add("SECTION 3 — QUERY CATEGORY COVERAGE")
    for label in (
        "diagnosis only",
        "symptoms only",
        "clinical_presentation only",
        "diagnosis + symptoms",
        "diagnosis + clinical_presentation",
        "symptoms + clinical_presentation",
        "all three",
    ):
        add(f"{label}: {category_coverage[label]}")
    add("")

    add("SECTION 4 — FULL-TEXT ACCESS SIGNALS")
    add(f"PMCID only: {pmcid_only}")
    add(f"DOI only: {doi_only}")
    add(f"Both: {both_access}")
    add(f"Neither: {neither_access}")
    add("")

    add("SECTION 5 — EVIDENCE TYPE COUNTS")
    add("Counts overlap; a record may have multiple publication types.")
    for publication_type in EVIDENCE_PUBLICATION_TYPES:
        add(f"{publication_type}: {evidence_counts[publication_type]}")
    add(f"pubmed_book_article: {book_article_count}")
    add("")

    add("SECTION 6 — DISEASE-LEVEL CATEGORY COVERAGE")
    add("disease_id | disease_name | total_candidates | diagnosis | symptoms | clinical_presentation | PMCID | DOI")
    for disease_id in sorted(diseases, key=disease_sort_key):
        stats = metrics[disease_id]
        add(
            f"{disease_id} | {stats['disease_name']} | {stats['total_candidates']} | "
            f"{stats['diagnosis']} | {stats['symptoms']} | "
            f"{stats['clinical_presentation']} | {stats['pmcid']} | {stats['doi']}"
        )
    add("")
    add(f"Diseases with all 3 query categories represented: {diseases_with_all_categories}")
    add(f"Diseases missing diagnosis coverage: {disease_category_gaps['diagnosis']}")
    add(f"Diseases missing symptoms coverage: {disease_category_gaps['symptoms']}")
    add(f"Diseases missing clinical presentation coverage: {disease_category_gaps['clinical_presentation']}")
    add(f"Diseases with at least 1 PMCID candidate: {diseases_with_pmcid}")
    add(f"Diseases with 0 PMCID candidates: {diseases_without_pmcid}")
    add("")

    add("SECTION 7 — DISEASE-LEVEL ACCESSIBILITY")
    add("30 diseases with the fewest PMCID candidates among represented diseases:")
    add("disease_id | disease_name | PMCID candidates")
    for disease_id, stats in lowest_access:
        add(f"{disease_id} | {stats['disease_name']} | {stats['pmcid']}")
    add("20 diseases with the most PMCID candidates among represented diseases:")
    add("disease_id | disease_name | PMCID candidates")
    for disease_id, stats in highest_access:
        add(f"{disease_id} | {stats['disease_name']} | {stats['pmcid']}")
    add("")

    add("SECTION 8 — RECENCY BY DISEASE")
    add("disease_id | disease_name | oldest_publication_year | newest_publication_year | candidates_2020_onward")
    for disease_id, disease_name, oldest, newest, recent_count in sorted(
        recency_rows, key=lambda row: disease_sort_key(row[0])
    ):
        add(
            f"{disease_id} | {disease_name} | {oldest if oldest is not None else 'N/A'} | "
            f"{newest if newest is not None else 'N/A'} | {recent_count}"
        )
    add("")
    if dated_recency:
        oldest_years = [row[2] for row in dated_recency]
        newest_years = [row[3] for row in dated_recency]
        add(f"Diseases with at least one valid year: {len(dated_recency)}")
        add(f"Mean oldest year per disease: {statistics.mean(oldest_years):.1f}")
        add(f"Median oldest year per disease: {statistics.median(oldest_years):.1f}")
        add(f"Mean newest year per disease: {statistics.mean(newest_years):.1f}")
        add(f"Median newest year per disease: {statistics.median(newest_years):.1f}")
    else:
        add("Diseases with at least one valid year: 0")
    add(f"Diseases with at least one 2020+ candidate: {sum(count > 0 for count in recent_counts)}")
    add(f"Total candidates from 2020 onward: {sum(recent_counts)}")
    add("20 diseases with the lowest 2020+ candidate counts among represented diseases:")
    add("disease_id | disease_name | candidates_2020_onward")
    for disease_id, disease_name, _, _, recent_count in lowest_recent:
        add(f"{disease_id} | {disease_name} | {recent_count}")
    add("")

    add("SECTION 9 — MULTI-DISEASE ARTICLES")
    add(f"Total multi-disease records: {len(multi_disease_rows)}")
    add("Top 20 records by disease association count:")
    add("pmid | title | disease_association_count | query_categories")
    for association_count, row in multi_disease_rows[:20]:
        add(
            f"{(row.get('pmid') or '').strip()} | {(row.get('title') or '').strip()} | "
            f"{association_count} | {format_combo(query_categories(row))}"
        )
    add("")

    add("SECTION 10 — FINAL SUMMARY")
    add(
        f"The audit covers {len(stage3_rows)} Stage 3 records. "
        f"Category labels co-occur; full-text signals are PMCID/DOI indicators; "
        "publication years and disease associations are descriptive signals."
    )
    add(
        f"The dataset has {diseases_with_all_categories} diseases represented in all three query categories, "
        f"{diseases_with_pmcid} diseases with at least one PMCID candidate, and "
        f"{len(multi_disease_rows)} multi-disease records."
    )
    add("These counts describe available signals only and do not recommend a ranking formula or select papers.")

    return "\n".join(lines) + "\n", len(stage3_rows)


def main():
    if REPORT_FILE.exists():
        print(f"ERROR: Refusing to overwrite existing report: {REPORT_FILE}", file=sys.stderr)
        return 1
    try:
        metadata_rows = read_csv(METADATA_FILE, REQUIRED_METADATA_COLUMNS)
        manifest_rows = read_csv(MANIFEST_FILE, ("disease_id", "disease_name"))
        report, stage3_count = build_report(metadata_rows, manifest_rows)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"ERROR: Could not complete ranking signal audit: {error}", file=sys.stderr)
        return 1

    try:
        with REPORT_FILE.open("x", encoding="utf-8", newline="") as report_file:
            report_file.write(report)
    except OSError as error:
        print(f"ERROR: Could not write report without overwriting: {error}", file=sys.stderr)
        return 1

    print(report, end="")
    print("\nRANKING SIGNAL AUDIT COMPLETE")
    print(f"Records inspected: {stage3_count}")
    print(f"Report: {REPORT_FILE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())