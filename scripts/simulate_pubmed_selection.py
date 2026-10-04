import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_FILE = DATA_DIR / "pubmed_complete_metadata.csv"
MANIFEST_FILE = DATA_DIR / "disease_literature_manifest.csv"
REPORT_FILE = DATA_DIR / "pubmed_selection_simulation.txt"

CAPS = (3, 5, 8, 10)
QUERY_CATEGORIES = ("diagnosis", "symptoms", "clinical_presentation")
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
EVIDENCE_FLAGS = (
    ("journal_article", {"journal_article"}),
    ("review", {"review"}),
    ("systematic_review", {"systematic_review"}),
    ("meta_analysis", {"meta_analysis", "meta_analyses"}),
    ("case_report", {"case_report", "case_reports"}),
    ("practice_guideline", {"practice_guideline", "practice_guidelines"}),
    ("consensus_statement", {"consensus_statement", "consensus_statements"}),
    (
        "randomized_controlled_trial",
        {"randomized_controlled_trial", "randomised_controlled_trial"},
    ),
    ("clinical_trial", {"clinical_trial", "clinical_trials"}),
    ("observational_study", {"observational_study", "observational_studies"}),
    ("book_article", {"book_article"}),
)
REQUIRED_METADATA_COLUMNS = (
    "pmid",
    "title",
    "language",
    "abstract",
    "publication_types",
    "publication_year",
    "publication_date",
    "pmcid",
    "doi",
    "record_type",
    "disease_ids",
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


def normalize_type(value):
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def publication_type_profile(row):
    return tuple(sorted({normalize_type(value) for value in split_values(row.get("publication_types"))}))


def evidence_flags(row):
    types = set(publication_type_profile(row))
    flags = {
        flag
        for flag, aliases in EVIDENCE_FLAGS
        if types & aliases
    }
    if (row.get("record_type") or "").strip().casefold() == "pubmed_book_article":
        flags.add("book_article")
    return flags


def row_categories(row):
    available = {value.casefold() for value in split_values(row.get("query_categories"))}
    return {category for category in QUERY_CATEGORIES if category in available}


def publication_year(row):
    for value in (row.get("publication_year"), row.get("publication_date")):
        if value:
            match = re.search(r"\b(\d{4})\b", value)
            if match:
                return int(match.group(1))
    return None


def disease_sort_key(disease_id):
    try:
        return 0, int(disease_id)
    except ValueError:
        return 1, disease_id.casefold()


def load_manifest(rows):
    diseases = {}
    for row_number, row in enumerate(rows, start=2):
        disease_id = (row.get("disease_id") or "").strip()
        disease_name = (row.get("disease_name") or "").strip()
        if not disease_id or not disease_name:
            raise ValueError(f"Manifest row {row_number} has an empty disease ID or name.")
        if disease_id in diseases:
            raise ValueError(f"Duplicate disease_id in manifest: {disease_id}")
        diseases[disease_id] = disease_name
    return diseases


def stage3_eligible(row):
    languages = {value.casefold() for value in split_values(row.get("language"))}
    if "eng" not in languages or not has_value(row, "abstract"):
        return False
    pub_types = {value.casefold() for value in split_values(row.get("publication_types"))}
    return not bool(pub_types & EXCLUDED_PUBLICATION_TYPES)


def choose_candidate(available_rows, selected_rows, selected_category_counts, category):
    if category is None:
        choices = list(available_rows)
    else:
        choices = [row for row in available_rows if category in row_categories(row)]

    selected_profiles = {publication_type_profile(row) for row in selected_rows}
    novel_profile_choices = [
        row for row in choices
        if publication_type_profile(row) not in selected_profiles
    ]
    if novel_profile_choices:
        choices = novel_profile_choices

    abstract_choices = [row for row in choices if has_value(row, "abstract")]
    if abstract_choices:
        choices = abstract_choices

    pmcid_choices = [row for row in choices if has_value(row, "pmcid")]
    if pmcid_choices:
        choices = pmcid_choices

    def tie_break(row):
        year = publication_year(row)
        return (
            -(year if year is not None else -1),
            -int(has_value(row, "pmcid")),
            (row.get("pmid") or "").strip(),
        )

    return min(choices, key=tie_break)


def simulate_disease(candidates, maximum_selections):
    available = sorted(candidates, key=lambda row: (row.get("pmid") or "").strip())
    selected = []
    selected_pmids = set()
    category_counts = Counter()

    while available and len(selected) < maximum_selections:
        remaining_categories = set().union(*(row_categories(row) for row in available))
        if remaining_categories:
            category = min(
                remaining_categories,
                key=lambda value: (category_counts[value], QUERY_CATEGORIES.index(value)),
            )
        else:
            category = None

        candidate = choose_candidate(
            available,
            selected,
            category_counts,
            category,
        )
        pmid = (candidate.get("pmid") or "").strip()
        if pmid not in selected_pmids:
            selected.append(candidate)
            selected_pmids.add(pmid)
            category_counts.update(row_categories(candidate))
        available = [row for row in available if (row.get("pmid") or "").strip() != pmid]

    return selected


def selected_for_cap(selections_by_disease, cap):
    return {
        disease_id: selected_rows[:cap]
        for disease_id, selected_rows in selections_by_disease.items()
    }


def selected_assignments(selection):
    return [
        (disease_id, row)
        for disease_id, rows in selection.items()
        for row in rows
    ]


def unique_selected_records(selection):
    records = {}
    for _, row in selected_assignments(selection):
        pmid = (row.get("pmid") or "").strip()
        if pmid:
            records.setdefault(pmid, row)
    return records


def access_counts(records):
    pmcid = sum(has_value(row, "pmcid") for row in records.values())
    doi = sum(has_value(row, "doi") for row in records.values())
    both = sum(has_value(row, "pmcid") and has_value(row, "doi") for row in records.values())
    neither = sum(not has_value(row, "pmcid") and not has_value(row, "doi") for row in records.values())
    return pmcid, doi, both, neither


def recency_bucket(year):
    if year is None:
        return "missing/invalid"
    if year < 2000:
        return "before 2000"
    if year <= 2009:
        return "2000-2009"
    if year <= 2014:
        return "2010-2014"
    if year <= 2019:
        return "2015-2019"
    if year <= 2024:
        return "2020-2024"
    if year == 2025:
        return "2025"
    if year == 2026:
        return "2026"
    return "2027+"


def summarize_cap(cap, selection, diseases):
    assignments = selected_assignments(selection)
    records = unique_selected_records(selection)
    covered = sum(bool(rows) for rows in selection.values())
    full_cap = sum(len(rows) == cap for rows in selection.values())
    below_cap = sum(len(rows) < cap for rows in selection.values())

    disease_category_counts = {}
    for disease_id, rows in selection.items():
        categories = set().union(*(row_categories(row) for row in rows)) if rows else set()
        disease_category_counts[disease_id] = len(categories)

    selected_assignment_category_counts = Counter()
    selected_unique_category_pmids = {category: set() for category in QUERY_CATEGORIES}
    for _, row in assignments:
        categories = row_categories(row)
        selected_assignment_category_counts.update(categories)
    for pmid, row in records.items():
        for category in row_categories(row):
            selected_unique_category_pmids[category].add(pmid)

    evidence_counts = Counter()
    for row in records.values():
        evidence_counts.update(evidence_flags(row))

    years = Counter(recency_bucket(publication_year(row)) for row in records.values())
    pmcid_count, doi_count, both_count, neither_count = access_counts(records)
    category_receipt_counts = Counter(disease_category_counts.values())

    return {
        "cap": cap,
        "selection": selection,
        "assignments": assignments,
        "records": records,
        "selected_record_disease_assignments": len(assignments),
        "unique_pmids": len(records),
        "diseases_covered": covered,
        "diseases_full_cap": full_cap,
        "diseases_below_cap": below_cap,
        "diseases_all_three_categories": category_receipt_counts[3],
        "diseases_two_categories": category_receipt_counts[2],
        "diseases_one_category": category_receipt_counts[1],
        "diseases_zero_categories": category_receipt_counts[0],
        "assignment_category_counts": selected_assignment_category_counts,
        "unique_category_pmids": selected_unique_category_pmids,
        "evidence_counts": evidence_counts,
        "years": years,
        "pmcid": pmcid_count,
        "doi": doi_count,
        "both": both_count,
        "neither": neither_count,
        "category_receipt_counts": category_receipt_counts,
    }


def format_report(metadata_rows, diseases, candidates_by_disease):
    stage3_count = sum(stage3_eligible(row) for row in metadata_rows)
    represented = sum(bool(rows) for rows in candidates_by_disease.values())
    zero_diseases = [
        disease_id for disease_id, rows in candidates_by_disease.items() if not rows
    ]
    max_cap = max(CAPS)
    selected_to_max = {
        disease_id: simulate_disease(rows, max_cap)
        for disease_id, rows in candidates_by_disease.items()
    }
    cap_summaries = {
        cap: summarize_cap(cap, selected_for_cap(selected_to_max, cap), diseases)
        for cap in CAPS
    }

    lines = []
    add = lines.append
    add("PUBMED DISEASE-AWARE SELECTION SIMULATION")
    add("=" * 80)
    add("")

    add("SECTION 1 — INPUT SUMMARY")
    add(f"Total metadata records: {len(metadata_rows)}")
    add(f"Stage 3 eligible records: {stage3_count}")
    add(f"Unique diseases in manifest: {len(diseases)}")
    add(f"Diseases represented in eligible pool: {represented}")
    add(f"Zero-candidate diseases: {len(zero_diseases)}")
    add("")

    add("SECTION 2 — SIMULATION SUMMARY")
    for cap in CAPS:
        summary = cap_summaries[cap]
        add(f"Cap: {cap} articles per disease")
        add(f"  Selected record-disease assignments: {summary['selected_record_disease_assignments']}")
        add(f"  Unique selected PMIDs: {summary['unique_pmids']}")
        add(f"  Diseases receiving at least 1 candidate: {summary['diseases_covered']}")
        add(f"  Diseases receiving the full cap: {summary['diseases_full_cap']}")
        add(f"  Diseases receiving fewer than the cap: {summary['diseases_below_cap']}")
        add(f"  Diseases receiving all 3 query categories: {summary['diseases_all_three_categories']}")
        add(f"  Diseases receiving 2 query categories: {summary['diseases_two_categories']}")
        add(f"  Diseases receiving only 1 query category: {summary['diseases_one_category']}")
        add(f"  Diseases receiving no query categories: {summary['diseases_zero_categories']}")
        add(f"  Selected unique records with PMCID: {summary['pmcid']}")
        add(f"  Selected unique records with DOI: {summary['doi']}")
        add(f"  Selected unique records with neither PMCID nor DOI: {summary['neither']}")
    add("")

    add("SECTION 3 — QUERY CATEGORY COVERAGE")
    for cap in CAPS:
        summary = cap_summaries[cap]
        add(f"Cap {cap}:")
        for category in QUERY_CATEGORIES:
            add(
                f"  {category}: {summary['assignment_category_counts'][category]} assignments; "
                f"{len(summary['unique_category_pmids'][category])} unique PMIDs"
            )
    add("")

    add("SECTION 4 — PUBLICATION TYPE DIVERSITY")
    for cap in CAPS:
        summary = cap_summaries[cap]
        add(f"Cap {cap} (selected unique PMIDs):")
        for flag, _ in EVIDENCE_FLAGS:
            add(f"  {flag}: {summary['evidence_counts'][flag]}")
    add("")

    add("SECTION 5 — FULL-TEXT SIGNALS")
    for cap in CAPS:
        summary = cap_summaries[cap]
        unique_count = summary["unique_pmids"]
        add(
            f"Cap {cap}: PMCID={summary['pmcid']}, no PMCID={unique_count - summary['pmcid']}, "
            f"DOI={summary['doi']}, no DOI={unique_count - summary['doi']}, "
            f"both={summary['both']}, neither={summary['neither']}"
        )
    add("")

    add("SECTION 6 — RECENCY")
    year_bucket_order = (
        "before 2000",
        "2000-2009",
        "2010-2014",
        "2015-2019",
        "2020-2024",
        "2025",
        "2026",
        "2027+",
        "missing/invalid",
    )
    for cap in CAPS:
        years = cap_summaries[cap]["years"]
        add(f"Cap {cap} (selected unique PMIDs):")
        for bucket in year_bucket_order:
            add(f"  {bucket}: {years[bucket]}")
    add("Publication year is reported as a descriptive attribute, not a quality verdict.")
    add("")

    add("SECTION 7 — LOW-COVERAGE DISEASES")
    for cap in CAPS:
        summary = cap_summaries[cap]
        add(f"Cap {cap}; diseases receiving fewer than {cap} assignments:")
        add("disease_id | disease_name | available_candidates | selected")
        low_coverage = sorted(
            [
                (
                    disease_id,
                    len(candidates_by_disease[disease_id]),
                    len(summary["selection"][disease_id]),
                )
                for disease_id in diseases
                if len(summary["selection"][disease_id]) < cap
            ],
            key=lambda item: (item[2], disease_sort_key(item[0])),
        )
        for disease_id, available, selected in low_coverage:
            add(f"{disease_id} | {diseases[disease_id]} | {available} | {selected}")
    add("")

    add("SECTION 8 — ZERO-CANDIDATE DISEASES")
    add(f"Count: {len(zero_diseases)}")
    add("disease_id | disease_name")
    for disease_id in sorted(zero_diseases, key=disease_sort_key):
        add(f"{disease_id} | {diseases[disease_id]}")
    add("")

    add("SECTION 9 — REPRESENTATIVE DISEASE EXAMPLES")
    for cap in CAPS:
        summary = cap_summaries[cap]
        high_availability = sorted(
            diseases,
            key=lambda disease_id: (-len(candidates_by_disease[disease_id]), disease_sort_key(disease_id)),
        )[:5]
        low_availability = sorted(
            diseases,
            key=lambda disease_id: (len(candidates_by_disease[disease_id]), disease_sort_key(disease_id)),
        )[:5]
        add(f"Cap {cap}:")
        add("High candidate availability:")
        add("disease_id | disease_name | available_candidates | selected | selected_query_categories | selected_pmcid_count | selected_doi_count")
        for disease_id in high_availability:
            rows = summary["selection"][disease_id]
            categories = sorted(set().union(*(row_categories(row) for row in rows))) if rows else []
            pmcid_count = sum(has_value(row, "pmcid") for row in rows)
            doi_count = sum(has_value(row, "doi") for row in rows)
            add(
                f"{disease_id} | {diseases[disease_id]} | {len(candidates_by_disease[disease_id])} | "
                f"{len(rows)} | {format_categories(categories)} | {pmcid_count} | {doi_count}"
            )
        add("Low candidate availability:")
        add("disease_id | disease_name | available_candidates | selected | selected_query_categories | selected_pmcid_count | selected_doi_count")
        for disease_id in low_availability:
            rows = summary["selection"][disease_id]
            categories = sorted(set().union(*(row_categories(row) for row in rows))) if rows else []
            pmcid_count = sum(has_value(row, "pmcid") for row in rows)
            doi_count = sum(has_value(row, "doi") for row in rows)
            add(
                f"{disease_id} | {diseases[disease_id]} | {len(candidates_by_disease[disease_id])} | "
                f"{len(rows)} | {format_categories(categories)} | {pmcid_count} | {doi_count}"
            )
    add("")

    add("SECTION 10 — COMPARISON TABLE")
    add("Cap | Unique PMIDs | Disease assignments | Diseases covered | All 3 categories | PMCID records | DOI records")
    for cap in CAPS:
        summary = cap_summaries[cap]
        add(
            f"{cap} | {summary['unique_pmids']} | {summary['selected_record_disease_assignments']} | "
            f"{summary['diseases_covered']} | {summary['diseases_all_three_categories']} | "
            f"{summary['pmcid']} | {summary['doi']}"
        )
    add("")

    add("SECTION 11 — INTERPRETATION DATA ONLY")
    for cap in CAPS:
        summary = cap_summaries[cap]
        add(
            f"Cap {cap}: {summary['diseases_below_cap']} diseases cannot fill the cap; "
            f"{summary['diseases_all_three_categories']} receive all three categories; "
            f"{summary['pmcid']} selected unique PMIDs have a PMCID; "
            f"{summary['unique_pmids']} unique PMIDs are selected."
        )
    add("One selected PMID may contribute to multiple disease assignments.")
    add("Signals are descriptive; no cap is ranked or recommended.")

    return "\n".join(lines) + "\n", stage3_count


def format_categories(categories):
    return ", ".join(category for category in QUERY_CATEGORIES if category in categories) or "(none)"


def main():
    if REPORT_FILE.exists():
        print(f"ERROR: Refusing to overwrite existing report: {REPORT_FILE}", file=sys.stderr)
        return 1
    try:
        metadata_rows = read_csv(METADATA_FILE, REQUIRED_METADATA_COLUMNS)
        manifest_rows = read_csv(MANIFEST_FILE, ("disease_id", "disease_name"))
        diseases = load_manifest(manifest_rows)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"ERROR: Could not load simulation inputs: {error}", file=sys.stderr)
        return 1

    stage3_rows = [row for row in metadata_rows if stage3_eligible(row)]
    records_by_pmid = {}
    duplicate_pmids = 0
    for row in stage3_rows:
        pmid = (row.get("pmid") or "").strip()
        if not pmid:
            continue
        if pmid in records_by_pmid:
            duplicate_pmids += 1
            continue
        records_by_pmid[pmid] = row

    candidates_by_disease = {disease_id: [] for disease_id in diseases}
    unknown_disease_ids = set()
    for row in records_by_pmid.values():
        for disease_id in split_values(row.get("disease_ids")):
            if disease_id not in candidates_by_disease:
                unknown_disease_ids.add(disease_id)
                continue
            candidates_by_disease[disease_id].append(row)
    for rows in candidates_by_disease.values():
        rows.sort(key=lambda row: (row.get("pmid") or "").strip())

    try:
        report, simulated_stage3_count = format_report(
            metadata_rows,
            diseases,
            candidates_by_disease,
        )
    except (ValueError, KeyError) as error:
        print(f"ERROR: Could not simulate candidate selection: {error}", file=sys.stderr)
        return 1

    if simulated_stage3_count != len(records_by_pmid) or duplicate_pmids:
        print(
            f"WARNING: Stage 3 eligible rows={simulated_stage3_count}, "
            f"unique non-empty PMIDs={len(records_by_pmid)}, duplicate rows={duplicate_pmids}.",
            file=sys.stderr,
        )
    if unknown_disease_ids:
        print(
            f"WARNING: {len(unknown_disease_ids)} disease ID(s) were absent from the manifest.",
            file=sys.stderr,
        )

    try:
        with REPORT_FILE.open("x", encoding="utf-8", newline="") as report_file:
            report_file.write(report)
    except OSError as error:
        print(f"ERROR: Could not write report without overwriting: {error}", file=sys.stderr)
        return 1

    print("SIMULATION COMPLETE")
    print(f"Records inspected: {len(metadata_rows)}")
    print(f"Report: {REPORT_FILE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())