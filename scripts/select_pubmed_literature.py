import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_FILE = DATA_DIR / "pubmed_complete_metadata.csv"
DISEASE_MANIFEST_FILE = DATA_DIR / "disease_literature_manifest.csv"
SELECTED_FILE = DATA_DIR / "pubmed_selected_literature.csv"
AUDIT_FILE = DATA_DIR / "pubmed_selection_audit.csv"
SUMMARY_FILE = DATA_DIR / "pubmed_selection_summary.txt"

SELECTION_CAP = 5
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
EVIDENCE_FLAG_ALIASES = {
    "journal_article": {"journal_article"},
    "review": {"review"},
    "systematic_review": {"systematic_review"},
    "meta_analysis": {"meta_analysis", "meta_analyses"},
    "case_report": {"case_report", "case_reports"},
    "practice_guideline": {"practice_guideline", "practice_guidelines"},
    "consensus_statement": {"consensus_statement", "consensus_statements"},
    "randomized_controlled_trial": {
        "randomized_controlled_trial",
        "randomised_controlled_trial",
    },
    "clinical_trial": {"clinical_trial", "clinical_trials"},
    "observational_study": {"observational_study", "observational_studies"},
    "book_article": {"book_article"},
}
YEAR_BUCKETS = (
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
SOURCE_COLUMNS = (
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
SELECTED_COLUMNS = SOURCE_COLUMNS + (
    "selected_disease_ids",
    "selected_disease_names",
    "selected_selection_ranks",
    "selected_query_categories",
    "selected_selection_reasons",
)
AUDIT_COLUMNS = (
    "disease_id",
    "disease_name",
    "pmid",
    "title",
    "selection_rank",
    "query_categories",
    "publication_types",
    "publication_year",
    "has_pmc",
    "doi",
    "record_type",
    "selection_reason",
)


def read_csv(path, required_columns):
    if not path.is_file():
        raise FileNotFoundError(f"Required input file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = set(required_columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path.name} is missing required columns: " + ", ".join(sorted(missing))
            )
        return list(reader)


def split_values(value):
    return {part.strip() for part in (value or "").split("|") if part.strip()}


def has_value(row, field):
    return bool((row.get(field) or "").strip())


def normalize_type(value):
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def publication_types(row):
    return {value.casefold() for value in split_values(row.get("publication_types"))}


def evidence_flags(row):
    normalized = {normalize_type(value) for value in split_values(row.get("publication_types"))}
    flags = {
        flag
        for flag, aliases in EVIDENCE_FLAG_ALIASES.items()
        if normalized & aliases
    }
    if (row.get("record_type") or "").strip().casefold() == "pubmed_book_article":
        flags.add("book_article")
    return flags


def row_categories(row):
    available = {value.casefold() for value in split_values(row.get("query_categories"))}
    return {category for category in QUERY_CATEGORIES if category in available}


def parse_publication_year(row):
    for value in (row.get("publication_year"), row.get("publication_date")):
        if value:
            match = re.search(r"\b(\d{4})\b", value)
            if match:
                return int(match.group(1))
    return None


def year_sort_key(row):
    year = parse_publication_year(row)
    return year if year is not None else -1


def pmid_sort_key(pmid):
    try:
        return 0, int(pmid)
    except ValueError:
        return 1, pmid


def disease_sort_key(disease_id):
    try:
        return 0, int(disease_id)
    except ValueError:
        return 1, disease_id.casefold()


def stage3_eligible(row):
    languages = {value.casefold() for value in split_values(row.get("language"))}
    if "eng" not in languages or not has_value(row, "abstract"):
        return False
    return not bool(publication_types(row) & EXCLUDED_PUBLICATION_TYPES)


def load_diseases(rows):
    diseases = {}
    for row_number, row in enumerate(rows, start=2):
        disease_id = (row.get("disease_id") or "").strip()
        disease_name = (row.get("disease_name") or "").strip()
        if not disease_id or not disease_name:
            raise ValueError(f"Disease manifest row {row_number} has an empty ID or name.")
        if disease_id in diseases:
            raise ValueError(f"Duplicate disease_id in disease manifest: {disease_id}")
        diseases[disease_id] = disease_name
    return diseases


def preference_select(candidates, primary_key, primary_label):
    if not candidates:
        raise ValueError("Cannot select from an empty candidate list.")

    reason_components = []
    best_primary = max(primary_key(row) for row in candidates)
    finalists = [row for row in candidates if primary_key(row) == best_primary]
    if primary_label and best_primary:
        reason_components.append(primary_label)

    pmcid_candidates = [row for row in finalists if has_value(row, "pmcid")]
    if pmcid_candidates:
        if len(pmcid_candidates) < len(finalists):
            reason_components.append("pmcid_available")
        finalists = pmcid_candidates

    newest_year = max(year_sort_key(row) for row in finalists)
    newest_candidates = [row for row in finalists if year_sort_key(row) == newest_year]
    if newest_year >= 0 and len(newest_candidates) < len(finalists):
        reason_components.append("recency")
    finalists = newest_candidates

    if len(finalists) > 1:
        reason_components.append("deterministic_tiebreak")
    selected = min(finalists, key=lambda row: pmid_sort_key(row["pmid"].strip()))
    if not reason_components:
        reason_components.append("deterministic_tiebreak")
    return selected, reason_components


def select_for_disease(disease_id, candidates):
    remaining = {row["pmid"].strip(): row for row in candidates}
    selected = []
    represented_categories = set()
    available_categories = set().union(*(row_categories(row) for row in remaining.values())) if remaining else set()

    while len(selected) < SELECTION_CAP and available_categories - represented_categories:
        uncovered = available_categories - represented_categories
        target_category = next(
            category for category in QUERY_CATEGORIES if category in uncovered
        )
        category_candidates = [
            row for row in remaining.values()
            if target_category in row_categories(row)
        ]
        chosen, reasons = preference_select(
            category_candidates,
            lambda row: len(row_categories(row) - represented_categories),
            "category_coverage",
        )
        selected.append({
            "disease_id": disease_id,
            "row": chosen,
            "selection_reason": "; ".join(reasons),
        })
        represented_categories.update(row_categories(chosen))
        remaining.pop(chosen["pmid"].strip())

    selected_evidence = set().union(*(evidence_flags(item["row"]) for item in selected)) if selected else set()
    while remaining and len(selected) < SELECTION_CAP:
        chosen, reasons = preference_select(
            list(remaining.values()),
            lambda row: len(evidence_flags(row) - selected_evidence),
            "evidence_type_diversity",
        )
        selected.append({
            "disease_id": disease_id,
            "row": chosen,
            "selection_reason": "; ".join(reasons),
        })
        selected_evidence.update(evidence_flags(chosen))
        remaining.pop(chosen["pmid"].strip())

    for rank, selection in enumerate(selected, start=1):
        selection["selection_rank"] = rank
    return selected


def category_string(row):
    return "+".join(category for category in QUERY_CATEGORIES if category in row_categories(row))


def build_outputs(metadata_rows, diseases):
    source_by_pmid = {}
    for row_number, source_row in enumerate(metadata_rows, start=2):
        pmid = (source_row.get("pmid") or "").strip()
        if not pmid:
            raise ValueError(f"Metadata input has an empty PMID at row {row_number}.")
        if pmid in source_by_pmid:
            raise ValueError(f"Duplicate PMID in metadata input: {pmid}")
        source_by_pmid[pmid] = {
            column: (source_row.get(column) or "")
            for column in SOURCE_COLUMNS
        }
        source_by_pmid[pmid]["pmid"] = pmid

    eligible_by_disease = {disease_id: [] for disease_id in diseases}
    for row in source_by_pmid.values():
        if not stage3_eligible(row):
            continue
        for disease_id in split_values(row.get("disease_ids")):
            if disease_id not in diseases:
                raise ValueError(f"Metadata references disease_id absent from manifest: {disease_id}")
            eligible_by_disease[disease_id].append(row)

    selections = []
    for disease_id in sorted(diseases, key=disease_sort_key):
        candidates = sorted(
            eligible_by_disease[disease_id],
            key=lambda row: pmid_sort_key(row["pmid"]),
        )
        selections.extend(select_for_disease(disease_id, candidates))

    selected_by_pmid = defaultdict(list)
    audit_rows = []
    for selection in selections:
        disease_id = selection["disease_id"]
        row = selection["row"]
        disease_name = diseases[disease_id]
        categories = category_string(row)
        selected_by_pmid[row["pmid"]].append({
            "disease_id": disease_id,
            "disease_name": disease_name,
            "selection_rank": selection["selection_rank"],
            "query_categories": categories,
            "selection_reason": selection["selection_reason"],
        })
        audit_rows.append({
            "disease_id": disease_id,
            "disease_name": disease_name,
            "pmid": row["pmid"],
            "title": row.get("title", ""),
            "selection_rank": selection["selection_rank"],
            "query_categories": categories,
            "publication_types": row.get("publication_types", ""),
            "publication_year": row.get("publication_year", ""),
            "has_pmc": row.get("has_pmc", ""),
            "doi": row.get("doi", ""),
            "record_type": row.get("record_type", ""),
            "selection_reason": selection["selection_reason"],
        })

    selected_rows = []
    for pmid in sorted(selected_by_pmid, key=pmid_sort_key):
        source = source_by_pmid[pmid]
        relationships = sorted(
            selected_by_pmid[pmid],
            key=lambda item: (disease_sort_key(item["disease_id"]), item["selection_rank"]),
        )
        selected_record = dict(source)
        selected_record.update({
            "selected_disease_ids": "||".join(item["disease_id"] for item in relationships),
            "selected_disease_names": "||".join(item["disease_name"] for item in relationships),
            "selected_selection_ranks": "||".join(str(item["selection_rank"]) for item in relationships),
            "selected_query_categories": "||".join(item["query_categories"] for item in relationships),
            "selected_selection_reasons": "||".join(item["selection_reason"] for item in relationships),
        })
        selected_rows.append(selected_record)

    audit_rows.sort(
        key=lambda row: (
            disease_sort_key(row["disease_id"]),
            int(row["selection_rank"]),
            pmid_sort_key(row["pmid"]),
        )
    )
    return source_by_pmid, eligible_by_disease, selections, selected_rows, audit_rows


def year_bucket(row):
    year = parse_publication_year(row)
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


def build_summary(metadata_rows, diseases, eligible_by_disease, selections, selected_rows):
    counts_by_disease = Counter(selection["disease_id"] for selection in selections)
    zero_diseases = [disease_id for disease_id in diseases if counts_by_disease[disease_id] == 0]
    unique_pmids = {row["pmid"] for row in selected_rows}
    stage3_count = sum(stage3_eligible(row) for row in metadata_rows)
    selected_assignments = len(selections)
    disease_categories = {
        disease_id: set().union(*(
            row_categories(selection["row"])
            for selection in selections
            if selection["disease_id"] == disease_id
        )) if counts_by_disease[disease_id] else set()
        for disease_id in diseases
    }
    category_receipt_counts = Counter(len(categories) for categories in disease_categories.values())
    assignment_category_counts = Counter()
    for selection in selections:
        assignment_category_counts.update(row_categories(selection["row"]))

    selected_records = {row["pmid"]: row for row in selected_rows}
    with_pmcid = sum(has_value(row, "pmcid") for row in selected_records.values())
    with_doi = sum(has_value(row, "doi") for row in selected_records.values())
    with_both = sum(has_value(row, "pmcid") and has_value(row, "doi") for row in selected_records.values())
    with_neither = sum(not has_value(row, "pmcid") and not has_value(row, "doi") for row in selected_records.values())
    evidence_counts = Counter()
    record_type_counts = Counter()
    for row in selected_records.values():
        evidence_counts.update(evidence_flags(row))
        record_type_counts[(row.get("record_type") or "").strip()] += 1
    year_counts = Counter(year_bucket(row) for row in selected_records.values())

    lines = []
    add = lines.append
    add("PUBMED LITERATURE SELECTION SUMMARY")
    add("=" * 80)
    add("")
    add("1. INPUT")
    add(f"Total metadata records: {len(metadata_rows)}")
    add(f"Stage 3 eligible records: {stage3_count}")
    add("")

    add("2. FINAL SELECTION")
    add(f"Unique selected PMIDs: {len(unique_pmids)}")
    add(f"Selected disease-article assignments: {selected_assignments}")
    add(f"Unique diseases represented: {sum(counts_by_disease[disease_id] > 0 for disease_id in diseases)}")
    for count in range(6):
        number = sum(counts_by_disease[disease_id] == count for disease_id in diseases)
        add(f"Diseases with {count} selected: {number}")
    add("")

    add("3. QUERY CATEGORY COVERAGE")
    for category in QUERY_CATEGORIES:
        add(f"Selected assignments containing {category}: {assignment_category_counts[category]}")
    add(f"Diseases receiving all 3 categories: {category_receipt_counts[3]}")
    add(f"Diseases receiving 2 categories: {category_receipt_counts[2]}")
    add(f"Diseases receiving 1 category: {category_receipt_counts[1]}")
    add(f"Diseases receiving 0 categories: {category_receipt_counts[0]}")
    add("")

    add("4. FULL-TEXT SIGNALS (UNIQUE SELECTED PMIDS)")
    add(f"With PMCID: {with_pmcid}")
    add(f"Without PMCID: {len(unique_pmids) - with_pmcid}")
    add(f"With DOI: {with_doi}")
    add(f"Without DOI: {len(unique_pmids) - with_doi}")
    add(f"Both: {with_both}")
    add(f"Neither: {with_neither}")
    add("")

    add("5. PUBLICATION TYPES (UNIQUE SELECTED PMIDS; CATEGORIES OVERLAP)")
    for flag in EVIDENCE_FLAG_ALIASES:
        add(f"{flag}: {evidence_counts[flag]}")
    add("")

    add("6. RECORD TYPES (UNIQUE SELECTED PMIDS)")
    add(f"pubmed_article: {record_type_counts['pubmed_article']}")
    add(f"pubmed_book_article: {record_type_counts['pubmed_book_article']}")
    add("")

    add("7. PUBLICATION YEARS (UNIQUE SELECTED PMIDS)")
    for bucket in YEAR_BUCKETS:
        add(f"{bucket}: {year_counts[bucket]}")
    add("")

    add("8. ZERO-CANDIDATE DISEASES")
    add(f"Count: {len(zero_diseases)}")
    add("disease_id | disease_name")
    for disease_id in sorted(zero_diseases, key=disease_sort_key):
        add(f"{disease_id} | {diseases[disease_id]}")
    add("")

    add("9. VALIDATION")
    add(f"Selected PMIDs unique in final CSV: {'PASS' if len(unique_pmids) == len(selected_rows) else 'FAIL'}")
    add("Audit disease_id + PMID pairs unique: PASS")
    add("Every selected PMID exists in input: PASS")
    add("No excluded publication type in final selection: PASS")
    add(f"No disease exceeds {SELECTION_CAP} selections: PASS")
    add("Every selected relationship appears in audit: PASS")
    add("Every selected PMID has at least one selected disease: PASS")
    add("Deterministic ordering: PASS")
    return "\n".join(lines) + "\n"


def validate_outputs(source_by_pmid, diseases, eligible_by_disease, selections, selected_rows, audit_rows):
    errors = []
    selected_pmids = [row["pmid"] for row in selected_rows]
    selected_pmid_set = set(selected_pmids)
    if len(selected_pmids) != len(selected_pmid_set):
        errors.append("selected literature contains duplicate PMIDs")
    if not selected_pmid_set <= set(source_by_pmid):
        errors.append("selected PMID is absent from metadata input")
    if any(not row.get("selected_disease_ids", "").strip() for row in selected_rows):
        errors.append("selected PMID has no selected disease association")
    if any(
        publication_types(row) & EXCLUDED_PUBLICATION_TYPES
        for row in selected_rows
    ):
        errors.append("selected output contains a Stage 3 excluded publication type")

    selection_pairs = [(item["disease_id"], item["row"]["pmid"]) for item in selections]
    audit_pairs = [(row["disease_id"], row["pmid"]) for row in audit_rows]
    if len(audit_pairs) != len(set(audit_pairs)):
        errors.append("audit contains duplicate disease_id + PMID relationships")
    if set(selection_pairs) != set(audit_pairs) or len(selection_pairs) != len(audit_pairs):
        errors.append("selected disease-article relationships do not match audit rows")

    count_by_disease = Counter(disease_id for disease_id, _ in selection_pairs)
    if any(count > SELECTION_CAP for count in count_by_disease.values()):
        errors.append("a disease exceeds the selection cap")
    expected_order = sorted(
        audit_rows,
        key=lambda row: (
            disease_sort_key(row["disease_id"]),
            int(row["selection_rank"]),
            pmid_sort_key(row["pmid"]),
        ),
    )
    if audit_rows != expected_order:
        errors.append("audit rows are not in deterministic order")
    expected_selected_order = sorted(selected_rows, key=lambda row: pmid_sort_key(row["pmid"]))
    if selected_rows != expected_selected_order:
        errors.append("selected literature rows are not in deterministic order")
    if any(disease_id not in diseases for disease_id in count_by_disease):
        errors.append("selection references a disease absent from the manifest")
    if any(len(eligible_by_disease[disease_id]) == 0 for disease_id in count_by_disease):
        errors.append("a disease without eligible candidates received a selection")

    if errors:
        raise ValueError("Selection validation failed: " + "; ".join(errors))


def write_csv(path, fieldnames, rows):
    with path.open("x", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        csv_file.flush()
        os.fsync(csv_file.fileno())


def write_outputs(selected_rows, audit_rows, summary):
    outputs = (SELECTED_FILE, AUDIT_FILE, SUMMARY_FILE)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in existing)
        )

    created = []
    try:
        write_csv(SELECTED_FILE, SELECTED_COLUMNS, selected_rows)
        created.append(SELECTED_FILE)
        write_csv(AUDIT_FILE, AUDIT_COLUMNS, audit_rows)
        created.append(AUDIT_FILE)
        with SUMMARY_FILE.open("x", encoding="utf-8", newline="") as summary_file:
            created.append(SUMMARY_FILE)
            summary_file.write(summary)
            summary_file.flush()
            os.fsync(summary_file.fileno())
    except OSError:
        for path in created:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def main():
    outputs = (SELECTED_FILE, AUDIT_FILE, SUMMARY_FILE)
    existing = [path for path in outputs if path.exists()]
    if existing:
        print(
            "ERROR: Refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in existing),
            file=sys.stderr,
        )
        return 1

    try:
        metadata_rows = read_csv(METADATA_FILE, SOURCE_COLUMNS)
        manifest_rows = read_csv(DISEASE_MANIFEST_FILE, ("disease_id", "disease_name"))
        diseases = load_diseases(manifest_rows)
        source_by_pmid, eligible_by_disease, selections, selected_rows, audit_rows = build_outputs(
            metadata_rows,
            diseases,
        )
        validate_outputs(
            source_by_pmid,
            diseases,
            eligible_by_disease,
            selections,
            selected_rows,
            audit_rows,
        )
        summary = build_summary(
            metadata_rows,
            diseases,
            eligible_by_disease,
            selections,
            selected_rows,
        )
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    try:
        write_outputs(selected_rows, audit_rows, summary)
    except OSError as error:
        print(f"ERROR: Could not safely write selection outputs: {error}", file=sys.stderr)
        return 1

    print("SELECTION COMPLETE")
    print(f"Unique selected PMIDs: {len(selected_rows)}")
    print(f"Disease-article assignments: {len(audit_rows)}")
    print(f"Zero-candidate diseases: {sum(not eligible_by_disease[disease_id] for disease_id in diseases)}")
    print(f"Output: {SELECTED_FILE.relative_to(PROJECT_ROOT)}")
    print(f"Audit: {AUDIT_FILE.relative_to(PROJECT_ROOT)}")
    print(f"Summary: {SUMMARY_FILE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())