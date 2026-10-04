import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_FILE = DATA_DIR / "disease_literature_manifest.csv"
METADATA_FILE = DATA_DIR / "pubmed_complete_metadata.csv"
OUTPUT_FILE = DATA_DIR / "pubmed_zero_candidate_recovery.csv"
SUMMARY_FILE = DATA_DIR / "pubmed_zero_candidate_recovery_summary.txt"
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
MESH_LOOKUP_URL = "https://id.nlm.nih.gov/mesh/lookup/term"
TOOL_NAME = "symptom_rag_analyser"

EXPECTED_MANIFEST_DISEASES = 866
RETMAX = 20
REQUEST_DELAY = 0.5
HTTP_TIMEOUT = 30
MESH_TIMEOUT = 20
MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 1.0
RESULT_COLUMNS = (
    "disease_id",
    "disease_name",
    "search_variant",
    "query",
    "result_count",
    "retrieved_pmids",
    "error_type",
    "error_message",
)
EVIDENCE_QUERY = (
    "(diagnos*[Title/Abstract] OR symptom*[Title/Abstract] OR "
    '"clinical presentation"[Title/Abstract] OR "clinical features"[Title/Abstract])'
)
LAST_REQUEST_TIME = 0.0


def read_csv(path, required_columns):
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


def read_zero_candidate_diseases():
    manifest = read_csv(MANIFEST_FILE, ("disease_id", "disease_name"))
    if len(manifest) != EXPECTED_MANIFEST_DISEASES:
        raise ValueError(
            f"Expected {EXPECTED_MANIFEST_DISEASES} manifest rows; found {len(manifest)}."
        )

    metadata = read_csv(METADATA_FILE, ("disease_ids",))
    covered_ids = set()
    for row in metadata:
        covered_ids.update(split_values(row.get("disease_ids")))

    diseases = []
    seen_ids = set()
    for row_number, row in enumerate(manifest, start=2):
        disease_id = (row.get("disease_id") or "").strip()
        disease_name = (row.get("disease_name") or "").strip()
        if not disease_id or not disease_name:
            raise ValueError(f"Manifest row {row_number} has an empty disease ID or name.")
        if disease_id in seen_ids:
            raise ValueError(f"Duplicate disease_id in manifest: {disease_id}")
        seen_ids.add(disease_id)
        if disease_id not in covered_ids:
            diseases.append({"disease_id": disease_id, "disease_name": disease_name})
    return diseases


def normalize_punctuation(value):
    return " ".join(re.sub(r"[^\w\s]", " ", value).split())


def apostrophe_variant(value):
    if "'" in value or "’" in value or "‘" in value:
        return value.replace("’", "'").replace("‘", "'").replace("'", "")

    match = re.match(
        r"^([A-Za-z]{4,})s(\s+)((?:Disease|Syndrome|Foot)\b.*)$",
        value,
        re.IGNORECASE,
    )
    if match:
        return f"{match.group(1)}'s{match.group(2)}{match.group(3)}"
    return ""


def parentheses_removed(value):
    without_groups = re.sub(r"\([^)]*\)", " ", value)
    without_groups = re.sub(r"[()]+", " ", without_groups)
    return " ".join(without_groups.split())


def base_variants(disease_name):
    variants = []
    seen = set()

    def add(label, term):
        term = " ".join(term.split()).strip()
        key = term.casefold()
        if term and key not in seen:
            seen.add(key)
            variants.append((label, term))

    add("original", disease_name)
    add("normalized", normalize_punctuation(disease_name))
    add("apostrophe_normalized", apostrophe_variant(disease_name))
    add("parentheses_removed", parentheses_removed(disease_name))
    return variants


def wait_for_request_interval():
    global LAST_REQUEST_TIME
    elapsed = time.monotonic() - LAST_REQUEST_TIME
    if LAST_REQUEST_TIME and elapsed < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - elapsed)
    LAST_REQUEST_TIME = time.monotonic()


def is_retryable(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 429) or 500 <= error.code < 600
    return isinstance(error, (urllib.error.URLError, TimeoutError, OSError))


def safe_error_message(error):
    if isinstance(error, urllib.error.HTTPError):
        return f"HTTP {error.code}" + (f": {error.reason}" if error.reason else "")
    if isinstance(error, urllib.error.URLError):
        return str(error.reason)
    return str(error)


def request_json(endpoint, parameters, headers, timeout):
    global LAST_REQUEST_TIME
    request = urllib.request.Request(
        f"{endpoint}?{urllib.parse.urlencode(parameters)}",
        headers=headers,
    )
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        wait_for_request_interval()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            LAST_REQUEST_TIME = time.monotonic()
            return payload, attempt, None
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as error:
            LAST_REQUEST_TIME = time.monotonic()
            last_error = error
            if attempt >= MAX_ATTEMPTS or not is_retryable(error):
                break
            delay = max(REQUEST_DELAY, BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
            print(f"  Request retry {attempt}/{MAX_ATTEMPTS} after {type(error).__name__}; waiting {delay:.1f}s.")
            time.sleep(delay)
    return None, attempt, last_error


def lookup_mesh_canonical(disease_name):
    payload, _, error = request_json(
        MESH_LOOKUP_URL,
        {"label": disease_name},
        {"Accept": "application/json", "User-Agent": TOOL_NAME},
        MESH_TIMEOUT,
    )
    if error is not None or not isinstance(payload, list):
        return ""
    for item in payload:
        if isinstance(item, dict) and item.get("label"):
            return " ".join(str(item["label"]).split())
    return ""


def build_query(search_term):
    escaped_term = search_term.replace('"', '\\"')
    return f'"{escaped_term}"[Title/Abstract] AND {EVIDENCE_QUERY}'


def esearch(query, email, api_key):
    parameters = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(RETMAX),
        "tool": TOOL_NAME,
        "email": email,
    }
    if api_key:
        parameters["api_key"] = api_key

    payload, attempts, error = request_json(
        ESEARCH_URL,
        parameters,
        {"User-Agent": TOOL_NAME},
        HTTP_TIMEOUT,
    )
    if error is not None:
        return None, [], attempts, error
    try:
        result = payload["esearchresult"]
        result_count = int(result["count"])
        pmids = list(dict.fromkeys(str(pmid) for pmid in result.get("idlist", [])))
        return result_count, pmids, attempts, None
    except (KeyError, TypeError, ValueError) as parse_error:
        return None, [], attempts, parse_error


def attempt_row(disease, variant, query, result_count, pmids, error=None):
    return {
        "disease_id": disease["disease_id"],
        "disease_name": disease["disease_name"],
        "search_variant": variant,
        "query": query,
        "result_count": "" if result_count is None else result_count,
        "retrieved_pmids": "|".join(pmids),
        "error_type": type(error).__name__ if error else "",
        "error_message": safe_error_message(error) if error else "",
    }


def write_csv(records):
    with OUTPUT_FILE.open("x", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
        output_file.flush()
        os.fsync(output_file.fileno())


def build_summary(zero_diseases, disease_results, all_attempts):
    recovered = [
        disease for disease in zero_diseases
        if disease["disease_id"] in disease_results
    ]
    unresolved = [
        disease for disease in zero_diseases
        if disease["disease_id"] not in disease_results
    ]
    unique_pmids = {
        pmid
        for result in disease_results.values()
        for pmid in result["retrieved_pmids"]
    }

    lines = [
        "PUBMED ZERO-CANDIDATE COVERAGE RECOVERY SUMMARY",
        "=" * 72,
        f"Original zero-candidate diseases: {len(zero_diseases)}",
        f"Diseases recovered: {len(recovered)}",
        f"Diseases still unresolved: {len(unresolved)}",
        f"Unique PMIDs recovered: {len(unique_pmids)}",
        "",
        "Candidate count per recovered disease:",
    ]
    if recovered:
        for disease in recovered:
            result = disease_results[disease["disease_id"]]
            lines.append(
                f"  {disease['disease_id']} | {disease['disease_name']} | "
                f"{len(result['retrieved_pmids'])} retrieved PMID(s) "
                f"(ESearch count={result['result_count']})"
            )
    else:
        lines.append("  (none)")

    lines.extend(("", "Diseases still unresolved:"))
    if unresolved:
        for disease in unresolved:
            lines.append(f"  {disease['disease_id']} | {disease['disease_name']}")
    else:
        lines.append("  (none)")
    lines.extend(("", f"Search attempts recorded: {len(all_attempts)}"))
    return "\n".join(lines) + "\n"


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    email = os.environ.get("NCBI_EMAIL", "").strip()
    if not email:
        print(
            "ERROR: NCBI_EMAIL is required by NCBI E-utilities usage guidance. "
            "Set NCBI_EMAIL in the project .env file or environment.",
            file=sys.stderr,
        )
        return 1
    api_key = os.environ.get("NCBI_API_KEY", "").strip()

    if OUTPUT_FILE.exists() or SUMMARY_FILE.exists():
        print(
            "ERROR: Refusing to overwrite an existing recovery CSV or summary file.",
            file=sys.stderr,
        )
        return 1

    try:
        zero_diseases = read_zero_candidate_diseases()
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"ERROR: Could not identify zero-candidate diseases: {error}", file=sys.stderr)
        return 1

    attempts = []
    disease_results = {}
    for disease_index, disease in enumerate(zero_diseases, start=1):
        print(
            f"[{disease_index}/{len(zero_diseases)}] "
            f"{disease['disease_id']} | {disease['disease_name']}"
        )
        variants = base_variants(disease["disease_name"])
        seen_terms = {term.casefold() for _, term in variants}
        tried_mesh = False
        stop_disease = False

        variant_index = 0
        while variant_index < len(variants) or not tried_mesh:
            if variant_index >= len(variants):
                tried_mesh = True
                mesh_term = lookup_mesh_canonical(disease["disease_name"])
                if mesh_term and mesh_term.casefold() not in seen_terms:
                    variants.append(("mesh_canonical", mesh_term))
                else:
                    break

            variant, term = variants[variant_index]
            variant_index += 1
            query = build_query(term)
            result_count, retrieved_pmids, _, error = esearch(query, email, api_key)
            row = attempt_row(disease, variant, query, result_count, retrieved_pmids, error)
            attempts.append(row)

            if error is not None:
                print(f"  {variant}: ESearch failed ({type(error).__name__}); moving to next disease.")
                stop_disease = True
                break
            print(f"  {variant}: count={result_count}, retrieved={len(retrieved_pmids)}")
            if retrieved_pmids:
                disease_results[disease["disease_id"]] = {
                    "result_count": result_count,
                    "retrieved_pmids": retrieved_pmids,
                    "search_variant": variant,
                }
                break

            if variant_index == len(variants) and not tried_mesh:
                tried_mesh = True
                mesh_term = lookup_mesh_canonical(disease["disease_name"])
                if mesh_term and mesh_term.casefold() not in seen_terms:
                    variants.append(("mesh_canonical", mesh_term))
                    seen_terms.add(mesh_term.casefold())

        if stop_disease:
            continue

    try:
        write_csv(attempts)
        summary = build_summary(zero_diseases, disease_results, attempts)
        with SUMMARY_FILE.open("x", encoding="utf-8", newline="") as summary_file:
            summary_file.write(summary)
    except OSError as error:
        print(f"ERROR: Could not save recovery results: {error}", file=sys.stderr)
        return 1

    print("\n" + summary, end="")
    print(f"\nResults: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    print(f"Summary: {SUMMARY_FILE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())