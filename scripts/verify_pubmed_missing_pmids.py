import csv
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAILURE_INPUT = PROJECT_ROOT / "data" / "processed" / "pubmed_metadata_failures.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_missing_pmid_verification.csv"
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
TOOL_NAME = "symptom_rag_analyser"

EXPECTED_PMID_COUNT = 1170
BATCH_SIZE = 100
REQUEST_DELAY = 0.5
HTTP_TIMEOUT = 30
MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 1.0
OUTPUT_COLUMNS = ("pmid", "pubmed_exists")


def read_input_pmids():
    with FAILURE_INPUT.open("r", encoding="utf-8-sig", newline="") as failure_file:
        reader = csv.DictReader(failure_file)
        if "pmids" not in (reader.fieldnames or []):
            raise ValueError("Failure log is missing the required 'pmids' column.")

        pmids = []
        seen = set()
        for row in reader:
            for value in (row.get("pmids") or "").split(","):
                pmid = value.strip()
                if pmid and pmid not in seen:
                    if not pmid.isdigit():
                        raise ValueError(f"Invalid non-numeric PMID value: {pmid!r}")
                    seen.add(pmid)
                    pmids.append(pmid)

    if len(pmids) != EXPECTED_PMID_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PMID_COUNT} unique non-empty PMIDs; found {len(pmids)}."
        )
    return pmids


def search_batch(pmids, email, api_key):
    term = " OR ".join(f"{pmid}[PMID]" for pmid in pmids)
    parameters = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": str(len(pmids)),
        "tool": TOOL_NAME,
        "email": email,
    }
    if api_key:
        parameters["api_key"] = api_key

    url = f"{ESEARCH_URL}?{urllib.parse.urlencode(parameters)}"
    request = urllib.request.Request(url, headers={"User-Agent": TOOL_NAME})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        payload = json.loads(response.read().decode("utf-8"))

    result = payload.get("esearchresult")
    if not isinstance(result, dict) or not isinstance(result.get("idlist"), list):
        raise ValueError("ESearch response is missing esearchresult.idlist.")

    requested = set(pmids)
    found = {str(pmid) for pmid in result["idlist"]}
    unexpected = found - requested
    if unexpected:
        raise ValueError("ESearch returned PMID(s) that were not requested.")
    return found


def is_retryable(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 429) or 500 <= error.code < 600
    return isinstance(
        error,
        (urllib.error.URLError, TimeoutError, json.JSONDecodeError),
    )


def search_with_retries(pmids, email, api_key):
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return search_batch(pmids, email, api_key), attempt, None
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt >= MAX_ATTEMPTS or not is_retryable(error):
                break
            delay = max(REQUEST_DELAY, BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
            print(
                f"  Attempt {attempt}/{MAX_ATTEMPTS} failed ({type(error).__name__}); "
                f"retrying in {delay:.1f}s."
            )
            time.sleep(delay)
        except Exception as error:
            last_error = error
            break
    return set(), attempt, last_error


def write_results(pmids, found_pmids):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing verification output: {OUTPUT_FILE}"
        )
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=OUTPUT_FILE.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.DictWriter(temporary_file, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for pmid in pmids:
                writer.writerow({
                    "pmid": pmid,
                    "pubmed_exists": "true" if pmid in found_pmids else "false",
                })
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(OUTPUT_FILE)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    email = os.environ.get("NCBI_EMAIL", "").strip()
    if not email:
        print(
            "ERROR: NCBI_EMAIL is required by NCBI E-utilities usage guidance. "
            "Set NCBI_EMAIL in the environment or project .env file.",
            file=sys.stderr,
        )
        return 1
    api_key = os.environ.get("NCBI_API_KEY", "").strip()

    try:
        pmids = read_input_pmids()
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if OUTPUT_FILE.exists():
        print(
            f"ERROR: Refusing to overwrite existing verification output: {OUTPUT_FILE}",
            file=sys.stderr,
        )
        return 1

    batches = [pmids[index:index + BATCH_SIZE] for index in range(0, len(pmids), BATCH_SIZE)]
    found_pmids = set()
    for batch_number, batch in enumerate(batches, start=1):
        print(f"Checking batch {batch_number}/{len(batches)} ({len(batch)} PMIDs)")
        batch_found, attempts, error = search_with_retries(batch, email, api_key)
        if error is not None:
            print(
                f"ERROR: ESearch batch {batch_number} failed after {attempts} attempt(s) "
                f"({type(error).__name__}). No verification output was written.",
                file=sys.stderr,
            )
            return 1

        found_pmids.update(batch_found)
        if batch_number < len(batches):
            time.sleep(REQUEST_DELAY)

    try:
        write_results(pmids, found_pmids)
    except OSError as error:
        print(f"ERROR: Could not safely write verification output: {error}", file=sys.stderr)
        return 1
    existing_count = len(found_pmids)
    missing_count = len(pmids) - existing_count

    print("\n" + "=" * 72)
    print("PUBMED MISSING PMID VERIFICATION")
    print("=" * 72)
    print(f"Input PMIDs: {len(pmids)}")
    print(f"PMIDs verified as existing: {existing_count}")
    print(f"PMIDs not found: {missing_count}")
    print(f"Verification coverage: {len(pmids)}/{len(pmids)} (100.0%)")
    print(f"\nOutput: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())