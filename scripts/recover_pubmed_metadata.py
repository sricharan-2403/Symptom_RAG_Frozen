import csv
import importlib.util
import os
import sys
import tempfile
import time
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAILURE_INPUT = PROJECT_ROOT / "data" / "processed" / "pubmed_metadata_failures.csv"
METADATA_INPUT = PROJECT_ROOT / "data" / "processed" / "pubmed_article_metadata.csv"
CANDIDATE_INPUT = PROJECT_ROOT / "data" / "processed" / "pubmed_candidate_articles.csv"
RECOVERED_OUTPUT = PROJECT_ROOT / "data" / "processed" / "pubmed_metadata_recovered.csv"
RECOVERY_FAILURE_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "pubmed_metadata_recovery_failures.csv"
)
FETCH_SCRIPT = PROJECT_ROOT / "scripts" / "fetch_pubmed_metadata.py"

EXPECTED_MISSING_PMIDS = 1170
BATCH_SIZE = 50
REQUEST_DELAY = 0.5
MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 1.0
FAILURE_COLUMNS = (
    "batch_number",
    "pmids",
    "error_type",
    "error_message",
    "attempts",
)


def load_fetch_helpers():
    spec = importlib.util.spec_from_file_location("pubmed_metadata_fetcher", FETCH_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load metadata helpers from {FETCH_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_failure_pmids():
    with FAILURE_INPUT.open("r", encoding="utf-8-sig", newline="") as failure_file:
        reader = csv.DictReader(failure_file)
        if "pmids" not in (reader.fieldnames or []):
            raise ValueError("Failure log is missing the required 'pmids' column.")

        pmids = []
        for row in reader:
            pmids.extend(
                pmid.strip()
                for pmid in (row.get("pmids") or "").split(",")
                if pmid.strip()
            )

    duplicates = len(pmids) - len(set(pmids))
    if duplicates:
        raise ValueError(f"Failure log contains {duplicates} duplicate PMID entr(y/ies).")
    if len(pmids) != EXPECTED_MISSING_PMIDS:
        raise ValueError(
            f"Expected {EXPECTED_MISSING_PMIDS} unique non-empty missing PMIDs; "
            f"found {len(pmids)}. No API requests were made."
        )
    return pmids


def read_metadata_pmids(path, columns):
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    pmids = set()
    with path.open("r", encoding="utf-8-sig", newline="") as metadata_file:
        reader = csv.DictReader(metadata_file)
        if reader.fieldnames != list(columns):
            raise ValueError(f"Unexpected metadata schema in {path}.")
        for row in reader:
            pmid = (row.get("pmid") or "").strip()
            if pmid:
                pmids.add(pmid)
    return pmids


def load_recovered_records(columns):
    if not RECOVERED_OUTPUT.exists():
        return {}

    records = {}
    duplicate_found = False
    with RECOVERED_OUTPUT.open("r", encoding="utf-8-sig", newline="") as recovered_file:
        reader = csv.DictReader(recovered_file)
        if reader.fieldnames != list(columns):
            raise ValueError(f"Unexpected recovery output schema in {RECOVERED_OUTPUT}.")
        for raw_row in reader:
            row = {column: raw_row.get(column) or "" for column in columns}
            pmid = row["pmid"].strip()
            if not pmid:
                continue
            if pmid in records:
                duplicate_found = True
            else:
                records[pmid] = row

    if duplicate_found:
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=RECOVERED_OUTPUT.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                writer = csv.DictWriter(temporary_file, fieldnames=columns)
                writer.writeheader()
                writer.writerows(records.values())
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            temporary_path.replace(RECOVERED_OUTPUT)
            print("WARNING: Removed duplicate PMID rows from the recovery output.")
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
    return records


def append_records(records, columns):
    if not records:
        return
    is_new_file = not RECOVERED_OUTPUT.exists() or RECOVERED_OUTPUT.stat().st_size == 0
    with RECOVERED_OUTPUT.open("a", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        if is_new_file:
            writer.writeheader()
        writer.writerows(records)
        output_file.flush()
        os.fsync(output_file.fileno())


def reset_failure_log():
    with RECOVERY_FAILURE_OUTPUT.open("w", encoding="utf-8", newline="") as failure_file:
        csv.DictWriter(failure_file, fieldnames=FAILURE_COLUMNS).writeheader()


def append_failure(batch_number, pmids, error, attempts):
    with RECOVERY_FAILURE_OUTPUT.open("a", encoding="utf-8", newline="") as failure_file:
        writer = csv.DictWriter(failure_file, fieldnames=FAILURE_COLUMNS)
        writer.writerow({
            "batch_number": batch_number,
            "pmids": ",".join(pmids),
            "error_type": type(error).__name__,
            "error_message": str(error).replace("\r", " ").replace("\n", " "),
            "attempts": attempts,
        })
        failure_file.flush()
        os.fsync(failure_file.fileno())


def fetch_with_retries(fetcher, pmids, email, api_key):
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fetcher.fetch_batch(pmids, email, api_key), attempt, None
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ET.ParseError) as error:
            last_error = error
            if attempt >= MAX_ATTEMPTS or not fetcher.is_retryable(error):
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
    return {}, attempt, last_error


def validate_records(records, requested_pmids, existing_pmids, candidate_rows, columns):
    validated = []
    seen = set(existing_pmids)
    requested = set(requested_pmids)
    for pmid, metadata in records.items():
        if pmid not in requested:
            raise ValueError(f"EFetch returned unrequested PMID {pmid}.")
        if pmid in seen:
            continue
        candidate = candidate_rows.get(pmid)
        if candidate is None:
            raise ValueError(f"Candidate provenance is missing for PMID {pmid}.")
        record = dict(metadata)
        record.update({
            column: candidate.get(column, "")
            for column in columns
            if column != "pmid" and column in candidate
        })
        if set(record) != set(columns) or record.get("pmid") != pmid:
            raise ValueError(f"Metadata schema or PMID validation failed for PMID {pmid}.")
        validated.append({column: record.get(column, "") for column in columns})
        seen.add(pmid)
    return validated


def write_batch_failure(batch_number, pmids, attempts, error=None):
    if error is not None:
        append_failure(batch_number, pmids, error, attempts)
        return
    missing_error = RuntimeError("PMID absent from EFetch response; metadata not invented.")
    append_failure(batch_number, pmids, missing_error, attempts)


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
        original_missing = read_failure_pmids()
        fetcher = load_fetch_helpers()
        candidate_rows = fetcher.load_candidates()
        missing_provenance = [pmid for pmid in original_missing if pmid not in candidate_rows]
        if missing_provenance:
            raise ValueError(
                f"Candidate provenance is missing for {len(missing_provenance)} failure PMID(s)."
            )
        main_metadata_pmids = read_metadata_pmids(METADATA_INPUT, fetcher.OUTPUT_COLUMNS)
        recovered = load_recovered_records(fetcher.OUTPUT_COLUMNS)
    except (OSError, UnicodeError, csv.Error, ValueError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    original_set = set(original_missing)
    already_in_metadata = original_set & main_metadata_pmids
    already_recovered = original_set & set(recovered)
    pending_pmids = [
        pmid for pmid in original_missing
        if pmid not in main_metadata_pmids and pmid not in recovered
    ]
    batches = [
        pending_pmids[index:index + BATCH_SIZE]
        for index in range(0, len(pending_pmids), BATCH_SIZE)
    ]

    reset_failure_log()
    failed_pmids = set()
    failed_batches = 0

    print("=" * 72)
    print("PUBMED METADATA RECOVERY")
    print("=" * 72)
    print(f"Original missing PMIDs: {len(original_missing)}")
    print(f"Already in metadata: {len(already_in_metadata)}")
    print(f"Already recovered: {len(already_recovered)}")
    print(f"Remaining to fetch: {len(pending_pmids)}")
    print(f"Batch size: {BATCH_SIZE}")

    for batch_number, batch_pmids in enumerate(batches, start=1):
        print(f"\nBatch {batch_number}/{len(batches)}: {len(batch_pmids)} PMIDs")
        records, attempts, error = fetch_with_retries(fetcher, batch_pmids, email, api_key)

        if error is not None:
            failed_batches += 1
            failed_pmids.update(batch_pmids)
            write_batch_failure(batch_number, batch_pmids, attempts, error)
            print(f"  FAILED after {attempts} attempt(s): {type(error).__name__}: {error}")
        else:
            try:
                validated_records = validate_records(
                    records,
                    batch_pmids,
                    set(recovered) | main_metadata_pmids,
                    candidate_rows,
                    fetcher.OUTPUT_COLUMNS,
                )
                append_records(validated_records, fetcher.OUTPUT_COLUMNS)
                recovered.update({record["pmid"]: record for record in validated_records})
            except (OSError, ValueError, csv.Error) as validation_error:
                failed_batches += 1
                failed_pmids.update(batch_pmids)
                write_batch_failure(batch_number, batch_pmids, attempts, validation_error)
                print(f"  FAILED validation/write: {type(validation_error).__name__}: {validation_error}")
            else:
                returned_pmids = set(records)
                missing_pmids = [pmid for pmid in batch_pmids if pmid not in returned_pmids]
                if missing_pmids:
                    failed_batches += 1
                    failed_pmids.update(missing_pmids)
                    write_batch_failure(batch_number, missing_pmids, attempts)
                    print(f"  Stored {len(validated_records)}; {len(missing_pmids)} absent from EFetch response.")
                else:
                    print(f"  Stored {len(validated_records)} metadata record(s).")

        if batch_number < len(batches):
            time.sleep(REQUEST_DELAY)

    recovered_records = list(recovered.values())
    print("\n[Recovery summary]")
    print(f"Original missing PMIDs: {len(original_missing)}")
    print(f"Recovered PMIDs: {len(original_set & set(recovered))}")
    print(f"Still missing: {len(failed_pmids)}")
    print(f"Recovery batches: {len(batches)}")
    print(f"Failed batches: {failed_batches}")

    print("\n[Metadata coverage for recovered records]")
    print(f"PMIDs with title: {sum(bool(row.get('title', '').strip()) for row in recovered_records)}")
    print(f"PMIDs with abstract: {sum(bool(row.get('abstract', '').strip()) for row in recovered_records)}")
    print(f"PMIDs with PMCID: {sum(bool(row.get('pmcid', '').strip()) for row in recovered_records)}")
    print(f"PMIDs with DOI: {sum(bool(row.get('doi', '').strip()) for row in recovered_records)}")
    print(f"PMIDs with MeSH terms: {sum(bool(row.get('mesh_terms', '').strip()) for row in recovered_records)}")
    print(f"\nOutput: {RECOVERED_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"Remaining failures: {RECOVERY_FAILURE_OUTPUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())