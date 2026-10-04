import argparse
import csv
import importlib.util
import os
import sys
import tempfile
import time
import urllib.error
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAILURE_INPUT = PROJECT_ROOT / "data" / "processed" / "pubmed_metadata_failures.csv"
RECOVERED_OUTPUT = PROJECT_ROOT / "data" / "processed" / "pubmed_metadata_recovered.csv"
RECOVERY_FAILURE_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "pubmed_metadata_recovery_failures.csv"
)
FETCH_SCRIPT = PROJECT_ROOT / "scripts" / "fetch_pubmed_metadata.py"

EXPECTED_PMID_COUNT = 1170
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


def load_fetcher():
    spec = importlib.util.spec_from_file_location("pubmed_metadata_fetcher", FETCH_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load PubMed metadata helpers from {FETCH_SCRIPT}")
    fetcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetcher)
    return fetcher


def read_failure_pmids():
    with FAILURE_INPUT.open("r", encoding="utf-8-sig", newline="") as failure_file:
        reader = csv.DictReader(failure_file)
        if "pmids" not in (reader.fieldnames or []):
            raise ValueError("Failure log is missing the required 'pmids' column.")

        pmids = []
        seen = set()
        for row in reader:
            for value in (row.get("pmids") or "").split(","):
                pmid = value.strip()
                if not pmid:
                    continue
                if not pmid.isdigit():
                    raise ValueError(f"Invalid non-numeric PMID value: {pmid!r}")
                if pmid in seen:
                    raise ValueError(f"Duplicate PMID in original failure log: {pmid}")
                seen.add(pmid)
                pmids.append(pmid)

    if len(pmids) != EXPECTED_PMID_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PMID_COUNT} unique PMIDs; found {len(pmids)}."
        )
    return pmids


def load_recovered_records(columns, target_pmids, path=RECOVERED_OUTPUT):
    if not path.exists():
        return {}

    records = {}
    with path.open("r", encoding="utf-8-sig", newline="") as recovered_file:
        reader = csv.DictReader(recovered_file)
        if reader.fieldnames != list(columns):
            raise ValueError(f"Existing recovery output has an unexpected schema: {path}")
        for row_number, raw_row in enumerate(reader, start=2):
            record = {column: raw_row.get(column) or "" for column in columns}
            pmid = record["pmid"].strip()
            if not pmid:
                raise ValueError(f"Existing recovery output has an empty PMID at row {row_number}.")
            if pmid not in target_pmids:
                raise ValueError(f"Existing recovered PMID {pmid} is not in the target failure list.")
            if pmid in records:
                raise ValueError(f"Duplicate PMID in existing recovery output: {pmid}")
            records[pmid] = record
    return records


def reset_failure_log():
    with RECOVERY_FAILURE_OUTPUT.open("w", encoding="utf-8", newline="") as failure_file:
        csv.DictWriter(failure_file, fieldnames=FAILURE_COLUMNS).writeheader()


def safe_error_message(error):
    if isinstance(error, urllib.error.HTTPError):
        return f"HTTP {error.code}" + (f": {error.reason}" if error.reason else "")
    if isinstance(error, urllib.error.URLError):
        return str(error.reason)
    return str(error)


def append_failure(batch_number, pmids, error, attempts):
    with RECOVERY_FAILURE_OUTPUT.open("a", encoding="utf-8", newline="") as failure_file:
        writer = csv.DictWriter(failure_file, fieldnames=FAILURE_COLUMNS)
        writer.writerow({
            "batch_number": batch_number,
            "pmids": ",".join(pmids),
            "error_type": type(error).__name__,
            "error_message": safe_error_message(error).replace("\r", " ").replace("\n", " "),
            "attempts": attempts,
        })
        failure_file.flush()
        os.fsync(failure_file.fileno())


def read_logged_failure_pmids(target_pmids):
    with RECOVERY_FAILURE_OUTPUT.open("r", encoding="utf-8-sig", newline="") as failure_file:
        reader = csv.DictReader(failure_file)
        missing_columns = set(FAILURE_COLUMNS) - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Recovery failure output is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        failed = set()
        for row in reader:
            for value in (row.get("pmids") or "").split(","):
                pmid = value.strip()
                if not pmid:
                    continue
                if pmid not in target_pmids:
                    raise ValueError(f"Failure log contains non-target PMID {pmid}.")
                if pmid in failed:
                    raise ValueError(f"Duplicate PMID in recovery failure log: {pmid}")
                failed.add(pmid)
    return failed


def fetch_with_retries(fetcher, pmids, email, api_key):
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fetcher.fetch_batch(pmids, email, api_key), attempt, None
        except Exception as error:
            last_error = error
            if attempt >= MAX_ATTEMPTS or not is_retryable(error):
                break
            delay = max(REQUEST_DELAY, BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
            print(
                f"  Attempt {attempt}/{MAX_ATTEMPTS} failed ({type(error).__name__}); "
                f"retrying in {delay:.1f}s."
            )
            time.sleep(delay)
    return {}, attempt, last_error


def is_retryable(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 429) or 500 <= error.code < 600
    return isinstance(error, (urllib.error.URLError, TimeoutError, OSError, ET.ParseError))


def prepare_records(fetched, batch_pmids, target_pmids, candidate_rows, columns, provenance_columns):
    requested = set(batch_pmids)
    validated = []
    seen = set()
    for pmid, metadata in fetched.items():
        if pmid not in target_pmids or pmid not in requested:
            raise ValueError(f"Fetcher returned unrequested PMID {pmid}.")
        if pmid in seen:
            raise ValueError(f"Fetcher returned duplicate PMID {pmid}.")
        candidate = candidate_rows.get(pmid)
        if candidate is None:
            raise ValueError(f"Candidate provenance is missing for PMID {pmid}.")

        record = dict(metadata)
        record.update({column: candidate.get(column, "") for column in provenance_columns})
        if set(record) != set(columns) or record.get("pmid") != pmid:
            raise ValueError(f"Recovered record for PMID {pmid} does not match output schema.")
        validated.append({column: record.get(column, "") for column in columns})
        seen.add(pmid)
    return validated


def append_recovered(records, columns, existing_pmids):
    new_pmids = [record["pmid"] for record in records]
    if len(new_pmids) != len(set(new_pmids)):
        raise ValueError("Recovery batch contains duplicate PMIDs.")
    duplicate_pmids = existing_pmids & set(new_pmids)
    if duplicate_pmids:
        raise ValueError("Recovery batch duplicates existing PMID(s): " + ",".join(sorted(duplicate_pmids)))
    if not records:
        return

    is_new_file = not RECOVERED_OUTPUT.exists() or RECOVERED_OUTPUT.stat().st_size == 0
    with RECOVERED_OUTPUT.open("a", encoding="utf-8", newline="") as recovered_file:
        writer = csv.DictWriter(recovered_file, fieldnames=columns)
        if is_new_file:
            writer.writeheader()
        writer.writerows(records)
        recovered_file.flush()
        os.fsync(recovered_file.fileno())


def force_refresh(fetcher, input_pmids, target_pmids, candidate_rows, email, api_key):
    refreshed = {}
    batches = [
        input_pmids[index:index + BATCH_SIZE]
        for index in range(0, len(input_pmids), BATCH_SIZE)
    ]
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=RECOVERED_OUTPUT.parent,
            prefix=f"{RECOVERED_OUTPUT.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.DictWriter(temporary_file, fieldnames=fetcher.OUTPUT_COLUMNS)
            writer.writeheader()

            for batch_number, batch_pmids in enumerate(batches, start=1):
                print(f"\nBatch {batch_number}/{len(batches)}: {len(batch_pmids)} PMIDs")
                fetched, attempts, error = fetch_with_retries(
                    fetcher, batch_pmids, email, api_key
                )

                if error is not None:
                    append_failure(batch_number, batch_pmids, error, attempts)
                    print(f"  Batch failed after {attempts} attempt(s): {type(error).__name__}")
                else:
                    try:
                        records = prepare_records(
                            fetched,
                            batch_pmids,
                            target_pmids,
                            candidate_rows,
                            fetcher.OUTPUT_COLUMNS,
                            fetcher.PROVENANCE_COLUMNS,
                        )
                        record_by_pmid = {record["pmid"]: record for record in records}
                        records = [
                            record_by_pmid[pmid]
                            for pmid in batch_pmids
                            if pmid in record_by_pmid
                        ]
                        writer.writerows(records)
                        temporary_file.flush()
                    except (OSError, ValueError, csv.Error) as validation_error:
                        append_failure(
                            batch_number, batch_pmids, validation_error, attempts
                        )
                        print(
                            "  Batch validation/write failed: "
                            f"{type(validation_error).__name__}"
                        )
                    else:
                        refreshed.update(
                            {record["pmid"]: record for record in records}
                        )
                        missing_pmids = [
                            pmid for pmid in batch_pmids if pmid not in fetched
                        ]
                        if missing_pmids:
                            missing_error = RuntimeError(
                                "EFetch response did not contain requested PMID(s)."
                            )
                            append_failure(
                                batch_number, missing_pmids, missing_error, attempts
                            )
                            print(
                                f"  Stored {len(records)}; "
                                f"{len(missing_pmids)} PMID(s) were absent."
                            )
                        else:
                            print(f"  Staged {len(records)} recovery record(s).")

                if batch_number < len(batches):
                    time.sleep(REQUEST_DELAY)

            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        missing_pmids = target_pmids - set(refreshed)
        try:
            failed_pmids = read_logged_failure_pmids(target_pmids)
        except (OSError, UnicodeError, csv.Error, ValueError):
            raise

        unaccounted_pmids = missing_pmids - failed_pmids
        if unaccounted_pmids:
            append_failure(
                0,
                sorted(unaccounted_pmids),
                RuntimeError("PMID was neither refreshed nor explicitly logged as a failure."),
                0,
            )
            failed_pmids = read_logged_failure_pmids(target_pmids)

        if failed_pmids & set(refreshed):
            raise ValueError("A PMID is both refreshed and logged as failed.")

        if missing_pmids or failed_pmids:
            return refreshed, False

        verified_records = load_recovered_records(
            fetcher.OUTPUT_COLUMNS, target_pmids, temporary_path
        )
        if set(verified_records) != target_pmids:
            raise ValueError("Staged refresh does not contain every target PMID.")

        os.replace(temporary_path, RECOVERED_OUTPUT)
        temporary_path = None
        return verified_records, True
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def print_summary(input_pmids, recovered, target_pmids):
    recovered_pmids = set(recovered)
    still_missing_pmids = target_pmids - recovered_pmids
    records = list(recovered.values())
    record_type_counts = Counter(
        (record.get("record_type") or "").strip() or "(empty)"
        for record in records
    )

    print("\nPUBMED BOOK ARTICLE RECOVERY")
    print(f"Input PMIDs: {len(input_pmids)}")
    print(f"Recovered: {len(recovered_pmids)}")
    print(f"Still missing: {len(still_missing_pmids)}")
    print(f"Unique recovered PMIDs: {len(recovered_pmids)}")
    print(
        "PMCID count: "
        f"{sum(bool((record.get('pmcid') or '').strip()) for record in records)}"
    )
    print(
        "DOI count: "
        f"{sum(bool((record.get('doi') or '').strip()) for record in records)}"
    )
    print("record_type distribution:")
    if record_type_counts:
        for record_type, count in sorted(record_type_counts.items()):
            print(f"  {record_type}: {count}")
    else:
        print("  (none)")
    print(f"Output path: {RECOVERED_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(
        "Recovery failure path: "
        f"{RECOVERY_FAILURE_OUTPUT.relative_to(PROJECT_ROOT)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Recover failed PubMed book articles.")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Refetch all target PMIDs and replace the recovery output only on full success.",
    )
    args = parser.parse_args()

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

    try:
        fetcher = load_fetcher()
        input_pmids = read_failure_pmids()
        target_pmids = set(input_pmids)
        candidate_rows = fetcher.load_candidates()
        missing_provenance = [pmid for pmid in input_pmids if pmid not in candidate_rows]
        if missing_provenance:
            raise ValueError(
                f"Candidate provenance is missing for {len(missing_provenance)} target PMID(s)."
            )
        recovered = (
            {}
            if args.force_refresh
            else load_recovered_records(fetcher.OUTPUT_COLUMNS, target_pmids)
        )
    except (OSError, UnicodeError, csv.Error, ValueError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    reset_failure_log()
    if args.force_refresh:
        print(f"Input PMIDs: {len(input_pmids)}")
        print("Already recovered: 0")
        print(f"Remaining to fetch: {len(input_pmids)}")
        try:
            refreshed, refresh_succeeded = force_refresh(
                fetcher,
                input_pmids,
                target_pmids,
                candidate_rows,
                email,
                api_key,
            )
        except (OSError, UnicodeError, csv.Error, ValueError) as error:
            print(f"ERROR: Forced refresh failed: {error}", file=sys.stderr)
            refreshed = {}
            refresh_succeeded = False

        print_summary(input_pmids, refreshed, target_pmids)
        if not refresh_succeeded:
            print("\nForced refresh failed; existing recovery output was not replaced.")
            return 1
        return 0

    pending = [pmid for pmid in input_pmids if pmid not in recovered]
    batches = [pending[index:index + BATCH_SIZE] for index in range(0, len(pending), BATCH_SIZE)]
    print(f"Input PMIDs: {len(input_pmids)}")
    print(f"Already recovered: {len(recovered)}")
    print(f"Remaining to fetch: {len(pending)}")

    for batch_number, batch_pmids in enumerate(batches, start=1):
        print(f"\nBatch {batch_number}/{len(batches)}: {len(batch_pmids)} PMIDs")
        fetched, attempts, error = fetch_with_retries(fetcher, batch_pmids, email, api_key)

        if error is not None:
            append_failure(batch_number, batch_pmids, error, attempts)
            print(f"  Batch failed after {attempts} attempt(s): {type(error).__name__}")
        else:
            try:
                records = prepare_records(
                    fetched,
                    batch_pmids,
                    target_pmids,
                    candidate_rows,
                    fetcher.OUTPUT_COLUMNS,
                    fetcher.PROVENANCE_COLUMNS,
                )
                append_recovered(records, fetcher.OUTPUT_COLUMNS, set(recovered))
            except (OSError, ValueError, csv.Error) as validation_error:
                append_failure(batch_number, batch_pmids, validation_error, attempts)
                print(f"  Batch validation/write failed: {type(validation_error).__name__}")
            else:
                recovered.update({record["pmid"]: record for record in records})
                missing_pmids = [pmid for pmid in batch_pmids if pmid not in fetched]
                if missing_pmids:
                    missing_error = RuntimeError(
                        "EFetch response did not contain requested PMID(s)."
                    )
                    append_failure(batch_number, missing_pmids, missing_error, attempts)
                    print(f"  Stored {len(records)}; {len(missing_pmids)} PMID(s) were absent.")
                else:
                    print(f"  Stored {len(records)} recovery record(s).")

        if batch_number < len(batches):
            time.sleep(REQUEST_DELAY)

    try:
        verified_recovered = load_recovered_records(fetcher.OUTPUT_COLUMNS, target_pmids)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"ERROR: Recovery output validation failed: {error}", file=sys.stderr)
        return 1

    recovered_pmids = set(verified_recovered)
    still_missing_pmids = target_pmids - recovered_pmids
    try:
        explicitly_failed_pmids = read_logged_failure_pmids(target_pmids)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"ERROR: Recovery failure log validation failed: {error}", file=sys.stderr)
        return 1

    unexpected_failure_pmids = explicitly_failed_pmids & recovered_pmids
    if unexpected_failure_pmids:
        print(
            "ERROR: PMID(s) are both recovered and logged as failed: "
            + ",".join(sorted(unexpected_failure_pmids)),
            file=sys.stderr,
        )
        return 1

    unaccounted_pmids = still_missing_pmids - explicitly_failed_pmids
    if unaccounted_pmids:
        unaccounted_error = RuntimeError("PMID was neither recovered nor explicitly logged as a failure.")
        append_failure(0, sorted(unaccounted_pmids), unaccounted_error, 0)
        try:
            explicitly_failed_pmids = read_logged_failure_pmids(target_pmids)
        except (OSError, UnicodeError, csv.Error, ValueError) as error:
            print(f"ERROR: Recovery failure log validation failed: {error}", file=sys.stderr)
            return 1

    if target_pmids != recovered_pmids | explicitly_failed_pmids:
        print("ERROR: Recovery output does not account for every target PMID.", file=sys.stderr)
        return 1

    print_summary(input_pmids, verified_recovered, target_pmids)

    return 1 if still_missing_pmids else 0


if __name__ == "__main__":
    raise SystemExit(main())