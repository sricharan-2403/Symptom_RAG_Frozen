import csv
import importlib.util
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAILURE_INPUT = PROJECT_ROOT / "data" / "processed" / "pubmed_metadata_failures.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_missing_pmid_diagnostic_20.csv"
FETCH_SCRIPT = PROJECT_ROOT / "scripts" / "fetch_pubmed_metadata.py"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TOOL_NAME = "symptom_rag_analyser"

PMIDS_TO_TEST = 20
REQUEST_DELAY = 0.5
HTTP_TIMEOUT = 30
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 1.0
OUTPUT_COLUMNS = (
    "pmid",
    "http_success",
    "record_found",
    "title",
    "journal",
    "publication_date",
    "pmcid",
    "doi",
    "abstract_present",
    "error_type",
    "error_message",
)


def load_metadata_parser():
    spec = importlib.util.spec_from_file_location("pubmed_metadata_parser", FETCH_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load PubMed metadata parser from {FETCH_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_first_pmids():
    with FAILURE_INPUT.open("r", encoding="utf-8-sig", newline="") as failure_file:
        reader = csv.DictReader(failure_file)
        if "pmids" not in (reader.fieldnames or []):
            raise ValueError("Failure log is missing the required 'pmids' column.")

        pmids = []
        seen = set()
        for row in reader:
            for value in (row.get("pmids") or "").split(","):
                pmid = value.strip()
                if not pmid or pmid in seen:
                    continue
                if not pmid.isdigit():
                    raise ValueError(f"Invalid non-numeric PMID value: {pmid!r}")
                seen.add(pmid)
                pmids.append(pmid)
                if len(pmids) == PMIDS_TO_TEST:
                    return pmids
    return pmids


def safe_error_message(error):
    if isinstance(error, urllib.error.HTTPError):
        reason = error.reason
        return f"HTTP {error.code}" + (f": {reason}" if reason else "")
    if isinstance(error, urllib.error.URLError):
        return str(error.reason)
    return str(error)


def request_pubmed_record(pmid, email, api_key, parser):
    parameters = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
        "rettype": "abstract",
        "tool": TOOL_NAME,
        "email": email,
    }
    if api_key:
        parameters["api_key"] = api_key

    url = f"{EFETCH_URL}?{urllib.parse.urlencode(parameters)}"
    request = urllib.request.Request(url, headers={"User-Agent": TOOL_NAME})
    http_success = False
    last_error = None

    for attempt in range(1, MAX_RETRIES + 2):
        matching_record_found = False
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
                http_success = True
                xml_data = response.read()
            root = ET.fromstring(xml_data)
            matching_article = None
            matching_pmid = ""
            for article in root.iter("PubmedArticle"):
                medline_citation = article.find("MedlineCitation")
                article_pmid = parser.child_text(medline_citation, "PMID")
                if article_pmid == pmid:
                    matching_article = article
                    matching_pmid = article_pmid
                    matching_record_found = True
                    break

            result = {
                "pmid": pmid,
                "http_success": "true" if http_success else "false",
                "record_found": "true" if matching_record_found else "false",
                "title": "",
                "journal": "",
                "publication_date": "",
                "pmcid": "",
                "doi": "",
                "abstract_present": "false",
                "error_type": "",
                "error_message": "",
            }
            if matching_article is not None:
                metadata = parser.parse_pubmed_article(matching_article)
                medline_citation = matching_article.find("MedlineCitation")
                article_element = (
                    medline_citation.find("Article")
                    if medline_citation is not None
                    else None
                )
                result.update({
                    "pmid": matching_pmid,
                    "record_found": "true",
                    "title": metadata["title"],
                    "journal": metadata["journal"],
                    "publication_date": metadata["publication_date"],
                    "pmcid": metadata["pmcid"],
                    "doi": metadata["doi"],
                    "abstract_present": "true"
                    if article_element is not None and article_element.find("Abstract") is not None
                    else "false",
                })
            return result
        except OSError as error:
            last_error = error
            if attempt > MAX_RETRIES or not is_retryable(error):
                break
            delay = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            print(
                f"  PMID {pmid}: attempt {attempt}/{MAX_RETRIES + 1} failed "
                f"({type(error).__name__}); retrying in {delay:.1f}s."
            )
            time.sleep(delay)
        except (ET.ParseError, ValueError, KeyError) as error:
            result = {
                "pmid": pmid,
                "http_success": "true" if http_success else "false",
                "record_found": "true" if matching_record_found else "false",
                "title": "",
                "journal": "",
                "publication_date": "",
                "pmcid": "",
                "doi": "",
                "abstract_present": "false",
                "error_type": type(error).__name__,
                "error_message": safe_error_message(error),
            }
            return result

    return {
        "pmid": pmid,
        "http_success": "true" if http_success else "false",
        "record_found": "false",
        "title": "",
        "journal": "",
        "publication_date": "",
        "pmcid": "",
        "doi": "",
        "abstract_present": "false",
        "error_type": type(last_error).__name__ if last_error else "UnknownError",
        "error_message": safe_error_message(last_error) if last_error else "Request failed.",
    }


def is_retryable(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 429) or 500 <= error.code < 600
    return isinstance(error, (urllib.error.URLError, TimeoutError, OSError))


def write_results(records):
    with OUTPUT_FILE.open("x", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
        output_file.flush()
        os.fsync(output_file.fileno())


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

    if OUTPUT_FILE.exists():
        print(f"ERROR: Refusing to overwrite existing output: {OUTPUT_FILE}", file=sys.stderr)
        return 1

    try:
        pmids = read_first_pmids()
        if len(pmids) < PMIDS_TO_TEST:
            raise ValueError(
                f"Expected at least {PMIDS_TO_TEST} unique PMIDs; found {len(pmids)}."
            )
        parser = load_metadata_parser()
    except (OSError, UnicodeError, csv.Error, ValueError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    results = []
    for index, pmid in enumerate(pmids, start=1):
        print(f"Request {index}/{len(pmids)}: PMID {pmid}")
        results.append(request_pubmed_record(pmid, email, api_key, parser))
        if index < len(pmids):
            time.sleep(REQUEST_DELAY)

    try:
        write_results(results)
    except OSError as error:
        print(f"ERROR: Could not safely write diagnostic output: {error}", file=sys.stderr)
        return 1

    http_success_count = sum(row["http_success"] == "true" for row in results)
    record_found_count = sum(row["record_found"] == "true" for row in results)
    abstract_count = sum(row["abstract_present"] == "true" for row in results)
    error_count = sum(bool(row["error_type"]) for row in results)

    print("\nPUBMED MISSING PMID DIAGNOSTIC")
    print(f"PMIDs tested: {len(results)}")
    print(f"HTTP-success count: {http_success_count}")
    print(f"Records found count: {record_found_count}")
    print(f"Records not found count: {len(results) - record_found_count}")
    print(f"Abstracts present count: {abstract_count}")
    print(f"Errors count: {error_count}")
    print("\nPer-PMID results:")
    for row in results:
        print(
            f"{row['pmid']} | HTTP success: {row['http_success']} | "
            f"record found: {row['record_found']} | "
            f"title present: {'true' if row['title'] else 'false'} | "
            f"abstract present: {row['abstract_present']}"
        )
    print(f"\nOutput: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())