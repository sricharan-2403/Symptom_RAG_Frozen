import csv
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
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_missing_pmid_record_types.csv"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TOOL_NAME = "symptom_rag_analyser"

EXPECTED_PMID_COUNT = 1170
REQUEST_DELAY = 0.5
HTTP_TIMEOUT = 30
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 1.0
OUTPUT_COLUMNS = (
    "pmid",
    "record_type",
    "http_success",
    "matching_record_found",
    "title_if_available",
    "publication_date_if_available",
    "error_type",
    "error_message",
)


def read_pmids():
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

    if len(pmids) != EXPECTED_PMID_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PMID_COUNT} unique non-empty PMIDs; found {len(pmids)}."
        )
    return pmids


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def element_text(element):
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def first_descendant(element, tag_name):
    if element is None:
        return None
    return next((item for item in element.iter() if local_name(item.tag) == tag_name), None)


def record_pmid(record, record_type):
    if record_type == "pubmed_article":
        container = first_descendant(record, "MedlineCitation")
        pmid_element = first_descendant(container, "PMID")
    else:
        pmid_element = first_descendant(record, "PMID")
    return element_text(pmid_element)


def extract_publication_date(record):
    pub_date = first_descendant(record, "PubDate")
    if pub_date is None:
        return ""
    medline_date = first_descendant(pub_date, "MedlineDate")
    if medline_date is not None and element_text(medline_date):
        return element_text(medline_date)

    parts = []
    for item in list(pub_date):
        value = element_text(item)
        if value:
            parts.append(value)
    return " ".join(parts)


def classify_xml(xml_bytes, pmid):
    root = ET.fromstring(xml_bytes)
    matching_articles = [
        record
        for record in root.iter()
        if local_name(record.tag) == "PubmedArticle"
        and record_pmid(record, "pubmed_article") == pmid
    ]
    matching_books = [
        record
        for record in root.iter()
        if local_name(record.tag) == "PubmedBookArticle"
        and record_pmid(record, "pubmed_book_article") == pmid
    ]

    if matching_articles:
        record_type = "pubmed_article"
        record = matching_articles[0]
        title = element_text(first_descendant(record, "ArticleTitle"))
        matching_record_found = True
    elif matching_books:
        record_type = "pubmed_book_article"
        record = matching_books[0]
        title = element_text(first_descendant(record, "ArticleTitle"))
        matching_record_found = True
    else:
        any_matching_pmid = any(
            local_name(item.tag) == "PMID" and element_text(item) == pmid
            for item in root.iter()
        )
        matching_record_found = any_matching_pmid
        record_type = "other" if any_matching_pmid else "not_found"
        record = root if any_matching_pmid else None
        title = element_text(first_descendant(record, "ArticleTitle"))

    return {
        "record_type": record_type,
        "matching_record_found": "true" if matching_record_found else "false",
        "title_if_available": title,
        "publication_date_if_available": extract_publication_date(record),
    }


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


def classify_pmid(pmid, email, api_key):
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

    request = urllib.request.Request(
        f"{EFETCH_URL}?{urllib.parse.urlencode(parameters)}",
        headers={"User-Agent": TOOL_NAME},
    )
    http_success = False
    last_error = None

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
                http_success = True
                xml_bytes = response.read()
            result = classify_xml(xml_bytes, pmid)
            return {
                "pmid": pmid,
                **result,
                "http_success": "true",
                "error_type": "",
                "error_message": "",
            }
        except OSError as error:
            last_error = error
            if attempt > MAX_RETRIES or not is_retryable(error):
                break
            delay = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            print(
                f"  PMID {pmid}: retry {attempt}/{MAX_RETRIES} after "
                f"{type(error).__name__}; waiting {delay:.1f}s."
            )
            time.sleep(delay)
        except (ET.ParseError, ValueError) as error:
            return {
                "pmid": pmid,
                "record_type": "error",
                "http_success": "true" if http_success else "false",
                "matching_record_found": "false",
                "title_if_available": "",
                "publication_date_if_available": "",
                "error_type": type(error).__name__,
                "error_message": safe_error_message(error),
            }

    return {
        "pmid": pmid,
        "record_type": "error",
        "http_success": "true" if http_success else "false",
        "matching_record_found": "false",
        "title_if_available": "",
        "publication_date_if_available": "",
        "error_type": type(last_error).__name__ if last_error else "UnknownError",
        "error_message": safe_error_message(last_error) if last_error else "Request failed.",
    }


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
            "Set NCBI_EMAIL in the project .env file or environment.",
            file=sys.stderr,
        )
        return 1
    api_key = os.environ.get("NCBI_API_KEY", "").strip()

    if OUTPUT_FILE.exists():
        print(f"ERROR: Refusing to overwrite existing output: {OUTPUT_FILE}", file=sys.stderr)
        return 1

    try:
        pmids = read_pmids()
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    records = []
    for index, pmid in enumerate(pmids, start=1):
        print(f"Request {index}/{len(pmids)}: PMID {pmid}")
        records.append(classify_pmid(pmid, email, api_key))
        if index < len(pmids):
            time.sleep(REQUEST_DELAY)

    try:
        write_results(records)
    except OSError as error:
        print(f"ERROR: Could not write output without overwriting: {error}", file=sys.stderr)
        return 1

    article_count = sum(row["record_type"] == "pubmed_article" for row in records)
    book_count = sum(row["record_type"] == "pubmed_book_article" for row in records)
    other_count = sum(row["record_type"] == "other" for row in records)
    not_found_count = sum(row["record_type"] == "not_found" for row in records)
    error_count = sum(row["record_type"] == "error" for row in records)
    matching_count = sum(row["matching_record_found"] == "true" for row in records)
    http_success_count = sum(row["http_success"] == "true" for row in records)

    print("\n" + "=" * 60)
    print("PUBMED MISSING PMID RECORD TYPE CLASSIFICATION")
    print("=" * 60)
    print(f"Input PMIDs: {len(records)}")
    print(f"HTTP-success: {http_success_count}")
    print(f"Errors: {error_count}")
    print(f"Matching records: {matching_count}")
    print(f"PubmedArticle: {article_count}")
    print(f"PubmedBookArticle: {book_count}")
    print(f"Other: {other_count}")
    print(f"Not found: {not_found_count}")

    print("\nRecord-type distribution:")
    for record_type, count in (
        ("pubmed_article", article_count),
        ("pubmed_book_article", book_count),
        ("other", other_count),
        ("not_found", not_found_count),
        ("error", error_count),
    ):
        print(f"{record_type}: {count}")

    print("\nFirst 20 classifications:")
    for row in records[:20]:
        print(f"{row['pmid']} | {row['record_type']} | {row['title_if_available']}")
    print(f"\nOutput: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())