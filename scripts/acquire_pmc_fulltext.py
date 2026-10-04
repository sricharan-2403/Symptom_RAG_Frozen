import argparse
import csv
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_fulltext_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "pmc"
RESULTS_FILE = PROJECT_ROOT / "data" / "processed" / "pmc_fulltext_acquisition_results.csv"
FAILURES_FILE = PROJECT_ROOT / "data" / "processed" / "pmc_fulltext_acquisition_failures.csv"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TOOL_NAME = "symptom_rag_analyser"
USER_AGENT = "Symptom-RAG-Analyser/1.0 (PMC full-text acquisition)"
HTTP_TIMEOUT = 60
MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 1.0
DEFAULT_REQUEST_DELAY = 0.34

RESULT_COLUMNS = (
    "pmid_requested",
    "pmcid_requested",
    "pmid_xml",
    "pmcid_xml",
    "title",
    "abstract",
    "status",
    "has_abstract",
    "has_body",
    "has_license",
    "license",
    "local_path",
    "source_url",
    "attempts",
    "error_type",
    "error_message",
    "disease_ids",
    "disease_names",
    "query_categories",
    "publication_year",
    "record_type",
    "source",
)
FAILURE_COLUMNS = RESULT_COLUMNS
STATUSES = (
    "FULL_TEXT_XML",
    "ABSTRACT_ONLY",
    "RESTRICTED_XML",
    "INVALID_XML",
    "REQUEST_FAILED",
)
MANIFEST_COLUMNS = {
    "pmid",
    "pmcid",
    "acquisition_route",
    "disease_ids",
    "disease_names",
    "query_categories",
    "publication_year",
    "record_type",
}


def local_name(name):
    return name.rsplit("}", 1)[-1].casefold()


def text_content(element):
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def element_attribute(element, name):
    for key, value in element.attrib.items():
        if local_name(key) == name.casefold():
            return value
    return ""


def article_metadata(article):
    for front in article:
        if local_name(front.tag) != "front":
            continue
        for metadata in front:
            if local_name(metadata.tag) == "article-meta":
                return metadata
    return None


def load_manifest_records():
    if not MANIFEST_FILE.is_file():
        raise FileNotFoundError(f"Full-text manifest not found: {MANIFEST_FILE}")

    with MANIFEST_FILE.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        missing = MANIFEST_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Manifest is missing required columns: " + ", ".join(sorted(missing))
            )
        return [
            row
            for row in reader
            if (row.get("acquisition_route") or "").strip() == "pmc"
            and (row.get("pmcid") or "").strip()
        ]


def is_retryable(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 429) or 500 <= error.code < 600
    return isinstance(
        error,
        (urllib.error.URLError, TimeoutError, ConnectionError, OSError),
    )


def safe_error_message(error):
    if isinstance(error, urllib.error.HTTPError):
        return f"HTTP {error.code}" + (f": {error.reason}" if error.reason else "")
    if isinstance(error, urllib.error.URLError):
        return str(error.reason)
    return str(error)


def source_url(pmcid):
    parameters = {
        "db": "pmc",
        "id": pmcid,
        "retmode": "xml",
        "rettype": "full",
        "tool": TOOL_NAME,
    }
    return f"{EFETCH_URL}?{urllib.parse.urlencode(parameters)}"


def request_xml(pmcid, email, api_key, request_delay):
    parameters = {
        "db": "pmc",
        "id": pmcid,
        "retmode": "xml",
        "rettype": "full",
        "tool": TOOL_NAME,
        "email": email,
    }
    if api_key:
        parameters["api_key"] = api_key

    request = urllib.request.Request(
        f"{EFETCH_URL}?{urllib.parse.urlencode(parameters)}",
        headers={"User-Agent": USER_AGENT},
    )
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
                status = getattr(response, "status", 200)
                xml_bytes = response.read()
            if not 200 <= status < 300:
                error = RuntimeError(f"Unexpected HTTP status: {status}")
                return None, attempt, error
            return xml_bytes, attempt, None
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as error:
            if attempt >= MAX_ATTEMPTS or not is_retryable(error):
                return None, attempt, error
            time.sleep(max(request_delay, BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))))
    return None, MAX_ATTEMPTS, RuntimeError("Request attempts exhausted.")


def article_identifiers(article):
    pmcid = ""
    pmid = ""
    metadata = article_metadata(article)
    if metadata is None:
        return pmid, pmcid

    for element in metadata.iter():
        name = local_name(element.tag)
        identifier_type = element_attribute(element, "pub-id-type").casefold()
        value = text_content(element)
        if not value:
            continue
        if name in {"pmcid", "pmc-id"} and not pmcid:
            pmcid = value
        elif name == "article-id" and identifier_type in {"pmc", "pmcid"} and not pmcid:
            pmcid = value
        elif name == "article-id" and identifier_type == "pmid" and not pmid:
            pmid = value
        elif name == "pmid" and not pmid:
            pmid = value

    if pmcid and not pmcid.casefold().startswith("pmc"):
        pmcid = f"PMC{pmcid}"
    return pmid, pmcid


def find_article(root, requested_pmcid):
    articles = [element for element in root.iter() if local_name(element.tag) == "article"]
    if not articles:
        return None, ""

    first_article = articles[0]
    first_pmcid = article_identifiers(first_article)[1]
    for article in articles:
        article_pmcid = article_identifiers(article)[1]
        if article_pmcid and article_pmcid.casefold() == requested_pmcid.casefold():
            return article, article_pmcid
    return first_article, first_pmcid


def detect_restriction(article):
    phrases = (
        "not available in pmc",
        "not available in pubmed central",
        "full text is not available",
        "full-text is not available",
        "access to full text is restricted",
        "restricted access",
        "embargoed",
    )
    for element in article.iter():
        tag_name = re.sub(r"[-_]", " ", local_name(element.tag))
        if re.search(r"\b(?:restricted|restriction|embargo)\b", tag_name):
            return True
        for key, value in element.attrib.items():
            attribute_name = re.sub(r"[-_]", " ", local_name(key))
            attribute_value = value.casefold()
            if re.search(r"\b(?:restricted|restriction|embargo)\b", attribute_name):
                if attribute_value not in {"", "false", "no", "0", "none"}:
                    return True
            if re.search(r"\b(?:restricted|embargoed)\b", attribute_value):
                return True
        content = text_content(element).casefold()
        if any(phrase in content for phrase in phrases):
            return True
    return False


def extract_license(article):
    values = []
    for element in article.iter():
        name = local_name(element.tag)
        if name in {"license", "license-p", "license-ref"}:
            text = text_content(element)
            href = next(
                (
                    value
                    for key, value in element.attrib.items()
                    if local_name(key) in {"href", "license"}
                ),
                "",
            )
            for value in (text, href):
                if value and value not in values:
                    values.append(value)
    return " | ".join(values)


def inspect_xml(xml_bytes, requested_pmcid):
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as error:
        return {
            "status": "INVALID_XML",
            "pmid_xml": "",
            "pmcid_xml": "",
            "title": "",
            "abstract": "",
            "has_abstract": False,
            "has_body": False,
            "has_license": False,
            "license": "",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

    article, pmcid_xml = find_article(root, requested_pmcid)
    if article is None:
        return {
            "status": "INVALID_XML",
            "pmid_xml": "",
            "pmcid_xml": "",
            "title": "",
            "abstract": "",
            "has_abstract": False,
            "has_body": False,
            "has_license": False,
            "license": "",
            "error_type": "InvalidPMCXML",
            "error_message": "Response does not contain a PMC article XML record.",
        }

    pmid_xml, article_pmcid = article_identifiers(article)
    if article_pmcid:
        pmcid_xml = article_pmcid
    metadata = article_metadata(article)
    title = ""
    abstract = ""
    body = next(
        (element for element in article if local_name(element.tag) == "body"),
        None,
    )
    body_text = text_content(body)
    for element in metadata.iter() if metadata is not None else ():
        name = local_name(element.tag)
        if name == "article-title" and not title:
            title = text_content(element)
        elif name == "abstract" and not abstract:
            abstract = text_content(element)

    license_text = extract_license(metadata) if metadata is not None else ""
    has_body = bool(body_text)
    has_abstract = bool(abstract)
    restriction_scope = metadata if metadata is not None else article
    restriction_detected = detect_restriction(restriction_scope)
    pmcid_matches = bool(pmcid_xml) and pmcid_xml.casefold() == requested_pmcid.casefold()

    if not pmcid_matches:
        status = "INVALID_XML"
        error_type = "PMCIDMismatch"
        error_message = (
            f"Requested {requested_pmcid}; XML identified {pmcid_xml or '(none)'}."
        )
    elif has_body:
        status = "FULL_TEXT_XML"
        error_type = ""
        error_message = ""
    elif restriction_detected:
        status = "RESTRICTED_XML"
        error_type = ""
        error_message = "XML indicates restricted or embargoed content without article body."
    elif has_abstract:
        status = "ABSTRACT_ONLY"
        error_type = ""
        error_message = ""
    else:
        status = "INVALID_XML"
        error_type = "MissingArticleContent"
        error_message = "Matching PMC article XML has neither body, abstract, nor restriction indicators."

    return {
        "status": status,
        "pmid_xml": pmid_xml,
        "pmcid_xml": pmcid_xml,
        "title": title,
        "abstract": abstract,
        "has_abstract": has_abstract,
        "has_body": has_body,
        "has_license": bool(license_text),
        "license": license_text,
        "error_type": error_type,
        "error_message": error_message,
    }


def valid_cached_xml(path, requested_pmcid):
    try:
        validation = inspect_xml(path.read_bytes(), requested_pmcid)
    except OSError as error:
        return None, error
    if validation["status"] == "INVALID_XML":
        return None, None
    return validation, None


def write_xml_atomically(path, xml_bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(xml_bytes)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def acquire_one(record, email, api_key, request_delay, pmcid_cache):
    pmid_requested = (record.get("pmid") or "").strip()
    pmcid_requested = (record.get("pmcid") or "").strip()
    output_path = OUTPUT_DIR / f"{pmcid_requested}.xml"
    url = source_url(pmcid_requested)
    source = ""
    attempts = 0
    error_type = ""
    error_message = ""

    if not re.fullmatch(r"PMC\d+", pmcid_requested, flags=re.IGNORECASE):
        validation = {
            "status": "INVALID_XML",
            "pmid_xml": "",
            "pmcid_xml": "",
            "title": "",
            "abstract": "",
            "has_abstract": False,
            "has_body": False,
            "has_license": False,
            "license": "",
        }
        error_type = "InvalidPMCID"
        error_message = f"Invalid requested PMCID format: {pmcid_requested}"
    elif pmcid_requested in pmcid_cache:
        cached = pmcid_cache[pmcid_requested]
        validation = cached["validation"]
        source = cached["source"]
        attempts = cached["attempts"]
        error_type = cached["error_type"]
        error_message = cached["error_message"]
    else:
        validation = None
        if output_path.is_file():
            validation, read_error = valid_cached_xml(output_path, pmcid_requested)
            if read_error is not None:
                error_type = type(read_error).__name__
                error_message = f"Could not read existing XML: {read_error}"
            elif validation is not None:
                source = "existing_valid_xml"
                pmcid_cache[pmcid_requested] = {
                    "validation": validation,
                    "source": source,
                    "attempts": 0,
                    "error_type": "",
                    "error_message": "",
                }

        if validation is None and not error_message:
            xml_bytes, attempts, request_error = request_xml(
                pmcid_requested, email, api_key, request_delay
            )
            if request_error is not None:
                validation = {
                    "status": "REQUEST_FAILED",
                    "pmid_xml": "",
                    "pmcid_xml": "",
                    "title": "",
                    "abstract": "",
                    "has_abstract": False,
                    "has_body": False,
                    "has_license": False,
                    "license": "",
                }
                error_type = type(request_error).__name__
                error_message = safe_error_message(request_error)
            else:
                validation = inspect_xml(xml_bytes, pmcid_requested)
                if validation["status"] != "INVALID_XML":
                    try:
                        write_xml_atomically(output_path, xml_bytes)
                    except OSError as write_error:
                        validation = {
                            "status": "REQUEST_FAILED",
                            "pmid_xml": validation["pmid_xml"],
                            "pmcid_xml": validation["pmcid_xml"],
                            "title": validation["title"],
                            "abstract": validation["abstract"],
                            "has_abstract": validation["has_abstract"],
                            "has_body": validation["has_body"],
                            "has_license": validation["has_license"],
                            "license": validation["license"],
                        }
                        error_type = type(write_error).__name__
                        error_message = f"Could not save validated XML: {write_error}"
                else:
                    error_type = validation["error_type"]
                    error_message = validation["error_message"]

            source = "downloaded"
            pmcid_cache[pmcid_requested] = {
                "validation": validation,
                "source": source,
                "attempts": attempts,
                "error_type": error_type,
                "error_message": error_message,
            }

        if validation is None:
            validation = {
                "status": "REQUEST_FAILED",
                "pmid_xml": "",
                "pmcid_xml": "",
                "title": "",
                "abstract": "",
                "has_abstract": False,
                "has_body": False,
                "has_license": False,
                "license": "",
            }
            pmcid_cache[pmcid_requested] = {
                "validation": validation,
                "source": "existing_file_unreadable",
                "attempts": 0,
                "error_type": error_type,
                "error_message": error_message,
            }

    source = source or (pmcid_cache.get(pmcid_requested, {}).get("source", "") if pmcid_requested else "")
    attempts = attempts or pmcid_cache.get(pmcid_requested, {}).get("attempts", 0)
    error_type = error_type or pmcid_cache.get(pmcid_requested, {}).get("error_type", "")
    error_message = error_message or pmcid_cache.get(pmcid_requested, {}).get("error_message", "")

    local_path = ""
    if validation["status"] in {"FULL_TEXT_XML", "ABSTRACT_ONLY", "RESTRICTED_XML"}:
        local_path = str(output_path.relative_to(PROJECT_ROOT))

    result = {
        "pmid_requested": pmid_requested,
        "pmcid_requested": pmcid_requested,
        "pmid_xml": validation["pmid_xml"],
        "pmcid_xml": validation["pmcid_xml"],
        "title": validation["title"],
        "abstract": validation["abstract"],
        "status": validation["status"],
        "has_abstract": str(validation["has_abstract"]).lower(),
        "has_body": str(validation["has_body"]).lower(),
        "has_license": str(validation["has_license"]).lower(),
        "license": validation["license"],
        "local_path": local_path,
        "source_url": url,
        "attempts": attempts,
        "error_type": error_type,
        "error_message": error_message,
        "disease_ids": record.get("disease_ids", ""),
        "disease_names": record.get("disease_names", ""),
        "query_categories": record.get("query_categories", ""),
        "publication_year": record.get("publication_year", ""),
        "record_type": record.get("record_type", ""),
        "source": source,
    }
    return result


def atomic_write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.DictWriter(temporary_file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_parser(default_delay):
    parser = argparse.ArgumentParser(description="Acquire and validate PMC full-text XML.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N eligible manifest rows (for a small test run).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=default_delay,
        help="Minimum seconds between PMC requests (default: %(default)s).",
    )
    return parser


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    default_delay = float(os.environ.get("PMC_REQUEST_DELAY", DEFAULT_REQUEST_DELAY))
    args = build_parser(default_delay).parse_args()
    if args.limit is not None and args.limit < 0:
        print("ERROR: --limit must be zero or greater.", file=sys.stderr)
        return 2
    if args.delay < 0:
        print("ERROR: --delay must be zero or greater.", file=sys.stderr)
        return 2

    email = os.environ.get("NCBI_EMAIL", "").strip()
    api_key = os.environ.get("NCBI_API_KEY", "").strip()
    if not email:
        print("ERROR: NCBI_EMAIL is required; set it in .env or the environment.", file=sys.stderr)
        return 1

    try:
        manifest_records = load_manifest_records()
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"ERROR: Could not read the PMC manifest: {error}", file=sys.stderr)
        return 1

    if args.limit is not None:
        manifest_records = manifest_records[:args.limit]

    print("========================================")
    print("PMC FULL-TEXT ACQUISITION")
    print("========================================")
    print(f"PMCID records: {len(manifest_records)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    pmcid_cache = {}
    fetched_pmcids = set()
    valid_existing_pmcids = set()

    for index, record in enumerate(manifest_records):
        pmcid = (record.get("pmcid") or "").strip()
        result = acquire_one(record, email, api_key, args.delay, pmcid_cache)
        results.append(result)
        if result["source"] == "existing_valid_xml":
            valid_existing_pmcids.add(pmcid)
        elif (
            result["source"] == "downloaded"
            and result["status"] in {"FULL_TEXT_XML", "ABSTRACT_ONLY", "RESTRICTED_XML"}
        ):
            fetched_pmcids.add(pmcid)

        if index + 1 < len(manifest_records) and result["source"] == "downloaded":
            time.sleep(args.delay)

    try:
        atomic_write_csv(RESULTS_FILE, RESULT_COLUMNS, results)
        failures = [
            result
            for result in results
            if result["status"] in {"INVALID_XML", "REQUEST_FAILED"}
        ]
        atomic_write_csv(FAILURES_FILE, FAILURE_COLUMNS, failures)
    except (OSError, csv.Error) as error:
        print(f"ERROR: Could not atomically update acquisition result files: {error}", file=sys.stderr)
        return 1

    status_counts = Counter(result["status"] for result in results)
    failure_count = status_counts["INVALID_XML"] + status_counts["REQUEST_FAILED"]
    print(f"Already valid: {len(valid_existing_pmcids)}")
    print(f"Fetched this run: {len(fetched_pmcids)}")
    for status in STATUSES:
        print(f"{status}: {status_counts[status]}")
    print(f"Failures: {failure_count}")
    print(f"Output: {RESULTS_FILE.relative_to(PROJECT_ROOT)}")
    print(f"Failures: {FAILURES_FILE.relative_to(PROJECT_ROOT)}")
    print(
        "\nStatus summary: "
        f"{len(valid_existing_pmcids)} unique PMCIDs reused, "
        f"{len(fetched_pmcids)} unique PMCIDs fetched, "
        f"{failure_count} record(s) need attention."
    )
    return 1 if failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
