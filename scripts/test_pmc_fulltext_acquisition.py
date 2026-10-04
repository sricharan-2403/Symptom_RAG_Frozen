import csv
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_fulltext_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "pmc"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TOOL_NAME = "symptom_rag_analyser"

REQUESTED_COUNT = 5
REQUEST_DELAY = 0.5
HTTP_TIMEOUT = 30
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 1.0
USER_AGENT = "Symptom-RAG-Analyser/1.0 (PMC full-text acquisition test)"


def text_content(element):
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def local_name(tag):
    return tag.rsplit("}", 1)[-1].casefold()


def load_test_records():
    if not MANIFEST_FILE.is_file():
        raise FileNotFoundError(f"Full-text manifest not found: {MANIFEST_FILE}")
    with MANIFEST_FILE.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required_columns = {"pmid", "pmcid", "acquisition_route"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Manifest is missing required columns: " + ", ".join(sorted(missing))
            )
        return [
            row for row in reader
            if (row.get("acquisition_route") or "").strip() == "pmc"
        ][:REQUESTED_COUNT]


def is_retryable(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 429) or 500 <= error.code < 600
    return isinstance(error, (urllib.error.URLError, TimeoutError, OSError))


def error_details(error):
    if isinstance(error, urllib.error.HTTPError):
        return f"HTTP {error.code}" + (f": {error.reason}" if error.reason else "")
    if isinstance(error, urllib.error.URLError):
        return str(error.reason)
    return str(error)


def fetch_xml(pmcid):
    parameters = {
        "db": "pmc",
        "id": pmcid,
        "retmode": "xml",
        "rettype": "full",
        "tool": TOOL_NAME,
    }
    request = urllib.request.Request(
        f"{EFETCH_URL}?{urllib.parse.urlencode(parameters)}",
        headers={"User-Agent": USER_AGENT},
    )

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
                status = getattr(response, "status", 200)
                content_type = response.headers.get("Content-Type", "")
                xml_bytes = response.read()
            return status, content_type, xml_bytes, None
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as error:
            if attempt > MAX_RETRIES or not is_retryable(error):
                return None, "", b"", error
            delay = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            print(
                f"  Retry {attempt}/{MAX_RETRIES} for {pmcid} after "
                f"{type(error).__name__}; waiting {delay:.1f}s."
            )
            time.sleep(delay)

    return None, "", b"", RuntimeError("Request attempts exhausted.")


def looks_like_xml(content_type, xml_bytes):
    content_type = (content_type or "").casefold()
    if "xml" in content_type:
        return True
    if "html" in content_type:
        return False
    return xml_bytes.lstrip().startswith((b"<?xml", b"<"))


def identify_pmcid(root):
    for element in root.iter():
        name = local_name(element.tag)
        pub_id_type = next(
            (
                value.casefold()
                for key, value in element.attrib.items()
                if local_name(key) == "pub-id-type"
            ),
            "",
        )
        if name in {"pmcid", "pmc-id"} or (name == "article-id" and pub_id_type in {"pmc", "pmcid"}):
            value = text_content(element)
            if value:
                return value if value.casefold().startswith("pmc") else f"PMC{value}"
    return ""


def identify_pmid(root):
    for element in root.iter():
        name = local_name(element.tag)
        pub_id_type = next(
            (
                value.casefold()
                for key, value in element.attrib.items()
                if local_name(key) == "pub-id-type"
            ),
            "",
        )
        if name == "pmid" or (name == "article-id" and pub_id_type == "pmid"):
            value = text_content(element)
            if value:
                return value
    return ""


def detect_restriction(root):
    restriction_phrases = (
        "not available in pmc",
        "not available in pubmed central",
        "full text is not available",
        "full-text is not available",
        "access to full text is restricted",
        "embargoed",
    )
    for element in root.iter():
        name = re.sub(r"[-_]", " ", local_name(element.tag))
        if re.search(r"\b(?:restricted|restriction|embargo)\b", name):
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
        if any(phrase in content for phrase in restriction_phrases):
            return True
        if re.search(r"\brestricted\s+access\b", content):
            return True
    return False


def detect_license(root):
    return any(
        local_name(element.tag) in {"license", "license-p", "license-ref"}
        or any("license" in local_name(key) for key in element.attrib)
        for element in root.iter()
    )


def validate_xml(xml_bytes, expected_pmcid, expected_pmid):
    root = ET.fromstring(xml_bytes)
    articles = [element for element in root.iter() if local_name(element.tag) == "article"]
    identified_pmcid = ""
    matching_article = None
    for article in articles:
        article_pmcid = identify_pmcid(article)
        if not identified_pmcid and article_pmcid:
            identified_pmcid = article_pmcid
        if article_pmcid.casefold() == expected_pmcid.casefold():
            matching_article = article
            identified_pmcid = article_pmcid
            break
    if not identified_pmcid:
        identified_pmcid = identify_pmcid(root)
    validation_scope = matching_article if matching_article is not None else (articles[0] if articles else root)
    pmcid_pass = identified_pmcid.casefold() == expected_pmcid.casefold()
    identified_pmid = identify_pmid(validation_scope)
    pmid_validation = (
        "NOT_AVAILABLE" if not identified_pmid
        else "PASS" if identified_pmid == expected_pmid
        else "FAIL"
    )
    title = ""
    abstract_present = False
    body_present = False
    for element in validation_scope.iter():
        if local_name(element.tag) == "article-title":
            title = text_content(element)
        elif local_name(element.tag) == "abstract" and text_content(element):
            abstract_present = True
        elif local_name(element.tag) == "body":
            body_present = True

    restriction_detected = detect_restriction(validation_scope)
    license_detected = detect_license(validation_scope)
    if not pmcid_pass:
        classification = "PMCID_MISMATCH"
    elif title and body_present:
        classification = "FULL_TEXT_XML"
    elif title and not body_present and restriction_detected:
        classification = "RESTRICTED_XML"
    elif title and abstract_present and not body_present:
        classification = "ABSTRACT_ONLY"
    else:
        classification = "UNCLASSIFIED_XML"

    return {
        "xml_parse": True,
        "pmcid_validation": pmcid_pass,
        "identified_pmcid": identified_pmcid,
        "identified_pmid": identified_pmid,
        "pmid_validation": pmid_validation,
        "title_present": bool(title),
        "title": title,
        "abstract_present": abstract_present,
        "body_present": body_present,
        "restriction_detected": restriction_detected,
        "license_detected": license_detected,
        "classification": classification,
    }


def save_without_overwrite(path, xml_bytes):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output_file:
        output_file.write(xml_bytes)
        output_file.flush()
        os.fsync(output_file.fileno())


def test_article(record):
    pmid = (record.get("pmid") or "").strip()
    pmcid = (record.get("pmcid") or "").strip()
    output_path = OUTPUT_DIR / f"{pmcid}.xml" if pmcid else None
    source = "existing_file" if output_path is not None and output_path.exists() else "downloaded"
    result = {
        "pmid": pmid,
        "pmcid": pmcid,
        "source": source,
        "http_status": "N/A",
        "xml_parse": False,
        "pmcid_validation": False,
        "pmid_validation": "N/A",
        "title_present": False,
        "abstract_present": False,
        "body_present": False,
        "restriction_detected": False,
        "license_detected": False,
        "classification": "REQUEST_FAILED",
        "saved_path": "",
        "errors": [],
    }

    if not pmid or not pmcid:
        result["errors"].append("Manifest row has an empty PMID or PMCID.")
        return result
    if not pmcid.upper().startswith("PMC") or not pmcid[3:].isdigit():
        result["errors"].append(f"Invalid PMCID format: {pmcid}")
        return result

    if source == "existing_file":
        result["saved_path"] = str(output_path.relative_to(PROJECT_ROOT))
        try:
            xml_bytes = output_path.read_bytes()
        except OSError as error:
            result["errors"].append(f"Could not read existing XML: {error}")
            return result
    else:
        status, content_type, xml_bytes, request_error = fetch_xml(pmcid)
        if request_error is not None:
            result["errors"].append(f"{type(request_error).__name__}: {error_details(request_error)}")
            return result

        result["http_status"] = status
        if not 200 <= status < 300:
            result["errors"].append(f"Unexpected HTTP status: {status}")
            return result

        if not looks_like_xml(content_type, xml_bytes):
            result["classification"] = "INVALID_XML"
            result["errors"].append(f"Response is not XML (Content-Type: {content_type or 'missing'}).")
            return result

        try:
            save_without_overwrite(output_path, xml_bytes)
            result["saved_path"] = str(output_path.relative_to(PROJECT_ROOT))
        except OSError as error:
            result["errors"].append(f"Could not save downloaded XML without overwriting: {error}")
            return result

    try:
        validation = validate_xml(xml_bytes, pmcid, pmid)
    except ET.ParseError as error:
        result["classification"] = "INVALID_XML"
        result["errors"].append(f"XML parse failed: {error}")
        return result

    result.update({
        "xml_parse": validation["xml_parse"],
        "pmcid_validation": validation["pmcid_validation"],
        "pmid_validation": validation["pmid_validation"],
        "title_present": validation["title_present"],
        "abstract_present": validation["abstract_present"],
        "body_present": validation["body_present"],
        "restriction_detected": validation["restriction_detected"],
        "license_detected": validation["license_detected"],
        "classification": validation["classification"],
    })
    if not validation["pmcid_validation"]:
        result["errors"].append(
            f"Requested {pmcid}; XML identified {validation['identified_pmcid'] or '(none)'}."
        )
    if validation["pmid_validation"] == "FAIL":
        result["errors"].append(
            f"Requested PMID {pmid}; XML identified PMID {validation['identified_pmid']}."
        )
    if not validation["title_present"]:
        result["errors"].append("Article title is missing.")
    return result


def report_result(result):
    print(f"\nPMID: {result['pmid'] or '(missing)'}")
    print(f"PMCID: {result['pmcid'] or '(missing)'}")
    print(f"Source: {result['source']}")
    print(f"HTTP status: {result['http_status']}")
    print(f"XML parse: {'PASS' if result['xml_parse'] else 'FAIL'}")
    print(f"PMCID validation: {'PASS' if result['pmcid_validation'] else 'FAIL'}")
    pmid_status = result["pmid_validation"]
    print(f"PMID validation: {'PASS' if pmid_status == 'PASS' else 'FAIL' if pmid_status == 'FAIL' else 'N/A'}")
    print(f"Title present: {'PASS' if result['title_present'] else 'FAIL'}")
    print(f"Abstract present: {'PASS' if result['abstract_present'] else 'FAIL'}")
    print(f"Body present: {'PASS' if result['body_present'] else 'FAIL'}")
    print(f"Restriction detected: {'YES' if result['restriction_detected'] else 'NO'}")
    print(f"License detected: {'YES' if result['license_detected'] else 'NO'}")
    print(f"Classification: {result['classification']}")
    print(f"Saved path: {result['saved_path'] or '(not saved)'}")
    for error in result["errors"]:
        print(f"Error: {error}")


def main():
    try:
        records = load_test_records()
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"ERROR: Could not read PMC manifest: {error}", file=sys.stderr)
        return 1

    results = []
    print("PMC FULL-TEXT ACQUISITION TEST")
    print("===============================")
    print(f"\nRequested: {REQUESTED_COUNT}")

    for index, record in enumerate(records):
        if index:
            time.sleep(REQUEST_DELAY)
        result = test_article(record)
        results.append(result)
        report_result(result)

    classification_counts = Counter(result["classification"] for result in results)
    for classification in (
        "FULL_TEXT_XML",
        "ABSTRACT_ONLY",
        "RESTRICTED_XML",
        "INVALID_XML",
        "REQUEST_FAILED",
    ):
        print(f"{classification}: {classification_counts[classification]}")
    extra_classifications = {
        name: count
        for name, count in classification_counts.items()
        if name not in {
            "FULL_TEXT_XML",
            "ABSTRACT_ONLY",
            "RESTRICTED_XML",
            "INVALID_XML",
            "REQUEST_FAILED",
        }
    }
    for name, count in sorted(extra_classifications.items()):
        print(f"{name}: {count}")

    successful = sum(
        result["xml_parse"]
        and result["classification"] not in {"INVALID_XML", "REQUEST_FAILED", "PMCID_MISMATCH", "UNCLASSIFIED_XML"}
        and result["pmcid_validation"]
        and result["pmid_validation"] != "FAIL"
        and result["title_present"]
        and bool(result["saved_path"])
        for result in results
    )
    failed = REQUESTED_COUNT - successful
    if len(records) != REQUESTED_COUNT:
        print(f"\nERROR: Only {len(records)} PMC-routed records were available; expected 5.")

    print(f"\nSuccessful: {successful}")
    print(f"Failed: {failed}")
    if successful == REQUESTED_COUNT and len(records) == REQUESTED_COUNT:
        print("\nTEST PASSED")
        return 0
    print("\nTEST FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())