import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES = (
    {
        "name": "CASE 1 - INVALID_XML",
        "pmcid": "PMC1407579",
        "expected_pmid": "29261889",
        "original_status": "INVALID_XML",
        "original_detail": "Matching PMCID XML had no body, abstract, or restriction indicator.",
    },
    {
        "name": "CASE 2 - REQUEST_FAILED",
        "pmcid": "PMC88950",
        "expected_pmid": "32644719",
        "original_status": "REQUEST_FAILED",
        "original_detail": "Original PMC EFetch request returned HTTP 400: Bad Request.",
    },
)
PMC_WEB_XML_URL = "https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/?report=xml"
BIOC_XML_URL = (
    "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/"
    "pmcoa.cgi/BioC_xml/{pmcid}/unicode"
)
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
USER_AGENT = "Symptom-RAG-Analyser/1.0 (targeted PMC exception diagnostic)"
HTTP_TIMEOUT = 30
MAX_ATTEMPTS = 2
REQUEST_DELAY = 0.34
RETRYABLE_HTTP_CODES = {408, 429}


def local_name(tag):
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1].casefold()


def text_content(element):
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def attribute(element, name):
    for key, value in element.attrib.items():
        if local_name(key) == name.casefold():
            return value
    return ""


def normalize_pmcid(value):
    value = (value or "").strip()
    if value and not value.casefold().startswith("pmc") and value.isdigit():
        return f"PMC{value}"
    return value


def redact_url(url):
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    safe_query = [
        (key, "[redacted]" if key.casefold() in {"email", "api_key"} else value)
        for key, value in query
    ]
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(safe_query), parsed.fragment)
    )


def is_transient(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code in RETRYABLE_HTTP_CODES or 500 <= error.code < 600
    return isinstance(
        error,
        (urllib.error.URLError, TimeoutError, ConnectionError, OSError),
    )


def fetch_endpoint(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
                return response.read(), getattr(response, "status", 200), attempt, None
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as error:
            last_error = error
            if attempt == MAX_ATTEMPTS or not is_transient(error):
                break
            time.sleep(REQUEST_DELAY * attempt)

    detail = ""
    if isinstance(last_error, urllib.error.HTTPError):
        try:
            detail = last_error.read(1000).decode("utf-8", errors="replace").strip()
        except OSError:
            pass
    return None, None, attempt, (last_error, detail)


def jats_article(root):
    if local_name(root.tag) == "article":
        candidates = [root]
    else:
        candidates = [element for element in root.iter() if local_name(element.tag) == "article"]
    for article in candidates:
        if any(local_name(child.tag) == "front" for child in article):
            return article
    return candidates[0] if candidates else None


def jats_metadata(article):
    if article is None:
        return None
    for front in article:
        if local_name(front.tag) == "front":
            for child in front:
                if local_name(child.tag) == "article-meta":
                    return child
    return None


def extract_jats(article):
    metadata = jats_metadata(article)
    pmcid = ""
    pmid = ""
    title = ""
    abstract = ""
    license_values = []

    if metadata is not None:
        for element in metadata.iter():
            name = local_name(element.tag)
            identifier_type = attribute(element, "pub-id-type").casefold()
            value = text_content(element)
            if name == "article-id" and identifier_type in {"pmc", "pmcid"} and not pmcid:
                pmcid = normalize_pmcid(value)
            elif name == "article-id" and identifier_type == "pmid" and not pmid:
                pmid = value
            elif name == "article-title" and not title:
                title = value
            elif name == "abstract" and not abstract:
                abstract = value
            if name in {"license", "license-p", "license-ref"}:
                for license_value in (value, attribute(element, "href")):
                    if license_value and license_value not in license_values:
                        license_values.append(license_value)

    body = next((child for child in article if local_name(child.tag) == "body"), None)
    body_present = bool(text_content(body))
    full_text = " ".join(text_content(element).casefold() for element in article.iter())
    restriction_terms = (
        "restricted access",
        "access to full text is restricted",
        "full text is not available",
        "full-text is not available",
        "not available in pubmed central",
        "does not allow downloading of the full text in xml form",
        "embargoed",
    )
    restricted = any(term in full_text for term in restriction_terms)
    restricted = restricted or any(
        "restrict" in local_name(element.tag) or "embargo" in local_name(element.tag)
        for element in article.iter()
    )
    return {
        "pmcid": pmcid,
        "pmid": pmid,
        "title": title,
        "has_abstract": bool(abstract),
        "has_body": body_present,
        "license": " | ".join(license_values),
        "restricted": restricted,
        "xml_format": "JATS",
    }


def extract_bioc(root):
    pmcid = ""
    pmid = ""
    title_parts = []
    abstract_parts = []
    body_parts = []
    license_values = []

    documents = [element for element in root.iter() if local_name(element.tag) == "document"]
    if not documents:
        return None

    for document in documents:
        for element in document.iter():
            name = local_name(element.tag)
            if name == "infon":
                key = attribute(element, "key").casefold().replace("-", "_")
                value = text_content(element)
                if key in {"pmcid", "article_id_pmc", "article_id_pmcid"} and not pmcid:
                    pmcid = normalize_pmcid(value)
                elif key in {"pmid", "article_id_pmid"} and not pmid:
                    pmid = value
                elif "license" in key and value and value not in license_values:
                    license_values.append(value)
            elif name == "passage":
                passage_text = ""
                section = ""
                for child in element:
                    child_name = local_name(child.tag)
                    if child_name == "text":
                        passage_text = text_content(child)
                    elif child_name == "infon" and attribute(child, "key").casefold() in {
                        "section_type",
                        "type",
                    }:
                        section = text_content(child).casefold()
                if passage_text:
                    if section in {"title", "article_title"}:
                        title_parts.append(passage_text)
                    elif section == "abstract":
                        abstract_parts.append(passage_text)
                    elif section in {"body", "results", "introduction", "discussion", "methods"}:
                        body_parts.append(passage_text)

    return {
        "pmcid": pmcid,
        "pmid": pmid,
        "title": " ".join(title_parts),
        "has_abstract": bool(abstract_parts),
        "has_body": bool(body_parts),
        "license": " | ".join(license_values),
        "restricted": False,
        "xml_format": "BioC XML",
    }


def extract_article(xml_bytes):
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.fromstring(xml_bytes, parser=parser)
    article = jats_article(root)
    if article is not None:
        return extract_jats(article)
    return extract_bioc(root)


def classify(data, requested_pmcid, expected_pmid):
    if data is None or not data.get("pmcid"):
        return "UNAVAILABLE"
    if data["pmcid"].casefold() != requested_pmcid.casefold():
        return "UNAVAILABLE"
    if data.get("pmid") and data["pmid"] != expected_pmid:
        return "PMCID_MATCH_BUT_PMID_MISMATCH"
    if not data.get("pmid"):
        return "UNAVAILABLE"
    if data.get("has_body"):
        return "MATCHED_FULL_TEXT"
    if data.get("has_abstract"):
        return "MATCHED_ABSTRACT_ONLY"
    return "UNAVAILABLE"


def diagnose_case(case, email, api_key):
    fetch_parameters = {
        "db": "pmc",
        "id": case["pmcid"][3:],
        "retmode": "xml",
        "rettype": "full",
        "tool": "symptom_rag_analyser",
    }
    if email:
        fetch_parameters["email"] = email
    if api_key:
        fetch_parameters["api_key"] = api_key

    endpoints = [
        (
            "NCBI PMC EFetch (numeric ID form)",
            f"{EFETCH_URL}?{urllib.parse.urlencode(fetch_parameters)}",
        ),
        ("PMC article XML", PMC_WEB_XML_URL.format(pmcid=case["pmcid"])),
    ]
    bioc_url = BIOC_XML_URL.format(pmcid=case["pmcid"])
    query = {"tool": "symptom_rag_analyser"}
    if email:
        query["email"] = email
    if api_key:
        query["api_key"] = api_key
    endpoints.append(("PMC BioC XML", f"{bioc_url}?{urllib.parse.urlencode(query)}"))

    observations = []
    final_data = None
    final_classification = "UNAVAILABLE"
    for endpoint_name, url in endpoints:
        payload, http_status, attempts, error = fetch_endpoint(url)
        observation = {
            "endpoint": endpoint_name,
            "url": redact_url(url),
            "http_status": http_status,
            "attempts": attempts,
            "error": error,
            "parse_error": "",
        }
        observations.append(observation)

        if error is not None:
            error_object, response_detail = error
            observation["error_type"] = type(error_object).__name__
            observation["error_message"] = str(error_object)
            observation["response_detail"] = response_detail
            continue

        try:
            data = extract_article(payload)
        except ET.ParseError as parse_error:
            observation["parse_error"] = str(parse_error)
            continue
        if data is None:
            observation["parse_error"] = "Response XML contains no recognized PMC/JATS or BioC article."
            continue

        observation["data"] = data
        classification = classify(data, case["pmcid"], case["expected_pmid"])
        if classification != "UNAVAILABLE":
            final_data = data
            final_classification = classification
            break
        if final_data is None:
            final_data = data

    if case["pmcid"] == "PMC88950":
        original_parameters = {
            "db": "pmc",
            "id": case["pmcid"],
            "retmode": "xml",
            "rettype": "full",
            "tool": "symptom_rag_analyser",
        }
        original_url = f"{EFETCH_URL}?{urllib.parse.urlencode(original_parameters)}"
        payload, http_status, attempts, error = fetch_endpoint(original_url)
        observation = {
            "endpoint": "Original EFetch request replay (PMC-prefixed ID)",
            "url": redact_url(original_url),
            "http_status": http_status,
            "attempts": attempts,
            "error": error,
            "parse_error": "",
        }
        if error is not None:
            error_object, response_detail = error
            observation["error_type"] = type(error_object).__name__
            observation["error_message"] = str(error_object)
            observation["response_detail"] = response_detail
        else:
            try:
                observation["data"] = extract_article(payload)
            except ET.ParseError as parse_error:
                observation["parse_error"] = str(parse_error)
        observations.append(observation)

    if final_classification == "UNAVAILABLE" and any(
        observation.get("error") is not None for observation in observations
    ) and all(observation.get("error") is not None for observation in observations):
        final_classification = "REQUEST_FAILED"

    return final_classification, final_data, observations


def print_case(case, classification, data, observations):
    print(f"\n{case['name']}")
    print(f"Requested PMCID: {case['pmcid']}")
    print(f"Expected PMID: {case['expected_pmid']}")
    print(f"Original acquisition status: {case['original_status']}")
    print(f"Original detail: {case['original_detail']}")
    for observation in observations:
        print(f"Fallback endpoint: {observation['endpoint']}")
        print(f"Endpoint URL: {observation['url']}")
        print(f"HTTP status: {observation['http_status'] or 'REQUEST_FAILED'}")
        print(f"Attempts: {observation['attempts']}")
        if observation.get("error") is not None:
            error_object, response_detail = observation["error"]
            print(f"Request error: {type(error_object).__name__}: {error_object}")
            if response_detail:
                print(f"Response detail: {response_detail[:500]}")
        if observation.get("parse_error"):
            print(f"XML diagnostic: {observation['parse_error']}")
        replay_data = observation.get("data")
        if observation["endpoint"].startswith("Original EFetch") and replay_data:
            print(f"Replay restriction indicator: {'YES' if replay_data.get('restricted') else 'NO'}")
            if replay_data.get("restricted"):
                print("Replay restriction detail: publisher XML comment indicates full-text XML download is not allowed.")

    data = data or {}
    pmcid_xml = data.get("pmcid", "")
    pmid_xml = data.get("pmid", "")
    print(f"Returned PMCID: {pmcid_xml or '(not available)'}")
    print(f"Returned PMID: {pmid_xml or '(not available)'}")
    print(f"Title: {data.get('title') or '(not available)'}")
    print(f"Abstract present: {'YES' if data.get('has_abstract') else 'NO'}")
    print(f"Body/full text present: {'YES' if data.get('has_body') else 'NO'}")
    license_text = data.get("license", "")
    print(f"License present: {'YES' if license_text else 'NO'}")
    print(f"License: {license_text or '(not available)'}")
    print(f"Restriction indicator: {'YES' if data.get('restricted') else 'NO'}")
    print(
        "PMCID_MATCH: "
        + ("YES" if pmcid_xml.casefold() == case["pmcid"].casefold() else "NO")
    )
    print(f"PMID_MATCH: {'YES' if pmid_xml == case['expected_pmid'] else 'NO'}")
    print(f"CLASSIFICATION: {classification}")
    if case["pmcid"] == "PMC88950":
        print(
            "HTTP 400 note: the original acquisition log retained only the status and "
            "reason, not the NCBI response body. This fallback can test PMC access and "
            "identity, but cannot reconstruct the original server's exact explanation."
        )


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    email = os.environ.get("NCBI_EMAIL", "").strip()
    api_key = os.environ.get("NCBI_API_KEY", "").strip()

    print("PMC EXCEPTION FALLBACK DIAGNOSTIC")
    print("Only the two requested PMCIDs will be queried; responses remain in memory.")
    if not email:
        print("NCBI_EMAIL not set; proceeding with the targeted public PMC endpoints.")

    all_classifications = []
    for index, case in enumerate(CASES):
        if index:
            time.sleep(REQUEST_DELAY)
        classification, data, observations = diagnose_case(case, email, api_key)
        print_case(case, classification, data, observations)
        all_classifications.append(classification)

    print("\nDIAGNOSTIC SUMMARY")
    for case, classification in zip(CASES, all_classifications):
        print(f"{case['pmcid']}: {classification}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
