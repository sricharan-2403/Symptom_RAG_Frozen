import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_raw_response_20301288.xml"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMID = "20301288"
TOOL_NAME = "symptom_rag_analyser"
HTTP_TIMEOUT = 30
SEARCH_TERMS = (
    "20301288",
    "PubmedArticle",
    "PubmedBookArticle",
    "ErrorList",
    "ERROR",
)


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


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
        print(f"ERROR: Refusing to overwrite existing file: {OUTPUT_FILE}", file=sys.stderr)
        return 1

    parameters = {
        "db": "pubmed",
        "id": PMID,
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
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            status_code = response.status
            content_type = response.headers.get("Content-Type", "")
            raw_bytes = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        print(f"ERROR: EFetch request failed: {error}", file=sys.stderr)
        return 1

    try:
        raw_xml = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        print(f"ERROR: Response could not be decoded as UTF-8: {error}", file=sys.stderr)
        return 1

    print(f"HTTP status code: {status_code}")
    print(f"Content-Type: {content_type}")
    print(f"Raw response byte length: {len(raw_bytes)}")
    print("\nFirst 3000 characters of raw XML:")
    print(raw_xml[:3000])

    root = None
    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError as error:
        print(f"\nXML parse error: {error}")

    if root is not None:
        children = [child.tag for child in list(root)]
        elements = list(root.iter())
        print(f"\nRoot tag: {root.tag}")
        print(f"Root attributes: {root.attrib}")
        print(f"Immediate child tags: {children}")
        for tag in ("PubmedArticle", "PubmedBookArticle", "PubmedArticleSet", "ErrorList", "ERROR"):
            count = sum(local_name(element.tag) == tag for element in elements)
            print(f"Count of {tag} elements: {count}")

        messages = []
        for element in elements:
            tag = local_name(element.tag).casefold()
            text = " ".join("".join(element.itertext()).split())
            if text and ("error" in tag or "message" in tag):
                messages.append((element.tag, text))
        if messages:
            print("\nNCBI error/message text:")
            for tag, text in messages:
                print(f"{tag}: {text}")
        else:
            print("\nNCBI error/message text: (none detected)")
    else:
        print("\nXML structure counts unavailable because parsing failed.")
        print("NCBI error/message text candidates:")
        for line in raw_xml.splitlines():
            if "error" in line.casefold() or "message" in line.casefold():
                print(line.strip())

    print("\nRaw XML string searches:")
    for term in SEARCH_TERMS:
        print(f'{term}: {"found" if term in raw_xml else "not found"}')

    try:
        with OUTPUT_FILE.open("xb") as output_file:
            output_file.write(raw_bytes)
            output_file.flush()
            os.fsync(output_file.fileno())
    except OSError as error:
        print(f"ERROR: Could not save raw XML without overwriting: {error}", file=sys.stderr)
        return 1

    print(f"\nRaw XML saved to:\n{OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())