import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_fulltext_manifest.csv"
PMC_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "pmc"
TARGET_PMCIDS = ("PMC6376932", "PMC3229952", "PMC6265087")
REQUIRED_COLUMNS = {"pmid", "pmcid", "doi", "title"}


def normalized(value):
    return " ".join((value or "").split()).casefold()


def normalize_doi(value):
    doi = (value or "").strip().casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
            break
    return doi.strip()


def first_text(root, tag, attribute=None, attribute_value=None):
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != tag:
            continue
        if attribute is not None and element.attrib.get(attribute) != attribute_value:
            continue
        text = " ".join("".join(element.itertext()).split())
        if text:
            return text
    return ""


def has_element(root, tag):
    return any(element.tag.rsplit("}", 1)[-1] == tag for element in root.iter())


def read_manifest_rows():
    with MANIFEST_FILE.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Manifest is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
        return list(reader)


def get_manifest_row(rows, pmcid):
    matches = [row for row in rows if (row.get("pmcid") or "").strip() == pmcid]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest row for {pmcid}, found {len(matches)}.")
    return matches[0]


def display(value):
    return value if value else "(not available)"


def inspect_record(rows, pmcid):
    manifest = get_manifest_row(rows, pmcid)
    xml_path = PMC_DIRECTORY / f"{pmcid}.xml"
    root = ET.parse(xml_path).getroot()

    manifest_pmid = (manifest.get("pmid") or "").strip()
    manifest_pmcid = (manifest.get("pmcid") or "").strip()
    manifest_title = (manifest.get("title") or "").strip()
    manifest_doi = (manifest.get("doi") or "").strip()
    xml_pmid = first_text(root, "article-id", "pub-id-type", "pmid")
    xml_pmcid = first_text(root, "article-id", "pub-id-type", "pmcid")
    xml_title = first_text(root, "article-title")
    xml_doi = first_text(root, "article-id", "pub-id-type", "doi")
    has_body = has_element(root, "body")
    has_abstract = has_element(root, "abstract")

    pmid_match = bool(manifest_pmid and xml_pmid and manifest_pmid == xml_pmid)
    pmcid_match = bool(manifest_pmcid and xml_pmcid and manifest_pmcid == xml_pmcid)
    title_match = bool(manifest_title and xml_title and normalized(manifest_title) == normalized(xml_title))
    if manifest_doi and xml_doi:
        doi_match = "YES" if normalize_doi(manifest_doi) == normalize_doi(xml_doi) else "NO"
    else:
        doi_match = "NOT_AVAILABLE"

    print(f"\n{'=' * 72}\n{pmcid}\n{'=' * 72}")
    print(f"Manifest PMID:  {display(manifest_pmid)}")
    print(f"XML PMID:       {display(xml_pmid)}")
    print(f"Manifest PMCID: {display(manifest_pmcid)}")
    print(f"XML PMCID:      {display(xml_pmcid)}")
    print(f"Manifest title: {display(manifest_title)}")
    print(f"XML title:      {display(xml_title)}")
    print(f"Manifest DOI:   {display(manifest_doi)}")
    print(f"XML DOI:        {display(xml_doi)}")
    print(f"XML contains <body>:     {'YES' if has_body else 'NO'}")
    print(f"XML contains abstract:   {'YES' if has_abstract else 'NO'}")
    print(f"PMID_MATCH: {'YES' if pmid_match else 'NO'}")
    print(f"PMCID_MATCH: {'YES' if pmcid_match else 'NO'}")
    print(f"TITLE_MATCH: {'YES' if title_match else 'NO'}")
    print(f"DOI_MATCH: {doi_match}")

    metadata_matches = pmid_match and title_match and doi_match != "NO"
    if has_abstract and not has_body:
        conclusion = "ABSTRACT_ONLY"
    elif pmcid_match and metadata_matches:
        conclusion = "CONSISTENT"
    elif pmcid_match:
        conclusion = "PMCID_MATCH_BUT_METADATA_MISMATCH"
    else:
        conclusion = "OTHER_MISMATCH"
    return conclusion


def main():
    try:
        manifest_rows = read_manifest_rows()
        conclusions = [
            (pmcid, inspect_record(manifest_rows, pmcid))
            for pmcid in TARGET_PMCIDS
        ]
    except (OSError, UnicodeError, csv.Error, ET.ParseError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"\n{'=' * 72}\nDIAGNOSTIC CONCLUSIONS\n{'=' * 72}")
    for pmcid, conclusion in conclusions:
        print(f"{pmcid}: {conclusion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
