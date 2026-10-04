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
from pathlib import Path
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_candidate_articles.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_article_metadata.csv"
FAILURE_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_metadata_failures.csv"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TOOL_NAME = "symptom_rag_analyser"

BATCH_SIZE = 200
REQUEST_DELAY = 0.34
HTTP_TIMEOUT = 60
MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 1.0

PROVENANCE_COLUMNS = (
    "disease_ids",
    "disease_names",
    "query_categories",
    "queries",
    "best_rank",
    "match_counts",
)
INPUT_COLUMNS = ("pmid",) + PROVENANCE_COLUMNS
OUTPUT_COLUMNS = (
    "pmid",
    "record_type",
    "pmcid",
    "doi",
    "title",
    "abstract",
    "journal",
    "publication_date",
    "publication_year",
    "publication_types",
    "language",
    "authors",
    "mesh_terms",
    "book_title",
    "publisher",
    "disease_ids",
    "disease_names",
    "query_categories",
    "queries",
    "best_rank",
    "match_counts",
    "has_pmc",
)
FAILURE_COLUMNS = (
    "batch_number",
    "pmids",
    "error_type",
    "error_message",
    "attempts",
)
MONTHS = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}


def element_text(element):
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def child_text(element, path):
    return element_text(element.find(path)) if element is not None else ""


def publication_date_text(date_element):
    if date_element is None:
        return ""

    medline_date = child_text(date_element, "MedlineDate")
    year = child_text(date_element, "Year")
    if not year:
        return medline_date

    month = child_text(date_element, "Month")
    day = child_text(date_element, "Day")
    season = child_text(date_element, "Season")
    if month:
        normalized_month = MONTHS.get(month[:3].casefold(), month)
        if normalized_month.isdigit() and day.isdigit():
            return f"{year}-{normalized_month}-{int(day):02d}"
        if normalized_month.isdigit():
            return f"{year}-{normalized_month}"
        return f"{year} {month}" + (f" {day}" if day else "")
    if season:
        return f"{year} {season}"
    return year


def extract_publication_date(article):
    pub_date = article.find("Journal/JournalIssue/PubDate")
    article_date = article.find("ArticleDate")

    for date_element in (pub_date, article_date):
        if date_element is not None and child_text(date_element, "Year"):
            return publication_date_text(date_element)
    for date_element in (pub_date, article_date):
        if date_element is not None:
            medline_date = child_text(date_element, "MedlineDate")
            if medline_date:
                return medline_date
    return ""


def extract_publication_year(publication_date):
    match = re.search(r"\b(\d{4})\b", publication_date)
    return match.group(1) if match else ""


def extract_abstract(article):
    abstract = article.find("Abstract")
    if abstract is None:
        return ""

    sections = []
    for section in abstract.findall("AbstractText"):
        text = element_text(section)
        if not text:
            continue
        label = section.get("Label", "").strip()
        sections.append(f"{label}: {text}" if label else text)
    return "\n".join(sections)


def extract_authors(article):
    authors = []
    for author in article.findall("AuthorList/Author"):
        collective_name = child_text(author, "CollectiveName")
        if collective_name:
            authors.append(collective_name)
            continue

        fore_name = child_text(author, "ForeName") or child_text(author, "Initials")
        last_name = child_text(author, "LastName")
        suffix = child_text(author, "Suffix")
        name = " ".join(part for part in (fore_name, last_name, suffix) if part)
        if name:
            authors.append(name)
    return "|".join(authors)


def extract_identifiers(pubmed_article, article):
    pmcid = ""
    doi = ""
    id_list = pubmed_article.findall("PubmedData/ArticleIdList/ArticleId")
    for identifier in id_list:
        value = element_text(identifier)
        identifier_type = identifier.get("IdType", "").casefold()
        if identifier_type == "pmc" and not pmcid:
            pmcid = value if value.upper().startswith("PMC") else f"PMC{value}"
        elif identifier_type == "doi" and not doi:
            doi = value

    for identifier in article.findall("ELocationID"):
        if identifier.get("EIdType", "").casefold() == "doi" and not doi:
            doi = element_text(identifier)
    return pmcid, doi


def parse_pubmed_article(pubmed_article):
    medline_citation = pubmed_article.find("MedlineCitation")
    article = medline_citation.find("Article") if medline_citation is not None else None
    if article is None:
        raise ValueError("PubmedArticle is missing MedlineCitation/Article")

    pmid = child_text(medline_citation, "PMID")
    if not pmid:
        raise ValueError("PubmedArticle is missing PMID")

    pmcid, doi = extract_identifiers(pubmed_article, article)
    publication_date = extract_publication_date(article)
    publication_types = [
        element_text(item) for item in article.findall("PublicationTypeList/PublicationType")
    ]
    languages = [element_text(item) for item in article.findall("Language")]
    mesh_terms = []
    for heading in medline_citation.findall("MeshHeadingList/MeshHeading"):
        descriptor = heading.find("DescriptorName")
        term = element_text(descriptor)
        if term:
            is_major = descriptor is not None and descriptor.get("MajorTopicYN") == "Y"
            mesh_terms.append(f"{term} [major]" if is_major else term)

    journal = child_text(article, "Journal/Title")
    return {
        "pmid": pmid,
        "record_type": "pubmed_article",
        "pmcid": pmcid,
        "doi": doi,
        "title": child_text(article, "ArticleTitle"),
        "abstract": extract_abstract(article),
        "journal": journal,
        "publication_date": publication_date,
        "publication_year": extract_publication_year(publication_date),
        "publication_types": "|".join(value for value in publication_types if value),
        "language": "|".join(value for value in languages if value),
        "authors": extract_authors(article),
        "mesh_terms": "|".join(mesh_terms),
        "book_title": "",
        "publisher": "",
        "has_pmc": "true" if pmcid else "false",
    }


def parse_pubmed_book_article(pubmed_book_article):
    book_document = pubmed_book_article.find("BookDocument")
    if book_document is None:
        raise ValueError("PubmedBookArticle is missing BookDocument")

    pmid = child_text(book_document, "PMID")
    if not pmid:
        raise ValueError("PubmedBookArticle is missing PMID")

    book = book_document.find("Book")
    book_title = child_text(book, "BookTitle")
    title = child_text(book_document, "ArticleTitle") or book_title

    pmcid = ""
    doi = ""
    for identifier in book_document.findall("ArticleIdList/ArticleId"):
        value = element_text(identifier)
        identifier_type = identifier.get("IdType", "").casefold()
        if identifier_type == "pmc" and not pmcid:
            pmcid = value if value.upper().startswith("PMC") else f"PMC{value}"
        elif identifier_type == "doi" and not doi:
            doi = value
    for identifier in book_document.findall(".//ELocationID"):
        if identifier.get("EIdType", "").casefold() == "doi" and not doi:
            doi = element_text(identifier)

    date_element = book.find("PubDate") if book is not None else None
    if date_element is None:
        date_element = book_document.find("ArticleDate")
    publication_date = publication_date_text(date_element)

    publication_types = [
        element_text(item)
        for item in book_document.findall("PublicationTypeList/PublicationType")
    ]
    publication_types.extend(
        element_text(item) for item in book_document.findall("PublicationType")
    )
    languages = [element_text(item) for item in book_document.findall("Language")]
    mesh_terms = []
    for heading in book_document.findall("MeshHeadingList/MeshHeading"):
        descriptor = heading.find("DescriptorName")
        term = element_text(descriptor)
        if term:
            is_major = descriptor is not None and descriptor.get("MajorTopicYN") == "Y"
            mesh_terms.append(f"{term} [major]" if is_major else term)

    publisher = child_text(book, "Publisher/PublisherName")
    return {
        "pmid": pmid,
        "record_type": "pubmed_book_article",
        "pmcid": pmcid,
        "doi": doi,
        "title": title,
        "abstract": extract_abstract(book_document),
        "journal": "",
        "publication_date": publication_date,
        "publication_year": extract_publication_year(publication_date),
        "publication_types": "|".join(value for value in publication_types if value),
        "language": "|".join(value for value in languages if value),
        "authors": extract_authors(book_document),
        "mesh_terms": "|".join(mesh_terms),
        "book_title": book_title,
        "publisher": publisher,
        "has_pmc": "true" if pmcid else "false",
    }


def merge_candidate_rows(existing, incoming):
    merged = dict(existing)
    for column in PROVENANCE_COLUMNS:
        delimiter = " || " if column == "queries" else "|"
        values = []
        for row in (existing, incoming):
            for value in row.get(column, "").split(delimiter):
                value = value.strip()
                if value and value not in values:
                    values.append(value)
        merged[column] = delimiter.join(values)

    ranks = [
        value.strip()
        for row in (existing, incoming)
        if (value := row.get("best_rank", "").strip()).isdigit()
    ]
    if ranks:
        merged["best_rank"] = str(min(map(int, ranks)))
    return merged


def load_candidates():
    candidates = {}
    with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        headers = {header.strip() for header in (reader.fieldnames or []) if header}
        missing = sorted(set(INPUT_COLUMNS) - headers)
        if missing:
            raise ValueError("Candidate CSV is missing required columns: " + ", ".join(missing))

        for row_number, raw_row in enumerate(reader, start=2):
            row = {key.strip(): (value or "").strip() for key, value in raw_row.items() if key}
            pmid = row.get("pmid", "")
            if not pmid:
                print(f"WARNING: Skipping candidate row {row_number} with an empty PMID.")
                continue
            if pmid in candidates:
                candidates[pmid] = merge_candidate_rows(candidates[pmid], row)
                print(f"WARNING: Merged duplicate candidate provenance for PMID {pmid}.")
            else:
                candidates[pmid] = row
    return candidates


def load_existing_records():
    if not OUTPUT_FILE.exists():
        return {}

    records = {}
    duplicate_found = False
    with OUTPUT_FILE.open("r", encoding="utf-8-sig", newline="") as output_file:
        reader = csv.DictReader(output_file)
        if reader.fieldnames != list(OUTPUT_COLUMNS):
            raise ValueError(
                f"Existing output has an unexpected schema: {OUTPUT_FILE}. "
                "Move it aside or restore the expected column order before resuming."
            )
        for raw_row in reader:
            row = {column: (raw_row.get(column) or "") for column in OUTPUT_COLUMNS}
            pmid = row["pmid"].strip()
            if pmid:
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
                dir=OUTPUT_FILE.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                writer = csv.DictWriter(temporary_file, fieldnames=OUTPUT_COLUMNS)
                writer.writeheader()
                writer.writerows(records.values())
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            temporary_path.replace(OUTPUT_FILE)
            print("WARNING: Removed duplicate PMID rows from the existing metadata output.")
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
    return records


def write_successful_records(records):
    if not records:
        return
    is_new_file = not OUTPUT_FILE.exists() or OUTPUT_FILE.stat().st_size == 0
    with OUTPUT_FILE.open("a", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        if is_new_file:
            writer.writeheader()
        writer.writerows(records)
        output_file.flush()
        os.fsync(output_file.fileno())


def append_failure(batch_number, pmids, error, attempts):
    is_new_file = not FAILURE_FILE.exists() or FAILURE_FILE.stat().st_size == 0
    with FAILURE_FILE.open("a", encoding="utf-8", newline="") as failure_file:
        writer = csv.DictWriter(failure_file, fieldnames=FAILURE_COLUMNS)
        if is_new_file:
            writer.writeheader()
        writer.writerow({
            "batch_number": batch_number,
            "pmids": ",".join(pmids),
            "error_type": type(error).__name__,
            "error_message": str(error).replace("\r", " ").replace("\n", " "),
            "attempts": attempts,
        })
        failure_file.flush()
        os.fsync(failure_file.fileno())


def fetch_batch(pmids, email, api_key):
    parameters = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "tool": TOOL_NAME,
        "email": email,
    }
    if api_key:
        parameters["api_key"] = api_key

    url = f"{EFETCH_URL}?{urllib.parse.urlencode(parameters)}"
    request = urllib.request.Request(url, headers={"User-Agent": TOOL_NAME})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        xml_data = response.read()
    root = ET.fromstring(xml_data)

    fetched = {}
    for pubmed_article in root.iter("PubmedArticle"):
        record = parse_pubmed_article(pubmed_article)
        if record["pmid"] in pmids:
            fetched.setdefault(record["pmid"], record)

    for pubmed_book_article in root.iter("PubmedBookArticle"):
        record = parse_pubmed_book_article(pubmed_book_article)
        if record["pmid"] in pmids:
            fetched.setdefault(record["pmid"], record)
    return fetched


def is_retryable(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 429) or 500 <= error.code < 600
    return isinstance(error, (urllib.error.URLError, TimeoutError, ET.ParseError))


def fetch_with_retries(pmids, email, api_key):
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            records = fetch_batch(pmids, email, api_key)
            return records, attempt, None
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ET.ParseError) as error:
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
    return {}, attempt, last_error


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    email = os.environ.get("NCBI_EMAIL", "").strip()
    if not email:
        print(
            "ERROR: NCBI_EMAIL is required by NCBI E-utilities usage guidance. "
            "Set NCBI_EMAIL in the environment before running this script.",
            file=sys.stderr,
        )
        return 1

    api_key = os.environ.get("NCBI_API_KEY", "").strip()
    try:
        candidates = load_candidates()
        existing = load_existing_records()
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    remaining = [pmid for pmid in candidates if pmid not in existing]
    batches = [remaining[index:index + BATCH_SIZE] for index in range(0, len(remaining), BATCH_SIZE)]
    fetched_this_run = 0
    failed_pmids = set()
    failed_batches = 0

    print("=" * 72)
    print("PUBMED METADATA ACQUISITION")
    print("=" * 72)
    print(f"Candidate PMIDs: {len(candidates)}")
    print(f"Already present: {len(set(candidates) & set(existing))}")
    print(f"Remaining PMIDs: {len(remaining)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"NCBI API key configured: {'yes' if api_key else 'no'}")

    for batch_number, batch_pmids in enumerate(batches, start=1):
        print(f"\nBatch {batch_number}/{len(batches)}: {len(batch_pmids)} PMIDs")
        fetched, attempts, error = fetch_with_retries(batch_pmids, email, api_key)

        new_records = []
        for pmid, metadata in fetched.items():
            metadata.update({column: candidates[pmid].get(column, "") for column in PROVENANCE_COLUMNS})
            new_records.append(metadata)
        if new_records:
            write_successful_records(new_records)
            existing.update({record["pmid"]: record for record in new_records})
            fetched_this_run += len(new_records)

        missing_pmids = [pmid for pmid in batch_pmids if pmid not in fetched]
        if error is not None:
            failed_batches += 1
            failed_pmids.update(missing_pmids)
            append_failure(batch_number, missing_pmids, error, attempts)
            print(
                f"  FAILED after {attempts} attempt(s): {type(error).__name__}: {error}"
            )
        elif missing_pmids:
            failed_batches += 1
            failed_pmids.update(missing_pmids)
            missing_error = RuntimeError(
                "EFetch response did not contain requested PMID(s): "
                + ",".join(missing_pmids)
            )
            append_failure(batch_number, missing_pmids, missing_error, attempts)
            print(f"  WARNING: {len(missing_pmids)} PMID(s) were absent from the response.")
        else:
            print(f"  Stored {len(new_records)} metadata record(s).")

        if batch_number < len(batches):
            time.sleep(REQUEST_DELAY)

    stored_records = list(existing.values())
    print("\n[Metadata coverage]")
    print(f"PMIDs with title: {sum(bool(row.get('title', '').strip()) for row in stored_records)}")
    print(f"PMIDs with abstract: {sum(bool(row.get('abstract', '').strip()) for row in stored_records)}")
    print(f"PMIDs with PMCID: {sum(bool(row.get('pmcid', '').strip()) for row in stored_records)}")
    print(f"PMIDs with DOI: {sum(bool(row.get('doi', '').strip()) for row in stored_records)}")
    print(f"PMIDs with MeSH terms: {sum(bool(row.get('mesh_terms', '').strip()) for row in stored_records)}")

    print("\n[Run summary]")
    print(f"Candidate PMIDs: {len(candidates)}")
    print(f"Fetched this run: {fetched_this_run}")
    print(f"Already present: {len(set(candidates) & set(existing)) - fetched_this_run}")
    print(f"Successfully stored: {len(existing)}")
    print(f"Failed PMIDs/batches: {len(failed_pmids)} PMIDs across {failed_batches} batch(es)")
    print(f"\nOutput: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    print(f"Failures: {FAILURE_FILE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())