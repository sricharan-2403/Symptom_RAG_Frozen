import os
import sys

from dotenv import load_dotenv

from fetch_pubmed_metadata import (
    PROJECT_ROOT,
    fetch_batch,
    parse_pubmed_book_article,
)


REQUESTED_PMIDS = ("28613483", "28613486", "28613499")


def display(value):
    return value if value else "(not available)"


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    email = os.environ.get("NCBI_EMAIL", "").strip()
    api_key = os.environ.get("NCBI_API_KEY", "").strip()
    if not email:
        print("ERROR: NCBI_EMAIL is required to fetch records from PubMed.", file=sys.stderr)
        print("FAIL")
        return 1

    try:
        records = fetch_batch(list(REQUESTED_PMIDS), email, api_key)
    except Exception as error:
        print(f"ERROR: PubMed EFetch failed: {type(error).__name__}: {error}", file=sys.stderr)
        print("FAIL")
        return 1

    all_passed = callable(parse_pubmed_book_article)
    for requested_pmid in REQUESTED_PMIDS:
        record = records.get(requested_pmid)
        if record is None:
            print(f"\nPMID: {requested_pmid}")
            print("record_type: (not returned)")
            print("PMCID: (not available)")
            print("title: (not available)")
            print("book_title: (not available)")
            print("publisher: (not available)")
            print("PMID_MATCH: NO")
            print("PMCID_PRESENT: NO")
            print("IS_PUBMED_BOOK_ARTICLE: NO")
            all_passed = False
            continue

        returned_pmid = str(record.get("pmid", "")).strip()
        record_type = record.get("record_type", "")
        pmcid = str(record.get("pmcid", "")).strip()
        title = str(record.get("title", "")).strip()
        book_title = str(record.get("book_title", "")).strip()
        publisher = str(record.get("publisher", "")).strip()
        pmid_matches = returned_pmid == requested_pmid
        is_book_article = record_type == "pubmed_book_article"

        print(f"\nPMID: {returned_pmid or '(not available)'}")
        print(f"record_type: {display(record_type)}")
        print(f"PMCID: {display(pmcid)}")
        print(f"title: {display(title)}")
        print(f"book_title: {display(book_title)}")
        print(f"publisher: {display(publisher)}")
        print(f"PMID_MATCH: {'YES' if pmid_matches else 'NO'}")
        print(f"PMCID_PRESENT: {'YES' if pmcid else 'NO'}")
        print(f"IS_PUBMED_BOOK_ARTICLE: {'YES' if is_book_article else 'NO'}")

        if not (pmid_matches and is_book_article and title):
            all_passed = False

    print("\nPASS" if all_passed else "\nFAIL")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
