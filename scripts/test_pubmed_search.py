from pathlib import Path
import csv
import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

QUERY_MANIFEST = Path(
    "data/processed/pubmed_query_manifest.csv"
)

PUBMED_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
)

# NCBI requests a valid email in E-utility requests.
NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "").strip()

NCBI_TOOL = "Symptom_RAG_Analyser"


# ---------------------------------------------------------
# Load selected test queries
# ---------------------------------------------------------

with open(
    QUERY_MANIFEST,
    "r",
    encoding="utf-8"
) as f:

    rows = list(csv.DictReader(f))


# We deliberately test a few different diseases/categories.
TEST_QUERY_IDS = [
    "1",      # positional vertigo - clinical presentation
    "7",      # abdominal aortic aneurysm - clinical presentation
    "13",     # acanthosis nigricans - clinical presentation
    "37",     # acne - clinical presentation
    "596",    # depends on manifest ordering; just a controlled test
]


selected = []

for row in rows:

    if row["query_id"] in TEST_QUERY_IDS:
        selected.append(row)


# ---------------------------------------------------------
# PubMed ESearch
# ---------------------------------------------------------

def pubmed_search(query: str, retmax: int = 5):

    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
        "tool": NCBI_TOOL,
        "email": NCBI_EMAIL,
    }

    url = (
        PUBMED_URL
        + "?"
        + urllib.parse.urlencode(params)
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                f"{NCBI_TOOL}/1.0 "
                f"({NCBI_EMAIL})"
            )
        }
    )

    with urllib.request.urlopen(
        request,
        timeout=30
    ) as response:

        data = response.read().decode("utf-8")

    return json.loads(data)


# ---------------------------------------------------------
# Run tests
# ---------------------------------------------------------

print("\n" + "=" * 85)
print("PUBMED LIVE SEARCH TEST")
print("=" * 85)

print(
    "\nTesting",
    len(selected),
    "queries."
)

print(
    "Only PubMed metadata/PMIDs will be requested."
)

print(
    "No full-text papers will be downloaded."
)


for row in selected:

    print("\n" + "-" * 85)

    print(
        f"Query ID      : {row['query_id']}"
    )

    print(
        f"Disease       : {row['disease_name']}"
    )

    print(
        f"Category      : {row['query_category']}"
    )

    print(
        f"PubMed query  : {row['pubmed_query']}"
    )

    try:

        result = pubmed_search(
            row["pubmed_query"],
            retmax=5
        )

        result_data = result.get(
            "esearchresult",
            {}
        )

        count = result_data.get(
            "count",
            "unknown"
        )

        pmids = result_data.get(
            "idlist",
            []
        )

        print(
            f"\nTotal PubMed matches : {count}"
        )

        print(
            f"Returned PMIDs       : {len(pmids)}"
        )

        if pmids:

            print("\nPMIDs:")

            for pmid in pmids:

                print(
                    f"  - {pmid}"
                )

        else:

            print(
                "\nNo PMIDs returned."
            )

    except Exception as e:

        print(
            "\nERROR:"
        )

        print(
            f"  {type(e).__name__}: {e}"
        )

    # Keep the test deliberately gentle.
    time.sleep(0.5)


print("\n" + "=" * 85)
print("PUBMED LIVE TEST COMPLETE")
print("=" * 85)