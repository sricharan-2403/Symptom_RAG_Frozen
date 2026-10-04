import json
import urllib.parse
import urllib.request
import time
import os 

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PUBMED_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
)

NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "").strip()
NCBI_TOOL = "Symptom_RAG_Analyser"


# ---------------------------------------------------------
# Test diseases
# ---------------------------------------------------------

TEST_CASES = [
    {
        "disease": "Dengue",
        "category": "diagnosis",
    },
    {
        "disease": "Myocardial Infarction",
        "category": "diagnosis",
    },
    {
        "disease": "Pneumonia",
        "category": "clinical presentation",
    },
    {
        "disease": "Malaria",
        "category": "symptoms",
    },
    {
        "disease": "Acanthosis Nigricans",
        "category": "diagnosis",
    },
]


# ---------------------------------------------------------
# Query strategies
# ---------------------------------------------------------

def build_queries(disease, category):

    if category == "diagnosis":

        return {
            "A_exact": (
                f'"{disease}" AND "diagnosis"'
            ),

            "B_title_abstract": (
                f'"{disease}"[Title/Abstract] '
                f'AND '
                f'(diagnos*[Title/Abstract] '
                f'OR "diagnostic criteria"[Title/Abstract])'
            ),

            "C_broader": (
                f'"{disease}"[Title/Abstract] '
                f'AND '
                f'(diagnos*[Title/Abstract] '
                f'OR "clinical diagnosis"[Title/Abstract] '
                f'OR "diagnostic criteria"[Title/Abstract])'
            ),
        }

    if category == "clinical presentation":

        return {
            "A_exact": (
                f'"{disease}" AND "clinical presentation"'
            ),

            "B_title_abstract": (
                f'"{disease}"[Title/Abstract] '
                f'AND '
                f'("clinical presentation"[Title/Abstract] '
                f'OR "clinical features"[Title/Abstract])'
            ),

            "C_broader": (
                f'"{disease}"[Title/Abstract] '
                f'AND '
                f'("clinical presentation"[Title/Abstract] '
                f'OR "clinical features"[Title/Abstract] '
                f'OR "signs and symptoms"[Title/Abstract] '
                f'OR manifestations[Title/Abstract])'
            ),
        }

    if category == "symptoms":

        return {
            "A_exact": (
                f'"{disease}" AND "symptoms"'
            ),

            "B_title_abstract": (
                f'"{disease}"[Title/Abstract] '
                f'AND '
                f'(symptom*[Title/Abstract])'
            ),

            "C_broader": (
                f'"{disease}"[Title/Abstract] '
                f'AND '
                f'(symptom*[Title/Abstract] '
                f'OR "signs"[Title/Abstract] '
                f'OR manifestation*[Title/Abstract])'
            ),
        }

    raise ValueError(
        f"Unsupported category: {category}"
    )


# ---------------------------------------------------------
# PubMed search
# ---------------------------------------------------------

def pubmed_search(query, retmax=5):

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

    result = json.loads(data)

    search_result = result.get(
        "esearchresult",
        {}
    )

    return {
        "count": int(
            search_result.get("count", 0)
        ),
        "pmids": search_result.get(
            "idlist",
            []
        ),
    }


# ---------------------------------------------------------
# Run comparison
# ---------------------------------------------------------

print("\n" + "=" * 90)
print("PUBMED QUERY STRATEGY COMPARISON")
print("=" * 90)

print(
    "\nIMPORTANT: This test compares candidate-search behavior."
)

for case in TEST_CASES:

    disease = case["disease"]
    category = case["category"]

    print("\n" + "-" * 90)
    print(
        f"DISEASE: {disease} | CATEGORY: {category}"
    )
    print("-" * 90)

    queries = build_queries(
        disease,
        category
    )

    for strategy, query in queries.items():

        print(f"\n[{strategy}]")
        print(f"Query: {query}")

        try:

            result = pubmed_search(
                query,
                retmax=5
            )

            print(
                f"Total matches: {result['count']}"
            )

            print(
                "Top PMIDs:"
            )

            for pmid in result["pmids"]:
                print(f"  - {pmid}")

        except Exception as e:

            print(
                f"ERROR: {type(e).__name__}: {e}"
            )

        # Stay comfortably below the API request-rate limit.
        time.sleep(0.5)


print("\n" + "=" * 90)
print("COMPARISON COMPLETE")
print("=" * 90)