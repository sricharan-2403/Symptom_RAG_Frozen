from pathlib import Path
import csv
import time
import requests


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "disease_literature_manifest.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pubmed_candidate_articles.csv"
)

PUBMED_ESEARCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
)

RETMAX_PER_QUERY = 20

# Be polite to NCBI.
REQUEST_DELAY = 0.5

TOOL_NAME = "symptom_rag_analyser"
EMAIL = "sricharanrampelli24@gmail.com"


# ============================================================
# QUERY GENERATION
# ============================================================

def build_queries(disease_name):
    """
    Build controlled PubMed queries for one disease.

    Current strategy:
    - Title/Abstract targeting
    - Precision-first
    """

    disease = disease_name.strip()

    return [
        {
            "category": "diagnosis",
            "query": (
                f'"{disease}"[Title/Abstract] AND '
                f'(diagnos*[Title/Abstract] OR '
                f'"diagnostic criteria"[Title/Abstract])'
            ),
        },
        {
            "category": "clinical_presentation",
            "query": (
                f'"{disease}"[Title/Abstract] AND '
                f'("clinical presentation"[Title/Abstract] OR '
                f'"clinical features"[Title/Abstract])'
            ),
        },
        {
            "category": "symptoms",
            "query": (
                f'"{disease}"[Title/Abstract] AND '
                f'symptom*[Title/Abstract]'
            ),
        },
    ]


# ============================================================
# PUBMED SEARCH
# ============================================================

def search_pubmed(query):
    """
    Search PubMed using ESearch and return PMIDs.
    """

    params = {
        "db": "pubmed",
        "term": query,
        "retmax": RETMAX_PER_QUERY,
        "retmode": "json",
        "sort": "relevance",
        "tool": TOOL_NAME,
        "email": EMAIL,
    }

    response = requests.get(
        PUBMED_ESEARCH_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    result = data["esearchresult"]

    count = int(result["count"])

    pmids = result.get("idlist", [])

    return count, pmids


# ============================================================
# MAIN ACQUISITION PIPELINE
# ============================================================

def main():

    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(
            f"Input manifest not found:\n{INPUT_MANIFEST}"
        )

    print("=" * 90)
    print("PUBMED CANDIDATE ACQUISITION")
    print("=" * 90)

    # --------------------------------------------------------
    # Load disease manifest
    # --------------------------------------------------------

    with INPUT_MANIFEST.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        diseases = list(csv.DictReader(f))

    print(f"\nDiseases loaded: {len(diseases)}")

    # --------------------------------------------------------
    # Candidate storage
    # --------------------------------------------------------

    candidates = {}

    total_queries = 0
    successful_queries = 0
    failed_queries = 0

    # --------------------------------------------------------
    # Search each disease
    # --------------------------------------------------------

    for index, disease in enumerate(diseases, start=1):

        disease_id = disease["disease_id"]
        disease_name = disease["disease_name"]

        queries = build_queries(disease_name)

        print(
            f"\n[{index}/{len(diseases)}] "
            f"{disease_name}"
        )

        for query_info in queries:

            total_queries += 1

            category = query_info["category"]
            query = query_info["query"]

            try:

                total_matches, pmids = search_pubmed(query)

                successful_queries += 1

                print(
                    f"  {category:22s} "
                    f"matches={total_matches:6d} "
                    f"returned={len(pmids):2d}"
                )

                for rank, pmid in enumerate(pmids, start=1):

                    if pmid not in candidates:

                        candidates[pmid] = {
                            "pmid": pmid,
                            "disease_ids": set(),
                            "disease_names": set(),
                            "query_categories": set(),
                            "queries": set(),
                            "best_rank": rank,
                            "match_counts": [],
                        }

                    candidate = candidates[pmid]

                    candidate["disease_ids"].add(disease_id)
                    candidate["disease_names"].add(disease_name)
                    candidate["query_categories"].add(category)
                    candidate["queries"].add(query)
                    candidate["match_counts"].append(total_matches)

                    candidate["best_rank"] = min(
                        candidate["best_rank"],
                        rank
                    )

                time.sleep(REQUEST_DELAY)

            except Exception as exc:

                failed_queries += 1

                print(
                    f"  ERROR in {category}: {exc}"
                )

    # --------------------------------------------------------
    # Save candidate manifest
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "pmid",
        "disease_ids",
        "disease_names",
        "query_categories",
        "queries",
        "best_rank",
        "match_counts",
    ]

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for pmid in sorted(
            candidates.keys(),
            key=lambda x: int(x)
        ):

            candidate = candidates[pmid]

            writer.writerow({
                "pmid": pmid,
                "disease_ids": "|".join(
                    sorted(candidate["disease_ids"])
                ),
                "disease_names": "|".join(
                    sorted(candidate["disease_names"])
                ),
                "query_categories": "|".join(
                    sorted(candidate["query_categories"])
                ),
                "queries": " || ".join(
                    sorted(candidate["queries"])
                ),
                "best_rank": candidate["best_rank"],
                "match_counts": "|".join(
                    map(
                        str,
                        candidate["match_counts"]
                    )
                ),
            })

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("ACQUISITION COMPLETE")
    print("=" * 90)

    print(f"Diseases:             {len(diseases)}")
    print(f"Queries attempted:    {total_queries}")
    print(f"Queries successful:   {successful_queries}")
    print(f"Queries failed:       {failed_queries}")
    print(f"Unique candidate PMIDs:{len(candidates)}")

    print(f"\nOutput:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()