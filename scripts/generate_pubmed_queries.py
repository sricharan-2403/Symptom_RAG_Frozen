from pathlib import Path
import csv


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

MANIFEST_PATH = Path(
    "data/processed/disease_literature_manifest.csv"
)

OUTPUT_PATH = Path(
    "data/processed/pubmed_query_manifest.csv"
)


# ---------------------------------------------------------
# Evidence categories
# ---------------------------------------------------------

QUERY_CATEGORIES = {
    "clinical_presentation": '"clinical presentation"',
    "symptoms": '"symptoms"',
    "diagnosis": '"diagnosis"',
    "clinical_findings": '"clinical findings"',
    "differential_diagnosis": '"differential diagnosis"',
    "complications": '"complications"',
}


# ---------------------------------------------------------
# Search-term overrides
#
# IMPORTANT:
# We only put terms here when we have explicitly verified
# that the alternative is appropriate.
#
# Everything else uses the original dataset name.
# ---------------------------------------------------------

SEARCH_TERM_OVERRIDES = {
    # Example:
    # "Copd": "COPD",
    # "Pcos": "Polycystic Ovary Syndrome",
}


# ---------------------------------------------------------
# Clean disease name
# ---------------------------------------------------------

def clean_name(name: str) -> str:

    return " ".join(name.split()).strip()


# ---------------------------------------------------------
# Determine search term
# ---------------------------------------------------------

def get_search_term(disease_name: str) -> str:

    disease_name = clean_name(disease_name)

    return SEARCH_TERM_OVERRIDES.get(
        disease_name,
        disease_name
    )


# ---------------------------------------------------------
# Build PubMed query
# ---------------------------------------------------------

def build_query(search_term: str, evidence_term: str) -> str:

    return f'"{search_term}" AND {evidence_term}'


# ---------------------------------------------------------
# Load disease manifest
# ---------------------------------------------------------

with open(
    MANIFEST_PATH,
    "r",
    encoding="utf-8"
) as f:

    diseases = list(csv.DictReader(f))


# ---------------------------------------------------------
# Generate queries
# ---------------------------------------------------------

query_rows = []

query_id = 1

for disease in diseases:

    disease_id = disease["disease_id"]
    disease_name = disease["disease_name"]
    coverage_group = disease["coverage_group"]

    search_term = get_search_term(disease_name)

    for category, evidence_term in QUERY_CATEGORIES.items():

        query = build_query(
            search_term,
            evidence_term
        )

        query_rows.append({
            "query_id": query_id,
            "disease_id": disease_id,
            "disease_name": disease_name,
            "search_term": search_term,
            "coverage_group": coverage_group,
            "query_category": category,
            "pubmed_query": query,
        })

        query_id += 1


# ---------------------------------------------------------
# Create output directory
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Write query manifest
# ---------------------------------------------------------

fieldnames = [
    "query_id",
    "disease_id",
    "disease_name",
    "search_term",
    "coverage_group",
    "query_category",
    "pubmed_query",
]


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(query_rows)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("PUBMED QUERY MANIFEST")
print("=" * 80)

print(f"\nDiseases              : {len(diseases)}")
print(f"Evidence categories   : {len(QUERY_CATEGORIES)}")
print(f"Queries generated     : {len(query_rows)}")

print("\nExpected:")
print(f"  {len(diseases)} × {len(QUERY_CATEGORIES)} "
      f"= {len(diseases) * len(QUERY_CATEGORIES)}")


print("\n" + "-" * 80)
print("FIRST 20 QUERIES")
print("-" * 80)

for row in query_rows[:20]:

    print(
        f"{row['query_id']:4d} | "
        f"{row['disease_name']:40s} | "
        f"{row['query_category']:25s} | "
        f"{row['pubmed_query']}"
    )


print("\n" + "-" * 80)
print("QUERY CATEGORY COUNTS")
print("-" * 80)

for category in QUERY_CATEGORIES:

    count = sum(
        1
        for row in query_rows
        if row["query_category"] == category
    )

    print(
        f"{category:25s}: {count}"
    )


print("\nOutput:")
print(f"  {OUTPUT_PATH}")

print("\n" + "=" * 80)
print("QUERY GENERATION COMPLETE")
print("=" * 80)