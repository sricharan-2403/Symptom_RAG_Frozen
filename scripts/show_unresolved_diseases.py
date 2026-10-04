import csv

# Load disease manifest
with open(
    "data/processed/disease_literature_manifest.csv",
    encoding="utf-8",
    newline=""
) as f:
    manifest = list(csv.DictReader(f))


# Load zero-candidate recovery attempts
with open(
    "data/processed/pubmed_zero_candidate_recovery.csv",
    encoding="utf-8",
    newline=""
) as f:
    recovery = list(csv.DictReader(f))


# The recovery CSV contains attempts only for the
# original zero-candidate diseases.
original_zero_candidate_ids = {
    row["disease_id"]
    for row in recovery
}


# A disease is recovered if ANY search variant returned
# at least one PubMed result.
recovered_ids = {
    row["disease_id"]
    for row in recovery
    if int(row["result_count"]) > 0
}


# Therefore unresolved = original zero-candidate diseases
# minus the diseases successfully recovered.
unresolved_ids = original_zero_candidate_ids - recovered_ids


# Look up their manifest information.
unresolved = [
    row
    for row in manifest
    if row["disease_id"] in unresolved_ids
]


print()
print("UNRESOLVED DISEASES")
print("=" * 80)
print(f"Original zero-candidate diseases: {len(original_zero_candidate_ids)}")
print(f"Recovered: {len(recovered_ids)}")
print(f"Still unresolved: {len(unresolved)}")
print()

for row in unresolved:
    print(
        f'{row["disease_id"]} | '
        f'{row["disease_name"]} | '
        f'{row["unique_symptom_count"]} | '
        f'{row["coverage_group"]}'
    )