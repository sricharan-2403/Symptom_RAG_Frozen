from pathlib import Path
import csv
import re


MANIFEST_PATH = Path(
    "data/processed/disease_literature_manifest.csv"
)


# ---------------------------------------------------------
# Load manifest
# ---------------------------------------------------------

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))


# ---------------------------------------------------------
# Heuristic checks
#
# These do NOT automatically modify disease names.
# They only identify names worth reviewing.
# ---------------------------------------------------------

def find_flags(name: str):

    flags = []

    # Parentheses
    if "(" in name or ")" in name:
        flags.append("parentheses")

    # Multiple spaces
    if "  " in name:
        flags.append("multiple_spaces")

    # Apostrophe-like missing forms
    lower = name.lower()

    if lower in {
        "addisons disease",
        "crohns disease",
        "parkinsons disease",
        "alzheimer disease",
    }:
        flags.append("possible_apostrophe_variant")

    # Common abbreviation-style labels
    abbreviation_candidates = {
        "adhd",
        "adult adhd",
        "copd",
        "uti",
        "ibs",
        "gerd",
        "pms",
        "ptsd",
    }

    if lower in abbreviation_candidates:
        flags.append("abbreviation")

    # Very short names
    if len(name.strip()) <= 4:
        flags.append("short_name")

    # Dataset-style spelling
    if "paroymsal" in lower:
        flags.append("possible_dataset_typo")

    return flags


# ---------------------------------------------------------
# Inspect
# ---------------------------------------------------------

flagged = []

for row in rows:

    disease_name = row["disease_name"]

    flags = find_flags(disease_name)

    if flags:

        flagged.append({
            "disease_id": row["disease_id"],
            "disease_name": disease_name,
            "unique_symptom_count": row["unique_symptom_count"],
            "flags": ", ".join(flags),
        })


# ---------------------------------------------------------
# Output
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("DISEASE SEARCH-NAME INSPECTION")
print("=" * 80)

print(f"\nTotal diseases : {len(rows)}")
print(f"Flagged        : {len(flagged)}")


print("\n" + "-" * 80)
print("FLAGGED DISEASE NAMES")
print("-" * 80)

for row in flagged:

    print(
        f"{int(row['disease_id']):4d} | "
        f"{row['disease_name']:50s} | "
        f"{row['flags']}"
    )


print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)