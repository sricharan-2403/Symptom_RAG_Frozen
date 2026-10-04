from pathlib import Path
from collections import Counter
import csv
import json


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_DIR = Path("data/raw/symptom-disease-dataset")
OUTPUT_DIR = Path("data/processed")

CSV_PATH = DATA_DIR / "symptom-disease-train-dataset.csv"
MAPPING_PATH = DATA_DIR / "mapping.json"

OUTPUT_PATH = OUTPUT_DIR / "disease_literature_manifest.csv"


# ---------------------------------------------------------
# Load disease mapping
# ---------------------------------------------------------

with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)

id_to_disease = {
    int(disease_id): disease_name
    for disease_name, disease_id in mapping.items()
}


# ---------------------------------------------------------
# Read dataset and calculate unique symptom counts
# ---------------------------------------------------------

unique_pairs = set()
disease_counts = Counter()

with open(CSV_PATH, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        text = (row.get("text") or "").strip()
        label = (row.get("label") or "").strip()

        if not text or not label:
            continue

        disease_id = int(label)

        pair = (text, disease_id)

        if pair not in unique_pairs:
            unique_pairs.add(pair)
            disease_counts[disease_id] += 1


# ---------------------------------------------------------
# Determine coverage group
# ---------------------------------------------------------

def get_coverage_group(count: int) -> str:

    if count == 1:
        return "low_coverage"

    if count <= 10:
        return "moderate_coverage"

    return "high_coverage"


# ---------------------------------------------------------
# Build manifest
# ---------------------------------------------------------

manifest = []

for disease_id in sorted(disease_counts):

    disease_name = id_to_disease[disease_id]
    count = disease_counts[disease_id]

    manifest.append({
        "disease_id": disease_id,
        "disease_name": disease_name,
        "unique_symptom_count": count,
        "coverage_group": get_coverage_group(count),
        "literature_status": "pending",
    })


# ---------------------------------------------------------
# Create output directory
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Write CSV
# ---------------------------------------------------------

fieldnames = [
    "disease_id",
    "disease_name",
    "unique_symptom_count",
    "coverage_group",
    "literature_status",
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
    writer.writerows(manifest)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

coverage_counts = Counter(
    row["coverage_group"]
    for row in manifest
)


print("\n" + "=" * 75)
print("DISEASE LITERATURE MANIFEST")
print("=" * 75)

print(f"\nDiseases included       : {len(manifest)}")
print(f"Unique symptom pairs    : {len(unique_pairs)}")

print("\nCoverage groups:")

for group, count in coverage_counts.items():
    print(f"  {group:20s}: {count}")

print(f"\nOutput file:")
print(f"  {OUTPUT_PATH}")

print("\nFirst 10 entries:")

for row in manifest[:10]:
    print(
        f"  {row['disease_id']:4d} | "
        f"{row['disease_name']:40s} | "
        f"{row['unique_symptom_count']:3d} | "
        f"{row['coverage_group']}"
    )

print("\n" + "=" * 75)
print("MANIFEST CREATION COMPLETE")
print("=" * 75)