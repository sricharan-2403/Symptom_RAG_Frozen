from pathlib import Path
from collections import Counter
import csv
import json


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_DIR = Path("data/raw/symptom-disease-dataset")

CSV_PATH = DATA_DIR / "symptom-disease-train-dataset.csv"
MAPPING_PATH = DATA_DIR / "mapping.json"


# ---------------------------------------------------------
# Load disease mapping
# ---------------------------------------------------------

with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)


# mapping.json:
# {
#     "Disease Name": disease_id
# }

id_to_disease = {
    int(disease_id): disease_name
    for disease_name, disease_id in mapping.items()
}


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

raw_rows = 0
empty_rows = 0

raw_disease_counts = Counter()

unique_pairs = set()
unique_disease_counts = Counter()


with open(CSV_PATH, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        raw_rows += 1

        text = (row.get("text") or "").strip()
        label = (row.get("label") or "").strip()

        if not text or not label:
            empty_rows += 1
            continue

        disease_id = int(label)

        raw_disease_counts[disease_id] += 1

        # Exact duplicate removal
        pair = (text, disease_id)

        if pair not in unique_pairs:
            unique_pairs.add(pair)
            unique_disease_counts[disease_id] += 1


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("DISEASE DISTRIBUTION ANALYSIS")
print("=" * 75)

print(f"\nCSV rows                 : {raw_rows}")
print(f"Empty/invalid rows      : {empty_rows}")
print(f"Unique symptom-disease  : {len(unique_pairs)}")
print(f"Unique diseases         : {len(unique_disease_counts)}")


# ---------------------------------------------------------
# Top diseases
# ---------------------------------------------------------

print("\n" + "-" * 75)
print("TOP 20 DISEASES — UNIQUE SYMPTOM COUNT")
print("-" * 75)

for disease_id, count in unique_disease_counts.most_common(20):

    disease_name = id_to_disease.get(
        disease_id,
        f"UNKNOWN_DISEASE_{disease_id}"
    )

    print(
        f"{disease_id:4d} | "
        f"{disease_name:45s} | "
        f"{count:3d}"
    )


# ---------------------------------------------------------
# Frequency buckets
# ---------------------------------------------------------

buckets = {
    "1 example": 0,
    "2-5 examples": 0,
    "6-10 examples": 0,
    "11-20 examples": 0,
    "21-50 examples": 0,
    "51-100 examples": 0,
    "101+ examples": 0,
}


for count in unique_disease_counts.values():

    if count == 1:
        buckets["1 example"] += 1

    elif count <= 5:
        buckets["2-5 examples"] += 1

    elif count <= 10:
        buckets["6-10 examples"] += 1

    elif count <= 20:
        buckets["11-20 examples"] += 1

    elif count <= 50:
        buckets["21-50 examples"] += 1

    elif count <= 100:
        buckets["51-100 examples"] += 1

    else:
        buckets["101+ examples"] += 1


print("\n" + "-" * 75)
print("DISEASE FREQUENCY BUCKETS")
print("-" * 75)

for bucket, count in buckets.items():
    print(f"{bucket:25s} {count:5d}")


# ---------------------------------------------------------
# Lowest-frequency diseases
# ---------------------------------------------------------

print("\n" + "-" * 75)
print("20 LOWEST-FREQUENCY DISEASES")
print("-" * 75)

for disease_id, count in sorted(
    unique_disease_counts.items(),
    key=lambda x: (x[1], id_to_disease.get(x[0], ""))
)[:20]:

    disease_name = id_to_disease.get(
        disease_id,
        f"UNKNOWN_DISEASE_{disease_id}"
    )

    print(
        f"{disease_id:4d} | "
        f"{disease_name:45s} | "
        f"{count:3d}"
    )


# ---------------------------------------------------------
# Mapping coverage
# ---------------------------------------------------------

missing_mapping = [
    disease_id
    for disease_id in unique_disease_counts
    if disease_id not in id_to_disease
]


print("\n" + "-" * 75)
print("MAPPING VALIDATION")
print("-" * 75)

print(f"Diseases in CSV          : {len(unique_disease_counts)}")
print(f"Diseases in mapping      : {len(id_to_disease)}")
print(f"Missing mapping entries  : {len(missing_mapping)}")


print("\n" + "=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)