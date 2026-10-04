import csv
import json
import re
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


MANIFEST_PATH = Path(
    "data/processed/disease_literature_manifest.csv"
)

RECOVERY_PATH = Path(
    "data/processed/pubmed_zero_candidate_recovery.csv"
)

OUTPUT_PATH = Path(
    "data/processed/unresolved_disease_terminology_audit.csv"
)

MESH_LOOKUP_URL = (
    "https://id.nlm.nih.gov/mesh/lookup/term?label={}&match=exact"
)

REQUEST_DELAY = 0.2


def normalize_name(name):
    """Basic normalization only; does not change the original label."""
    value = name.strip()
    value = re.sub(r"\s+", " ", value)
    return value


def lookup_mesh(term):
    """
    Look up an exact MeSH term/entry term using the
    official NLM MeSH RDF lookup service.
    """
    url = MESH_LOOKUP_URL.format(quote(term))

    request = Request(
        url,
        headers={
            "User-Agent": "symptom-rag-analyser/1.0"
        }
    )

    try:
        with urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))

        if not data:
            return None

        # The lookup service returns matching resources.
        # We record the first exact result only.
        return data[0]

    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None


def main():
    with open(
        MANIFEST_PATH,
        encoding="utf-8",
        newline=""
    ) as f:
        manifest = list(csv.DictReader(f))

    with open(
        RECOVERY_PATH,
        encoding="utf-8",
        newline=""
    ) as f:
        recovery = list(csv.DictReader(f))

    # The recovery file contains attempts only for
    # the original 32 zero-candidate diseases.
    original_zero_ids = {
        row["disease_id"]
        for row in recovery
    }

    recovered_ids = {
        row["disease_id"]
        for row in recovery
        if int(row["result_count"]) > 0
    }

    unresolved_ids = original_zero_ids - recovered_ids

    unresolved = [
        row
        for row in manifest
        if row["disease_id"] in unresolved_ids
    ]

    unresolved.sort(
        key=lambda row: int(row["disease_id"])
    )

    print()
    print("UNRESOLVED DISEASE TERMINOLOGY AUDIT")
    print("=" * 80)
    print(f"Original zero-candidate diseases: {len(original_zero_ids)}")
    print(f"Recovered: {len(recovered_ids)}")
    print(f"To audit: {len(unresolved)}")
    print()

    results = []

    for index, disease in enumerate(unresolved, start=1):
        disease_id = disease["disease_id"]
        original_name = disease["disease_name"]
        normalized = normalize_name(original_name)

        print(
            f"[{index}/{len(unresolved)}] "
            f"{disease_id} | {original_name}"
        )

        mesh_result = lookup_mesh(normalized)

        if mesh_result:
            status = "mesh_match"
            mesh_resource = mesh_result
            reason = "Exact MeSH lookup match."
        else:
            status = "no_exact_mesh_match"
            mesh_resource = ""
            reason = (
                "No exact MeSH lookup match; "
                "requires alternate terminology review."
            )

        results.append({
            "disease_id": disease_id,
            "original_name": original_name,
            "normalized_name": normalized,
            "status": status,
            "canonical_search_name": "",
            "alternate_terms": "",
            "mesh_resource": mesh_resource,
            "reason": reason,
        })

        time.sleep(REQUEST_DELAY)

    fieldnames = [
        "disease_id",
        "original_name",
        "normalized_name",
        "status",
        "canonical_search_name",
        "alternate_terms",
        "mesh_resource",
        "reason",
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
        writer.writerows(results)

    mesh_matches = sum(
        1
        for row in results
        if row["status"] == "mesh_match"
    )

    print()
    print("AUDIT COMPLETE")
    print("=" * 80)
    print(f"Audited: {len(results)}")
    print(f"Exact MeSH matches: {mesh_matches}")
    print(
        f"No exact MeSH match: "
        f"{len(results) - mesh_matches}"
    )
    print()
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()