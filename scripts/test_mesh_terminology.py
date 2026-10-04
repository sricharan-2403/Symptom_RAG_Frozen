import json
import urllib.parse
import urllib.request


MESH_LOOKUP_URL = "https://id.nlm.nih.gov/mesh/lookup/term"


TEST_TERMS = [
    "Adhd",
    "Adult Adhd",
    "Aids",
    "Cmv",
    "Copd",
    "Crohns Disease",
    "Gerd",
    "Mrsa",
    "Pcos",
    "Sars",
    "Addisons Disease",
    "Parkinsons Disease",
    "Dimorphic Hemmorhoids",
    "Vertigo",
]


def lookup_mesh(term: str):

    params = urllib.parse.urlencode({
        "label": term
    })

    url = f"{MESH_LOOKUP_URL}?{params}"

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Symptom-RAG-Analyser/1.0"
        }
    )

    try:

        with urllib.request.urlopen(request, timeout=20) as response:
            data = response.read().decode("utf-8")

        return json.loads(data)

    except Exception as e:

        return {
            "error": str(e)
        }


print("\n" + "=" * 80)
print("MeSH TERMINOLOGY LOOKUP TEST")
print("=" * 80)


for term in TEST_TERMS:

    print("\n" + "-" * 80)
    print(f"DATASET TERM: {term}")
    print("-" * 80)

    result = lookup_mesh(term)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )


print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)