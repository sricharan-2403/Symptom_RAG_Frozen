from pathlib import Path
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


def generate_variants(term: str):

    variants = []

    def add(value):
        value = " ".join(value.split()).strip()

        if value and value not in variants:
            variants.append(value)

    # Original
    add(term)

    # Normalize whitespace
    add(" ".join(term.split()))

    # Apostrophe variants
    replacements = {
        "Addisons": "Addison's",
        "Crohns": "Crohn's",
        "Parkinsons": "Parkinson's",
    }

    for old, new in replacements.items():

        if old.lower() in term.lower():

            variant = term.replace(old, new)
            add(variant)

            variant = term.replace(
                old.lower(),
                new.lower()
            )
            add(variant)

    # Known abbreviation expansions for this prototype.
    # These are deliberately small and explicit.
    abbreviation_map = {
        "Cmv": "Cytomegalovirus",
        "Pcos": "Polycystic Ovary Syndrome",
        "Sars": "Severe Acute Respiratory Syndrome",
        "Copd": "Chronic Obstructive Pulmonary Disease",
        "Gerd": "Gastroesophageal Reflux Disease",
        "Mrsa": "Methicillin-Resistant Staphylococcus Aureus",
        "Aids": "Acquired Immunodeficiency Syndrome",
        "Adhd": "Attention Deficit Hyperactivity Disorder",
    }

    if term in abbreviation_map:
        add(abbreviation_map[term])

    # Parenthetical cleanup
    if "(" in term and ")" in term:

        without_parentheses = term

        import re

        without_parentheses = re.sub(
            r"\([^)]*\)",
            "",
            without_parentheses
        )

        add(without_parentheses)

    return variants


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

        with urllib.request.urlopen(
            request,
            timeout=20
        ) as response:

            data = response.read().decode("utf-8")

        return json.loads(data)

    except Exception as e:

        return {
            "error": str(e)
        }


print("\n" + "=" * 85)
print("MeSH TERMINOLOGY LOOKUP — VARIANT TEST")
print("=" * 85)


for original in TEST_TERMS:

    print("\n" + "-" * 85)
    print(f"ORIGINAL: {original}")
    print("-" * 85)

    variants = generate_variants(original)

    found = False

    for variant in variants:

        result = lookup_mesh(variant)

        if isinstance(result, list) and result:

            found = True

            print(f"\nMATCH")
            print(f"  Query    : {variant}")

            for item in result:

                print(
                    f"  Label    : {item.get('label')}"
                )

                print(
                    f"  Resource : {item.get('resource')}"
                )

            break

    if not found:

        print("\nNO MATCH FOUND")

        print("Variants tried:")

        for variant in variants:
            print(f"  - {variant}")


print("\n" + "=" * 85)
print("VARIANT TEST COMPLETE")
print("=" * 85)