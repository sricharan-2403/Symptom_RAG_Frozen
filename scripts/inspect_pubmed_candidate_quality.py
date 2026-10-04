import csv
from collections import Counter


INPUT_PATH = "data/processed/pubmed_complete_metadata.csv"
OUTPUT_PATH = "data/processed/pubmed_candidate_quality_report.txt"


def split_pipe(value):
    if not value:
        return []
    return [
        item.strip()
        for item in value.split("|")
        if item.strip()
    ]


def main():
    with open(
        INPUT_PATH,
        encoding="utf-8",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    total = len(rows)

    record_types = Counter()
    languages = Counter()
    years = Counter()
    publication_types = Counter()
    query_categories = Counter()

    abstract_count = 0
    pmcid_count = 0
    doi_count = 0
    mesh_count = 0
    author_count = 0

    disease_association_counts = Counter()

    for row in rows:
        # Record type
        record_types[row.get("record_type", "") or "missing"] += 1

        # Language
        language = row.get("language", "").strip()
        languages[language or "missing"] += 1

        # Publication year
        year = row.get("publication_year", "").strip()
        years[year or "missing"] += 1

        # Publication types
        for item in split_pipe(row.get("publication_types", "")):
            publication_types[item] += 1

        # Query categories
        for item in split_pipe(row.get("query_categories", "")):
            query_categories[item] += 1

        # Metadata completeness
        if row.get("abstract", "").strip():
            abstract_count += 1

        if row.get("pmcid", "").strip():
            pmcid_count += 1

        if row.get("doi", "").strip():
            doi_count += 1

        if row.get("mesh_terms", "").strip():
            mesh_count += 1

        if row.get("authors", "").strip():
            author_count += 1

        # Disease associations
        disease_ids = split_pipe(row.get("disease_ids", ""))
        disease_association_counts[len(disease_ids)] += 1

    lines = []

    lines.append("PUBMED CANDIDATE QUALITY REPORT")
    lines.append("=" * 80)
    lines.append(f"Total metadata records: {total}")
    lines.append("")

    lines.append("RECORD TYPES")
    lines.append("-" * 80)
    for key, value in record_types.most_common():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("METADATA COMPLETENESS")
    lines.append("-" * 80)

    completeness = {
        "abstract": abstract_count,
        "pmcid": pmcid_count,
        "doi": doi_count,
        "mesh_terms": mesh_count,
        "authors": author_count,
    }

    for field, count in completeness.items():
        percentage = (count / total * 100) if total else 0
        lines.append(
            f"{field}: {count} "
            f"({percentage:.1f}%)"
        )

    lines.append("")
    lines.append("LANGUAGES")
    lines.append("-" * 80)
    for key, value in languages.most_common():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("PUBLICATION TYPES")
    lines.append("-" * 80)
    for key, value in publication_types.most_common():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("QUERY CATEGORY COVERAGE")
    lines.append("-" * 80)
    for key, value in query_categories.most_common():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("DISEASE ASSOCIATION COUNT")
    lines.append("-" * 80)
    for key, value in sorted(
        disease_association_counts.items()
    ):
        lines.append(
            f"{key} disease association(s): {value}"
        )

    lines.append("")
    lines.append("PUBLICATION YEARS")
    lines.append("-" * 80)

    # Show all years in chronological order
    for key, value in sorted(
        years.items(),
        key=lambda item: (
            999999
            if item[0] == "missing"
            else int(item[0])
        )
    ):
        lines.append(f"{key}: {value}")

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        f.write("\n".join(lines))

    print()
    print("PUBMED CANDIDATE QUALITY INSPECTION")
    print("=" * 80)
    print(f"Records inspected: {total}")
    print()
    print(f"Report written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()