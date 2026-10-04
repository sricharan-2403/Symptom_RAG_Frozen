from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET


PMC_DIR = Path("data/raw/pmc")


def local_name(tag: str) -> str:
    """Return XML tag name without namespace."""
    return tag.rsplit("}", 1)[-1]


def clean_text(element) -> str:
    """Extract normalized text from an XML element."""
    return " ".join("".join(element.itertext()).split())


def main() -> None:
    files = sorted(PMC_DIR.glob("*.xml"))

    if not files:
        print("ERROR: No PMC XML files found.")
        return

    print("=" * 80)
    print("PMC XML CORPUS STRUCTURE PROFILE")
    print("=" * 80)
    print(f"Directory : {PMC_DIR}")
    print(f"XML files : {len(files)}")
    print()

    tag_counts = Counter()
    section_title_counts = Counter()
    article_type_counts = Counter()
    root_counts = Counter()

    body_presence = Counter()
    abstract_presence = Counter()

    nested_section_count = 0
    table_count = 0
    figure_count = 0
    reference_list_count = 0
    supplementary_count = 0

    parse_errors = []

    for index, path in enumerate(files, start=1):
        try:
            tree = ET.parse(path)
            root = tree.getroot()

            root_counts[local_name(root.tag)] += 1

            for element in root.iter():
                tag_counts[local_name(element.tag)] += 1

            articles = [
                element
                for element in root.iter()
                if local_name(element.tag) == "article"
            ]

            if not articles:
                continue

            article = articles[0]

            article_type = article.attrib.get("article-type", "unknown")
            article_type_counts[article_type] += 1

            # Abstract
            abstracts = [
                element
                for element in article.iter()
                if local_name(element.tag) == "abstract"
            ]

            if abstracts:
                abstract_presence["with_abstract"] += 1
            else:
                abstract_presence["without_abstract"] += 1

            # Body
            bodies = [
                element
                for element in article.iter()
                if local_name(element.tag) == "body"
            ]

            if bodies:
                body_presence["with_body"] += 1
            else:
                body_presence["without_body"] += 1

            if bodies:
                body = bodies[0]

                sections = [
                    element
                    for element in body.iter()
                    if local_name(element.tag) == "sec"
                ]

                for section in sections:
                    title_elements = [
                        child
                        for child in section
                        if local_name(child.tag) == "title"
                    ]

                    if title_elements:
                        title = clean_text(title_elements[0])
                        if title:
                            section_title_counts[title] += 1

                    direct_child_sections = [
                        child
                        for child in section
                        if local_name(child.tag) == "sec"
                    ]

                    if direct_child_sections:
                        nested_section_count += len(direct_child_sections)

            table_count += sum(
                1
                for element in article.iter()
                if local_name(element.tag) in {"table-wrap", "table"}
            )

            figure_count += sum(
                1
                for element in article.iter()
                if local_name(element.tag) in {"fig", "graphic"}
            )

            reference_list_count += sum(
                1
                for element in article.iter()
                if local_name(element.tag) == "ref-list"
            )

            supplementary_count += sum(
                1
                for element in article.iter()
                if local_name(element.tag) in {
                    "supplementary-material",
                    "supplementary-materials",
                }
            )

        except Exception as exc:
            parse_errors.append((path.name, str(exc)))

        if index % 250 == 0:
            print(f"Processed {index}/{len(files)} XML files...")

    print()
    print("-" * 80)
    print("ROOT TAGS")
    print("-" * 80)
    for name, count in root_counts.most_common():
        print(f"{name:30} {count}")

    print()
    print("-" * 80)
    print("ARTICLE TYPES")
    print("-" * 80)
    for name, count in article_type_counts.most_common():
        print(f"{name:30} {count}")

    print()
    print("-" * 80)
    print("BODY / ABSTRACT")
    print("-" * 80)
    for name, count in body_presence.items():
        print(f"{name:30} {count}")

    for name, count in abstract_presence.items():
        print(f"{name:30} {count}")

    print()
    print("-" * 80)
    print("IMPORTANT STRUCTURAL ELEMENTS")
    print("-" * 80)
    print(f"Nested sections       : {nested_section_count}")
    print(f"Table/table-wrap tags : {table_count}")
    print(f"Figure/graphic tags   : {figure_count}")
    print(f"Reference lists       : {reference_list_count}")
    print(f"Supplementary tags    : {supplementary_count}")

    print()
    print("-" * 80)
    print("MOST COMMON SECTION TITLES")
    print("-" * 80)

    for title, count in section_title_counts.most_common(100):
        print(f"{count:5}  {title}")

    print()
    print("-" * 80)
    print("MOST COMMON XML TAGS")
    print("-" * 80)

    for tag, count in tag_counts.most_common(50):
        print(f"{count:8}  {tag}")

    print()
    print("-" * 80)
    print("PARSE ERRORS")
    print("-" * 80)

    if not parse_errors:
        print("None")
    else:
        for filename, error in parse_errors[:50]:
            print(f"{filename}: {error}")

        if len(parse_errors) > 50:
            print(f"... and {len(parse_errors) - 50} more")

    print()
    print("=" * 80)
    print("PROFILE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()