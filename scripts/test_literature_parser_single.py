import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from symptom_rag_analyzer.data.literature_parser import PMCLiteratureParser


XML_FILE = PROJECT_ROOT / "data" / "raw" / "pmc" / "PMC10001111.xml"


def excerpt(text, limit):
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "..."


def main():
    article = PMCLiteratureParser().parse_file(XML_FILE)

    print(f"XML file: {XML_FILE.relative_to(PROJECT_ROOT)}")
    print(f"PMID: {article.pmid}")
    print(f"PMCID: {article.pmcid}")
    print(f"DOI: {article.doi}")
    print(f"Title: {article.title}")
    print(f"Journal: {article.journal}")
    print(f"Publication date: {article.publication_date}")
    print(f"Publication year: {article.publication_year}")
    print(f"Article type: {article.article_type}")
    print(f"Record type: {article.record_type}")
    print(f"Number of authors: {len(article.authors)}")
    print(f"First 5 authors: {article.authors[:5]}")
    print(f"Number of MeSH terms: {len(article.mesh_terms)}")
    print(f"First 10 MeSH terms: {article.mesh_terms[:10]}")
    print(f"Number of abstract sections: {len(article.abstract_sections)}")
    print(f"Number of body sections: {len(article.body_sections)}")
    print(f"Has abstract: {article.has_abstract()}")
    print(f"Has body: {article.has_body()}")

    print("\nAbstract sections")
    for index, section in enumerate(article.abstract_sections, start=1):
        print(f"[{index}] level: {section.level}")
        print(f"    title: {section.title}")
        print(f"    normalized_title: {section.normalized_title}")
        print(f"    category: {section.category}")
        print(f"    section_path: {section.section_path}")
        print(f"    text: {excerpt(section.text, 300)}")

    print("\nFirst 15 body sections")
    for index, section in enumerate(article.body_sections[:15], start=1):
        print(f"[{index}] level: {section.level}")
        print(f"    title: {section.title}")
        print(f"    normalized_title: {section.normalized_title}")
        print(f"    category: {section.category}")
        print(f"    section_path: {section.section_path}")
        print(f"    text: {excerpt(section.text, 250)}")


if __name__ == "__main__":
    main()
