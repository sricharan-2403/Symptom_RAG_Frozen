import sys
from pathlib import Path
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from symptom_rag_analyzer.data.literature_parser import (
    PMCLiteratureParser,
    _IGNORED_TEXT_TAGS,
)


SNIPPETS = (
    "<p>The patient had <italic>severe</italic> fever.</p>",
    '<p>described in<xref ref-type="table">Table 1</xref> for the patients.</p>',
    '<p>6.6 months<xref ref-type="bibr">[1]</xref>(IQR: 5.1\u20139.2).</p>',
    "<p>This is <bold>clinically</bold> important.</p>",
    "<p>Patients with <sup>2</sup> symptoms.</p>",
)


def main():
    parser = PMCLiteratureParser()
    for index, snippet in enumerate(SNIPPETS, start=1):
        element = ET.fromstring(snippet)
        output = parser._collect_text(element, _IGNORED_TEXT_TAGS)
        print(f"CASE {index}")
        print(f"INPUT: {snippet}")
        print(f"OUTPUT: {output}")
        print()


if __name__ == "__main__":
    main()
