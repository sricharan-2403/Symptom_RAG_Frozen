import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SELECTOR_SCRIPT = SCRIPTS_DIR / "select_pubmed_literature.py"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_FILE = DATA_DIR / "pubmed_complete_metadata.csv"
DISEASE_MANIFEST_FILE = DATA_DIR / "disease_literature_manifest.csv"
SELECTED_FILE = DATA_DIR / "pubmed_selected_literature_STAGED.csv"
AUDIT_FILE = DATA_DIR / "pubmed_selection_audit_STAGED.csv"
SUMMARY_FILE = DATA_DIR / "pubmed_selection_summary_STAGED.txt"


def load_selector():
    spec = importlib.util.spec_from_file_location(
        "pubmed_literature_selector_for_staging",
        SELECTOR_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load selector logic from {SELECTOR_SCRIPT}")
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    return selector


def main():
    print(f"Metadata input: {METADATA_FILE}")
    print(f"Disease manifest input: {DISEASE_MANIFEST_FILE}")
    print(f"Staged selected output: {SELECTED_FILE}")
    print(f"Staged audit output: {AUDIT_FILE}")
    print(f"Staged summary output: {SUMMARY_FILE}")

    staged_outputs = (SELECTED_FILE, AUDIT_FILE, SUMMARY_FILE)
    existing = [path for path in staged_outputs if path.exists()]
    if existing:
        print(
            "ERROR: Refusing to overwrite existing staged output(s): "
            + ", ".join(str(path) for path in existing),
            file=sys.stderr,
        )
        return 1

    try:
        selector = load_selector()
    except (OSError, ImportError) as error:
        print(f"ERROR: Could not load selector logic: {error}", file=sys.stderr)
        return 1

    selector.METADATA_FILE = METADATA_FILE
    selector.DISEASE_MANIFEST_FILE = DISEASE_MANIFEST_FILE
    selector.SELECTED_FILE = SELECTED_FILE
    selector.AUDIT_FILE = AUDIT_FILE
    selector.SUMMARY_FILE = SUMMARY_FILE
    return selector.main()


if __name__ == "__main__":
    raise SystemExit(main())
