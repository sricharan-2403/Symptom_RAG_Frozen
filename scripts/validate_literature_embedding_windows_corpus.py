import csv
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_fulltext_manifest.csv"
PMC_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "pmc"
MAX_MODEL_TOKENS = 100
EMBEDDING_DIMENSION = 768
FLOAT_BYTES = 4
REQUIRED_COLUMNS = {"pmcid", "acquisition_route"}


def load_manifest_pmcids() -> tuple[int, list[Path]]:
    if not MANIFEST_FILE.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}")

    target_rows = 0
    pmcids = set()
    with MANIFEST_FILE.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Manifest is missing required columns: " + ", ".join(sorted(missing))
            )
        for row in reader:
            if (row.get("acquisition_route") or "").strip().casefold() != "pmc":
                continue
            pmcid = (row.get("pmcid") or "").strip()
            if pmcid:
                target_rows += 1
                pmcids.add(pmcid)

    if not PMC_DIRECTORY.is_dir():
        raise FileNotFoundError(f"PMC XML directory not found: {PMC_DIRECTORY}")
    xml_files = [
        PMC_DIRECTORY / f"{pmcid}.xml"
        for pmcid in sorted(pmcids)
        if (PMC_DIRECTORY / f"{pmcid}.xml").is_file()
    ]
    return target_rows, xml_files


def mutable_container_ids(value, found=None):
    if found is None:
        found = set()
    if isinstance(value, dict):
        if id(value) in found:
            return found
        found.add(id(value))
        for key, item in value.items():
            mutable_container_ids(key, found)
            mutable_container_ids(item, found)
    elif isinstance(value, (list, set, bytearray)):
        if id(value) in found:
            return found
        found.add(id(value))
        for item in value:
            mutable_container_ids(item, found)
    elif isinstance(value, tuple):
        for item in value:
            mutable_container_ids(item, found)
    return found


def token_ids(tokenizer, text, add_special_tokens):
    return tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )["input_ids"]


def contains_sequence(sequence, candidate):
    if not candidate or len(candidate) > len(sequence):
        return False
    return any(
        sequence[start:start + len(candidate)] == candidate
        for start in range(len(sequence) - len(candidate) + 1)
    )


def validate_chunk(chunk, windows, tokenizer, max_model_tokens, model_input_counts):
    issues = []
    original_tokens = token_ids(tokenizer, chunk.text, add_special_tokens=False)
    window_content_sequences = []
    text_is_contiguous = "".join(window.text for window in windows) == chunk.text
    empty_window_count = 0

    if not windows:
        issues.append("no_windows")

    for window in windows:
        if not isinstance(window.text, str) or not window.text.strip():
            empty_window_count += 1
            issues.append("empty_window")
            window_content_sequences.append([])
            continue

        content_ids = token_ids(tokenizer, window.text, add_special_tokens=False)
        model_ids = token_ids(tokenizer, window.text, add_special_tokens=True)
        model_input_counts[len(model_ids)] += 1
        if len(model_ids) > max_model_tokens:
            issues.append("window_over_model_limit")
        window_content_sequences.append(content_ids)

    flattened_tokens = [
        token_id
        for sequence in window_content_sequences
        for token_id in sequence
    ]
    exact_coverage = flattened_tokens == original_tokens
    duplicate_or_overlap = False
    cursor = 0
    for sequence in window_content_sequences:
        expected = original_tokens[cursor:cursor + len(sequence)]
        if sequence != expected:
            max_overlap = min(cursor, len(sequence))
            for overlap_size in range(1, max_overlap + 1):
                if sequence[:overlap_size] == original_tokens[cursor - overlap_size:cursor]:
                    duplicate_or_overlap = True
                    break
            if contains_sequence(original_tokens[:cursor], sequence):
                duplicate_or_overlap = True
        cursor += len(sequence)

    if not text_is_contiguous:
        issues.append("window_text_not_contiguous")
    if not exact_coverage:
        issues.append("token_coverage_not_exact")
    if duplicate_or_overlap:
        issues.append("duplicated_or_overlapping_tokens")

    if [window.embedding_window_index for window in windows] != list(range(len(windows))):
        issues.append("window_indices_not_contiguous")

    provenance_fields = (
        ("pmid", chunk.pmid),
        ("pmcid", chunk.pmcid),
        ("doi", chunk.doi),
        ("section_title", chunk.section_title),
        ("normalized_section_title", chunk.normalized_section_title),
        ("section_path", chunk.section_path),
        ("category", chunk.category),
        ("source", chunk.source),
        ("literature_chunk_index", chunk.chunk_index),
        ("metadata", chunk.metadata),
    )
    if any(
        getattr(window, field_name) != expected
        for window in windows
        for field_name, expected in provenance_fields
    ):
        issues.append("provenance_mismatch")
    if any(window.embedding_window_count != len(windows) for window in windows):
        issues.append("window_count_mismatch")

    shared_mutable_ids = mutable_container_ids((chunk.section_path, chunk.metadata))
    for window in windows:
        path_ids = mutable_container_ids(window.section_path)
        metadata_ids = mutable_container_ids(window.metadata)
        if path_ids & shared_mutable_ids or metadata_ids & shared_mutable_ids:
            issues.append("mutable_provenance_shared_with_source")
        if path_ids & metadata_ids:
            issues.append("mutable_provenance_shared_within_window")
        if path_ids & metadata_ids & shared_mutable_ids:
            issues.append("mutable_provenance_shared_with_source")
        shared_mutable_ids.update(path_ids)
        shared_mutable_ids.update(metadata_ids)

    return {
        "issues": issues,
        "exact_coverage": exact_coverage,
        "duplicate_or_overlap": duplicate_or_overlap,
        "empty_window_count": empty_window_count,
        "window_count": len(windows),
    }


def histogram_median(frequencies, item_count):
    if not item_count:
        return 0

    def value_at_rank(rank):
        cumulative = 0
        for value in sorted(frequencies):
            cumulative += frequencies[value]
            if cumulative > rank:
                return value
        raise ValueError("Histogram counts do not match the item count")

    return (
        value_at_rank((item_count - 1) // 2)
        + value_at_rank(item_count // 2)
    ) / 2


def run_validation(xml_files, parser, chunker, splitter, tokenizer):
    parsed_articles = 0
    total_chunks = 0
    total_windows = 0
    parse_failures = 0
    chunk_failures = 0
    window_failures = 0
    integrity_failures = 0
    incomplete_coverage_chunks = 0
    duplicated_coverage_chunks = 0
    exact_coverage_chunks = 0
    empty_window_count = 0
    oversized_window_count = 0
    window_counts = Counter()
    model_input_counts = Counter()
    integrity_failure_kinds = Counter()
    failure_examples = []

    def remember_failure(kind, filename, error):
        failure_examples.append(f"{kind} {filename}: {type(error).__name__}: {error}")

    for xml_path in xml_files:
        try:
            article = parser.parse_file(xml_path)
            parsed_articles += 1
        except Exception as error:
            parse_failures += 1
            remember_failure("parse", xml_path.name, error)
            continue

        try:
            chunks = chunker.chunk_article(article)
        except Exception as error:
            chunk_failures += 1
            remember_failure("chunk", xml_path.name, error)
            continue

        for chunk_index, chunk in enumerate(chunks):
            total_chunks += 1
            try:
                windows = splitter.split_chunk(chunk)
            except Exception as error:
                window_failures += 1
                incomplete_coverage_chunks += 1
                remember_failure("window", f"{xml_path.name} chunk {chunk_index}", error)
                continue

            total_windows += len(windows)
            try:
                result = validate_chunk(
                    chunk,
                    windows,
                    tokenizer,
                    splitter.max_tokens,
                    model_input_counts,
                )
            except Exception as error:
                integrity_failures += 1
                incomplete_coverage_chunks += 1
                integrity_failure_kinds["validation_exception"] += 1
                remember_failure("integrity", f"{xml_path.name} chunk {chunk_index}", error)
                continue

            if result["exact_coverage"]:
                exact_coverage_chunks += 1
            else:
                incomplete_coverage_chunks += 1
            if result["duplicate_or_overlap"]:
                duplicated_coverage_chunks += 1
            empty_window_count += result["empty_window_count"]
            window_counts[result["window_count"]] += 1

            issue_counts = Counter(result["issues"])
            integrity_failures += sum(issue_counts.values())
            integrity_failure_kinds.update(issue_counts)
            if issue_counts:
                remember_failure(
                    "integrity",
                    f"{xml_path.name} chunk {chunk_index}",
                    ", ".join(sorted(issue_counts)),
                )

    oversized_window_count = sum(
        count
        for token_count, count in model_input_counts.items()
        if token_count > MAX_MODEL_TOKENS
    )
    return {
        "parsed_articles": parsed_articles,
        "total_chunks": total_chunks,
        "total_windows": total_windows,
        "parse_failures": parse_failures,
        "chunk_failures": chunk_failures,
        "window_failures": window_failures,
        "integrity_failures": integrity_failures,
        "incomplete_coverage_chunks": incomplete_coverage_chunks,
        "duplicated_coverage_chunks": duplicated_coverage_chunks,
        "exact_coverage_chunks": exact_coverage_chunks,
        "empty_window_count": empty_window_count,
        "oversized_window_count": oversized_window_count,
        "window_counts": window_counts,
        "model_input_counts": model_input_counts,
        "integrity_failure_kinds": integrity_failure_kinds,
        "failure_examples": failure_examples,
    }


def print_report(target_rows, xml_files, result, splitter):
    total_chunks = result["total_chunks"]
    total_windows = result["total_windows"]
    window_counts = result["window_counts"]
    model_input_counts = result["model_input_counts"]
    one_window_chunks = window_counts[1]
    multiple_window_chunks = sum(
        chunks for count, chunks in window_counts.items() if count > 1
    )
    window_distribution = {
        "1 window": window_counts[1],
        "2 windows": window_counts[2],
        "3 windows": window_counts[3],
        "4 windows": window_counts[4],
        "5 windows": window_counts[5],
        ">5 windows": sum(
            chunks for count, chunks in window_counts.items() if count > 5
        ),
    }
    model_input_window_total = sum(model_input_counts.values())
    vector_bytes = total_windows * EMBEDDING_DIMENSION * FLOAT_BYTES

    print("CORPUS")
    print(f"Manifest PMC target rows: {target_rows}")
    print(f"Local manifest-backed XML files: {len(xml_files)}")
    print(f"Successfully parsed articles: {result['parsed_articles']}")
    print(f"Total LiteratureChunks: {total_chunks}")

    print("\nWINDOW PROFILE")
    print(f"Total embedding windows: {total_windows}")
    print(f"Minimum windows/chunk: {min(window_counts, default=0)}")
    print(
        "Median windows/chunk: "
        f"{histogram_median(window_counts, sum(window_counts.values()))}"
    )
    print(f"Maximum windows/chunk: {max(window_counts, default=0)}")
    print(
        "Average windows/chunk: "
        f"{total_windows / total_chunks if total_chunks else 0.0:.2f}"
    )
    print(f"Chunks requiring exactly 1 window: {one_window_chunks}")
    print(f"Chunks requiring >1 window: {multiple_window_chunks}")

    print("\nWINDOW DISTRIBUTION")
    for label, count in window_distribution.items():
        print(f"{label}: {count}")

    print("\nMODEL INPUT VALIDATION")
    print(f"Maximum allowed model-input tokens: {splitter.max_tokens}")
    print(f"Tokenizer special tokens per input: {splitter.special_token_count}")
    print(
        "Minimum model-input token count: "
        f"{min(model_input_counts, default=0)}"
    )
    print(
        "Median model-input token count: "
        f"{histogram_median(model_input_counts, model_input_window_total)}"
    )
    print(
        "Maximum model-input token count: "
        f"{max(model_input_counts, default=0)}"
    )
    print(f"Count of windows >100 model tokens: {result['oversized_window_count']}")
    print(f"Count of empty windows: {result['empty_window_count']}")

    print("\nTOKEN COVERAGE")
    print(
        "Chunks with incomplete token coverage: "
        f"{result['incomplete_coverage_chunks']}"
    )
    print(
        "Chunks with duplicated/overlapping token coverage: "
        f"{result['duplicated_coverage_chunks']}"
    )
    print(f"Chunks with exact coverage: {result['exact_coverage_chunks']}")

    print("\nFAILURE SUMMARY")
    print(f"Parse failures: {result['parse_failures']}")
    print(f"Chunk failures: {result['chunk_failures']}")
    print(f"Window failures: {result['window_failures']}")
    print(f"Integrity failures: {result['integrity_failures']}")
    if result["integrity_failure_kinds"]:
        print("Integrity failure kinds:")
        for name, count in sorted(result["integrity_failure_kinds"].items()):
            print(f"  {name}: {count}")

    print("\nACTUAL EMBEDDING EXPANSION")
    print(f"Original LiteratureChunks: {total_chunks}")
    print(f"Actual embedding windows: {total_windows}")
    print(
        "Expansion factor: "
        f"{total_windows / total_chunks if total_chunks else 0.0:.4f}"
    )

    print("\nSTORAGE ESTIMATE (raw float32 vectors, 768 dimensions)")
    print(f"Raw vector memory: {vector_bytes / (1024 ** 2):.2f} MiB")
    print(f"Raw vector memory: {vector_bytes / (1024 ** 3):.4f} GiB")

    window_distribution_count = sum(window_counts.values())
    window_distribution_total = sum(
        window_count * chunk_count
        for window_count, chunk_count in window_counts.items()
    )
    aggregate_integrity_failures = (
        int(window_distribution_count != total_chunks)
        + int(window_distribution_total != total_windows)
    )
    integrity_failures = result["integrity_failures"] + aggregate_integrity_failures
    print(f"Integrity failures: {integrity_failures}")

    integrity_ok = (
        result["parsed_articles"] == len(xml_files)
        and result["total_chunks"] > 0
        and result["exact_coverage_chunks"] == total_chunks
        and result["incomplete_coverage_chunks"] == 0
        and result["duplicated_coverage_chunks"] == 0
        and result["empty_window_count"] == 0
        and result["oversized_window_count"] == 0
        and result["parse_failures"] == 0
        and result["chunk_failures"] == 0
        and result["window_failures"] == 0
        and integrity_failures == 0
        and window_distribution_count == total_chunks
        and total_windows == model_input_window_total
        and window_distribution_total == total_windows
    )
    return integrity_ok


def main() -> int:
    try:
        target_rows, xml_files = load_manifest_pmcids()
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from symptom_rag_analyzer.data.literature_chunker import LiteratureSectionChunker
        from symptom_rag_analyzer.data.literature_parser import PMCLiteratureParser
        from symptom_rag_analyzer.embeddings.literature_embedding_windows import (
            LiteratureEmbeddingWindowSplitter,
        )
        from symptom_rag_analyzer.embeddings.model import BiomedicalEmbeddingModel

        embedding_model = BiomedicalEmbeddingModel()
        splitter = LiteratureEmbeddingWindowSplitter(embedding_model)
        if splitter.max_tokens != MAX_MODEL_TOKENS:
            raise ValueError(
                f"Expected a {MAX_MODEL_TOKENS}-token model limit, "
                f"got {splitter.max_tokens}"
            )

        result = run_validation(
            xml_files,
            PMCLiteratureParser(),
            LiteratureSectionChunker(target_size=1000, overlap=150),
            splitter,
            embedding_model.model.tokenizer,
        )
        passed = print_report(target_rows, xml_files, result, splitter)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        print("\nLITERATURE_EMBEDDING_WINDOW_CORPUS_VALIDATION: FAIL")
        return 1

    if result["failure_examples"]:
        print("\nFAILURE DETAILS (first 20)")
        for failure in result["failure_examples"][:20]:
            print(failure)
        if len(result["failure_examples"]) > 20:
            print(f"... and {len(result['failure_examples']) - 20} more")

    print(
        "\nLITERATURE_EMBEDDING_WINDOW_CORPUS_VALIDATION: "
        + ("PASS" if passed else "FAIL")
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())