import csv
import math
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_fulltext_manifest.csv"
PMC_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "pmc"
MAX_WINDOW_TOKENS = 100
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


def histogram_median(frequencies: Counter, item_count: int):
    if not item_count:
        return 0

    def value_at_rank(rank):
        cumulative = 0
        for value in sorted(frequencies):
            cumulative += frequencies[value]
            if cumulative > rank:
                return value
        raise ValueError("Histogram counts do not match the item count")

    lower = value_at_rank((item_count - 1) // 2)
    upper = value_at_rank(item_count // 2)
    return (lower + upper) / 2


def profile_corpus(xml_files, parser, chunker, tokenizer):
    token_frequencies = Counter()
    window_frequencies = Counter()
    total_chunks = 0
    total_tokens = 0
    total_windows_per_chunk = 0
    minimum_tokens = None
    maximum_tokens = 0
    maximum_windows = 0
    negative_token_count_chunks = 0
    zero_window_chunks = 0
    chunks_with_at_least_one_window = 0
    parsed_articles = 0
    failures = []

    for xml_path in xml_files:
        try:
            article = parser.parse_file(xml_path)
            parsed_articles += 1
            chunks = chunker.chunk_article(article)
            if not chunks:
                continue

            tokenized_chunks = tokenizer(
                [chunk.text for chunk in chunks],
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                verbose=False,
            )["input_ids"]

            for chunk, token_ids in zip(chunks, tokenized_chunks):
                token_count = len(token_ids)
                window_count = math.ceil(token_count / MAX_WINDOW_TOKENS)
                total_chunks += 1
                total_tokens += token_count
                total_windows_per_chunk += window_count
                token_frequencies[token_count] += 1
                window_frequencies[window_count] += 1
                minimum_tokens = (
                    token_count if minimum_tokens is None else min(minimum_tokens, token_count)
                )
                maximum_tokens = max(maximum_tokens, token_count)
                maximum_windows = max(maximum_windows, window_count)
                negative_token_count_chunks += token_count < 0
                zero_window_chunks += window_count == 0
                chunks_with_at_least_one_window += window_count >= 1
        except Exception as error:
            failures.append((xml_path.name, type(error).__name__, str(error)))

    windows_from_distribution = sum(
        window_count * chunk_count
        for window_count, chunk_count in window_frequencies.items()
    )
    return {
        "parsed_articles": parsed_articles,
        "total_chunks": total_chunks,
        "total_tokens": total_tokens,
        "total_windows": total_windows_per_chunk,
        "minimum_tokens": minimum_tokens or 0,
        "median_tokens": histogram_median(token_frequencies, total_chunks),
        "maximum_tokens": maximum_tokens,
        "median_windows": histogram_median(window_frequencies, total_chunks),
        "maximum_windows": maximum_windows,
        "token_frequencies": token_frequencies,
        "window_frequencies": window_frequencies,
        "negative_token_count_chunks": negative_token_count_chunks,
        "zero_window_chunks": zero_window_chunks,
        "chunks_with_at_least_one_window": chunks_with_at_least_one_window,
        "windows_from_distribution": windows_from_distribution,
        "failures": failures,
    }


def print_profile(target_rows, xml_files, profile):
    total_chunks = profile["total_chunks"]
    total_windows = profile["total_windows"]
    one_window_chunks = profile["window_frequencies"][1]
    multiple_window_chunks = sum(
        count
        for window_count, count in profile["window_frequencies"].items()
        if window_count > 1
    )
    distribution = {
        "1 window": profile["window_frequencies"][1],
        "2 windows": profile["window_frequencies"][2],
        "3 windows": profile["window_frequencies"][3],
        "4 windows": profile["window_frequencies"][4],
        "5 windows": profile["window_frequencies"][5],
        ">5 windows": sum(
            count
            for window_count, count in profile["window_frequencies"].items()
            if window_count > 5
        ),
    }

    def percentage(count):
        return 100 * count / total_chunks if total_chunks else 0.0

    vector_bytes = total_windows * EMBEDDING_DIMENSION * FLOAT_BYTES
    vector_mib = vector_bytes / (1024 ** 2)
    vector_gib = vector_bytes / (1024 ** 3)

    print("CORPUS")
    print(f"Manifest PMC target rows: {target_rows}")
    print(f"Local manifest-backed XML files: {len(xml_files)}")
    print(f"Successfully parsed articles: {profile['parsed_articles']}")
    print(f"Total LiteratureChunks: {total_chunks}")

    print("\nTOKEN WINDOW PROFILE")
    print(f"Minimum tokens/chunk: {profile['minimum_tokens']}")
    print(f"Median tokens/chunk: {profile['median_tokens']}")
    print(f"Maximum tokens/chunk: {profile['maximum_tokens']}")
    print(f"Total token count across all chunks: {profile['total_tokens']}")
    print(f"Total embedding windows required: {total_windows}")
    print(
        "Average windows per LiteratureChunk: "
        f"{total_windows / total_chunks if total_chunks else 0.0:.2f}"
    )
    print(f"Median windows per LiteratureChunk: {profile['median_windows']}")
    print(f"Maximum windows for a single LiteratureChunk: {profile['maximum_windows']}")

    print("\nWINDOW DISTRIBUTION")
    for label, count in distribution.items():
        print(f"{label}: {count} ({percentage(count):.2f}%)")
    print(
        f"Chunks requiring exactly 1 window: {one_window_chunks} "
        f"({percentage(one_window_chunks):.2f}%)"
    )
    print(
        f"Chunks requiring >1 window: {multiple_window_chunks} "
        f"({percentage(multiple_window_chunks):.2f}%)"
    )

    print("\nESTIMATED EMBEDDING EXPANSION")
    print(f"Original LiteratureChunk count: {total_chunks}")
    print(f"Estimated embedding-vector count: {total_windows}")
    print(
        "Expansion factor (windows / original chunks): "
        f"{total_windows / total_chunks if total_chunks else 0.0:.4f}"
    )

    print("\nSTORAGE ESTIMATE (raw float32 vectors, 768 dimensions)")
    print(f"Raw vector memory: {vector_mib:.2f} MiB")
    print(f"Raw vector memory: {vector_gib:.4f} GiB")

    distribution_window_sum = profile["windows_from_distribution"]
    checks = {
        "no_negative_token_counts": profile["negative_token_count_chunks"] == 0,
        "no_zero_window_chunks": profile["zero_window_chunks"] == 0,
        "every_processed_chunk_contributes_at_least_one_window": (
            profile["chunks_with_at_least_one_window"] == total_chunks
        ),
        "total_windows_equals_sum_of_per_chunk_windows": total_windows == distribution_window_sum,
        "all_local_manifest_backed_articles_parsed": (
            profile["parsed_articles"] == len(xml_files) and not profile["failures"]
        ),
        "nonempty_corpus": total_chunks > 0,
    }

    print("\nINTEGRITY CHECKS")
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"Parse failures: {len(profile['failures'])}")
    print(f"Integrity check failures: {sum(not passed for passed in checks.values())}")
    return all(checks.values())


def main() -> int:
    try:
        target_rows, xml_files = load_manifest_pmcids()
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from symptom_rag_analyzer.data.literature_chunker import LiteratureSectionChunker
        from symptom_rag_analyzer.data.literature_parser import PMCLiteratureParser
        from symptom_rag_analyzer.embeddings.model import BiomedicalEmbeddingModel

        embedding_model = BiomedicalEmbeddingModel()
        profile = profile_corpus(
            xml_files,
            PMCLiteratureParser(),
            LiteratureSectionChunker(target_size=1000, overlap=150),
            embedding_model.model.tokenizer,
        )
        passed = print_profile(target_rows, xml_files, profile)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        print("\nEMBEDDING_WINDOW_PROFILE: FAIL")
        return 1

    if profile["failures"]:
        print("\nPARSE/TOKENIZATION FAILURES")
        for filename, error_type, message in profile["failures"][:20]:
            print(f"{filename}: {error_type}: {message}")
        if len(profile["failures"]) > 20:
            print(f"... and {len(profile['failures']) - 20} more")

    print(
        "\nEMBEDDING_WINDOW_PROFILE: "
        + ("PASS" if passed else "FAIL")
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())