import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_fulltext_manifest.csv"
PMC_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "pmc"
SAMPLE_SIZE = 500
SAMPLE_SEED = 20260930
REQUIRED_COLUMNS = {"pmcid", "acquisition_route"}


def load_manifest_pmcids() -> tuple[int, list[str]]:
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
    return target_rows, [str(path) for path in xml_files]


def collect_deterministic_sample(xml_files, parser, chunker):
    rng = random.Random(SAMPLE_SEED)
    sample = []
    total_chunks = 0
    failures = []

    for xml_filename in xml_files:
        xml_path = Path(xml_filename)
        try:
            article = parser.parse_file(xml_path)
            for chunk in chunker.chunk_article(article):
                total_chunks += 1
                item = (total_chunks, chunk, xml_path.stem)
                if len(sample) < SAMPLE_SIZE:
                    sample.append(item)
                    continue

                replacement_index = rng.randrange(total_chunks)
                if replacement_index < SAMPLE_SIZE:
                    sample[replacement_index] = item
        except Exception as error:
            failures.append((xml_path.name, type(error).__name__, str(error)))

    sample.sort(key=lambda item: item[0])
    return sample, total_chunks, failures


def tokenize_sample(sample, embedding_model):
    tokenizer = embedding_model.model.tokenizer
    max_seq_length = embedding_model.model.max_seq_length
    measurements = []

    for ordinal, chunk, source_pmcid in sample:
        token_ids_before = tokenizer(
            chunk.text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )["input_ids"]
        token_ids_after = tokenizer(
            chunk.text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_seq_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )["input_ids"]

        token_count = len(token_ids_before)
        truncated_token_count = len(token_ids_after)
        measurements.append({
            "ordinal": ordinal,
            "pmid": chunk.pmid,
            "pmcid": chunk.pmcid or source_pmcid,
            "section_title": chunk.section_title,
            "text": chunk.text,
            "character_count": len(chunk.text),
            "token_count": token_count,
            "truncated_token_count": truncated_token_count,
            "truncated": truncated_token_count < token_count,
        })

    return measurements, max_seq_length


def evenly_spaced_examples(records, count=5):
    if len(records) <= count:
        return records
    indexes = [round(index * (len(records) - 1) / (count - 1)) for index in range(count)]
    return [records[index] for index in indexes]


def print_examples(label, records):
    examples = evenly_spaced_examples(records)
    print(f"\n{label} ({len(examples)} selected from {len(records)} available)")
    for index, record in enumerate(examples, start=1):
        print(
            f"{index}. PMID={record['pmid'] or '(missing)'}"
            f" | PMCID={record['pmcid'] or '(missing)'}"
            f" | section={record['section_title'] or '(untitled)'}"
            f" | characters={record['character_count']}"
            f" | tokens={record['token_count']}"
            f" | excerpt={record['text'][:250]!r}"
        )


def print_profile(target_rows, xml_files, total_chunks, failures, records, max_seq_length):
    sample_count = len(records)
    character_counts = [record["character_count"] for record in records]
    token_counts = [record["token_count"] for record in records]
    truncated_count = sum(record["truncated"] for record in records)
    fitting_count = sample_count - truncated_count
    buckets = Counter()

    for token_count in token_counts:
        if token_count <= 50:
            buckets["<=50 tokens"] += 1
        elif token_count <= 100:
            buckets["51-100 tokens"] += 1
        elif token_count <= 150:
            buckets["101-150 tokens"] += 1
        elif token_count <= 200:
            buckets["151-200 tokens"] += 1
        else:
            buckets[">200 tokens"] += 1

    def percentage(count):
        return 100 * count / sample_count if sample_count else 0.0

    print("LITERATURE EMBEDDING TOKEN PROFILE")
    print(f"Manifest PMC target rows: {target_rows}")
    print(f"Local manifest-backed XML files: {len(xml_files)}")
    print(f"Chunks found in parsed local XMLs: {total_chunks}")
    print(f"Articles with parse/chunk failures: {len(failures)}")
    print(f"Deterministic sample size: {sample_count}")
    print(f"Reservoir sample seed: {SAMPLE_SEED}")
    print(f"Effective model token limit: {max_seq_length}")

    print("\nCHARACTER LENGTHS")
    print(f"Minimum: {min(character_counts, default=0)}")
    print(f"Median: {median(character_counts) if character_counts else 0}")
    print(f"Maximum: {max(character_counts, default=0)}")

    print("\nTOKEN COUNTS BEFORE TRUNCATION")
    print(f"Minimum: {min(token_counts, default=0)}")
    print(f"Median: {median(token_counts) if token_counts else 0}")
    print(f"Maximum: {max(token_counts, default=0)}")
    truncated_counts = [record["truncated_token_count"] for record in records]
    print("TOKEN COUNTS AFTER TRUNCATION")
    print(f"Minimum: {min(truncated_counts, default=0)}")
    print(f"Median: {median(truncated_counts) if truncated_counts else 0}")
    print(f"Maximum: {max(truncated_counts, default=0)}")

    print("\nTOKEN-LENGTH BUCKETS (based on counts before truncation)")
    for label in ("<=50 tokens", "51-100 tokens", "101-150 tokens", "151-200 tokens", ">200 tokens"):
        count = buckets[label]
        print(f"{label}: {count} ({percentage(count):.2f}%)")

    print(
        f"Chunks exceeding {max_seq_length} tokens: "
        f"{truncated_count} ({percentage(truncated_count):.2f}%)"
    )
    print(
        f"Chunks fitting completely within {max_seq_length} tokens: "
        f"{fitting_count} ({percentage(fitting_count):.2f}%)"
    )

    print_examples(
        f"EXAMPLES FITTING WITHIN {max_seq_length} TOKENS",
        [record for record in records if not record["truncated"]],
    )
    print_examples(
        f"EXAMPLES EXCEEDING {max_seq_length} TOKENS",
        [record for record in records if record["truncated"]],
    )

    return (
        sample_count == min(SAMPLE_SIZE, total_chunks)
        and sample_count > 0
        and not failures
        and fitting_count >= 5
        and truncated_count >= 5
        and all(record["truncated_token_count"] <= max_seq_length for record in records)
    )


def main() -> int:
    try:
        target_rows, xml_files = load_manifest_pmcids()
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from symptom_rag_analyzer.data.literature_chunker import LiteratureSectionChunker
        from symptom_rag_analyzer.data.literature_parser import PMCLiteratureParser
        from symptom_rag_analyzer.embeddings.model import BiomedicalEmbeddingModel

        parser = PMCLiteratureParser()
        chunker = LiteratureSectionChunker(target_size=1000, overlap=150)
        sample, total_chunks, failures = collect_deterministic_sample(
            xml_files, parser, chunker
        )
        embedding_model = BiomedicalEmbeddingModel()
        records, max_seq_length = tokenize_sample(sample, embedding_model)
        passed = print_profile(
            target_rows,
            xml_files,
            total_chunks,
            failures,
            records,
            max_seq_length,
        )
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        print("\nEMBEDDING_TOKEN_PROFILE: FAIL")
        return 1

    if failures:
        print("\nPARSE/CHUNK FAILURES")
        for filename, error_type, message in failures[:20]:
            print(f"{filename}: {error_type}: {message}")
        if len(failures) > 20:
            print(f"... and {len(failures) - 20} more")

    print(
        "\nEMBEDDING_TOKEN_PROFILE: "
        + ("PASS" if passed else "FAIL")
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())