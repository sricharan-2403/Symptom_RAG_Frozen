import argparse
import csv
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "pubmed_fulltext_manifest.csv"
DEFAULT_PMC_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "pmc"
DEFAULT_QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"
PRODUCTION_COLLECTION = "literature_chunks"
FORBIDDEN_COLLECTIONS = {"symptom_chunks", "literature_chunks_benchmark"}
EMBEDDING_DIMENSION = 768


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally embed all manifest-backed local PMC literature and upsert "
            "the vectors into the dedicated literature_chunks collection."
        )
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Maximum literature windows per embedding/upsert batch (default: 32)",
    )
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=10,
        help="Print progress every N embedding batches (default: 10)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Authoritative PMC manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--pmc-directory",
        type=Path,
        default=DEFAULT_PMC_DIRECTORY,
        help=f"Directory containing PMC XML files (default: {DEFAULT_PMC_DIRECTORY})",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=DEFAULT_QDRANT_PATH,
        help=f"Persistent Qdrant directory (default: {DEFAULT_QDRANT_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse/count the corpus and tokenizer windows without embeddings or Qdrant writes",
    )
    return parser


def load_manifest_xml_files(manifest_path, pmc_directory):
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    pmcids = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required_columns = {"pmcid", "acquisition_route"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Manifest is missing required columns: " + ", ".join(sorted(missing))
            )
        for row in reader:
            if (row.get("acquisition_route") or "").strip().casefold() != "pmc":
                continue
            pmcid = (row.get("pmcid") or "").strip()
            if pmcid:
                pmcids.add(pmcid)

    if not pmc_directory.is_dir():
        raise FileNotFoundError(f"PMC XML directory not found: {pmc_directory}")
    return [
        pmc_directory / f"{pmcid}.xml"
        for pmcid in sorted(pmcids)
        if (pmc_directory / f"{pmcid}.xml").is_file()
    ]


def inspect_qdrant_read_only(qdrant_path, target_collection):
    if target_collection in FORBIDDEN_COLLECTIONS:
        raise ValueError(f"Refusing forbidden Qdrant target collection: {target_collection}")
    if target_collection != PRODUCTION_COLLECTION:
        raise ValueError(
            f"Resolved target must be {PRODUCTION_COLLECTION!r}, got {target_collection!r}"
        )

    client = QdrantClient(path=str(qdrant_path))
    try:
        if not client.collection_exists("symptom_chunks"):
            raise RuntimeError("Required existing symptom_chunks collection is missing")
        symptom_count = client.count("symptom_chunks", exact=True).count

        literature_exists = client.collection_exists(PRODUCTION_COLLECTION)
        literature_count = (
            client.count(PRODUCTION_COLLECTION, exact=True).count
            if literature_exists
            else 0
        )
        if literature_exists:
            info = client.get_collection(PRODUCTION_COLLECTION)
            vectors = info.config.params.vectors
            if not isinstance(vectors, VectorParams):
                raise RuntimeError("literature_chunks must use a single unnamed vector")
            if vectors.size != EMBEDDING_DIMENSION:
                raise RuntimeError(
                    f"literature_chunks dimension is {vectors.size}, "
                    f"expected {EMBEDDING_DIMENSION}"
                )
            if vectors.distance != Distance.COSINE:
                raise RuntimeError("literature_chunks must use cosine distance")

        return {
            "target_collection": PRODUCTION_COLLECTION,
            "symptom_count": symptom_count,
            "literature_existed": literature_exists,
            "literature_count": literature_count,
        }
    finally:
        client.close()


def count_corpus_work(xml_files, parser, chunker, window_splitter):
    totals = {
        "articles": 0,
        "chunks": 0,
        "windows": 0,
        "failures": 0,
    }
    for xml_path in xml_files:
        try:
            article = parser.parse_file(xml_path)
        except Exception as error:
            totals["failures"] += 1
            print(
                f"Failure PMID=(unavailable) PMCID={xml_path.stem} "
                f"stage=parse error={type(error).__name__}: {error}",
                file=sys.stderr,
            )
            continue
        totals["articles"] += 1
        try:
            chunks = chunker.chunk_article(article)
        except Exception as error:
            totals["failures"] += 1
            print(
                f"Failure PMID={article.pmid or '(missing)'} "
                f"PMCID={article.pmcid or xml_path.stem} "
                f"stage=chunk error={type(error).__name__}: {error}",
                file=sys.stderr,
            )
            continue
        for chunk in chunks:
            totals["chunks"] += 1
            try:
                totals["windows"] += len(window_splitter.split_chunk(chunk))
            except Exception as error:
                totals["failures"] += 1
                print(
                    f"Failure PMID={chunk.pmid or '(missing)'} "
                    f"PMCID={chunk.pmcid or xml_path.stem} "
                    f"stage=window chunk_index={chunk.chunk_index} "
                    f"error={type(error).__name__}: {error}",
                    file=sys.stderr,
                )
    return totals


def print_progress(
    result,
    totals,
    started,
    *,
    force=False,
):
    elapsed = max(time.monotonic() - started, 0.0)
    windows_per_second = result.embedding_windows / elapsed if elapsed > 0 else 0.0
    windows_remaining = max(totals["windows"] - result.embedding_windows, 0)
    eta_seconds = windows_remaining / windows_per_second if windows_per_second else None
    eta = "unknown" if eta_seconds is None else f"{eta_seconds / 3600:.2f}h"
    print(
        f"Progress: articles={result.articles_processed}/{totals['articles']}, "
        f"chunks={result.chunks_processed}/{totals['chunks']}, "
        f"windows={result.embedding_windows}/{totals['windows']}, "
        f"embeddings={result.embeddings_generated}, "
        f"vectors_upserted={result.vectors_upserted}, "
        f"elapsed={elapsed / 3600:.2f}h, windows/sec={windows_per_second:.2f}, "
        f"ETA={eta}"
    )


class ProgressLiteratureIndexer:
    """Add total-aware progress reporting around the production indexer."""

    def __init__(self, indexer, totals):
        self.indexer = indexer
        self.totals = totals
        self.embedding_batch_size = indexer.embedding_batch_size
        self.collection_name = indexer.vector_store.collection_name

    def index_corpus(self):
        original_reporter = self.indexer._report_progress

        def report(batch_number, result, started, *, force=False):
            if not force and batch_number % self.indexer.progress_every_batches:
                return
            print_progress(result, self.totals, started, force=force)

        self.indexer._report_progress = report
        try:
            return self.indexer.index_corpus()
        finally:
            self.indexer._report_progress = original_reporter


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    if args.embedding_batch_size <= 0:
        print("ERROR: --embedding-batch-size must be positive", file=sys.stderr)
        return 2
    if args.progress_every_batches <= 0:
        print("ERROR: --progress-every-batches must be positive", file=sys.stderr)
        return 2

    target_collection = PRODUCTION_COLLECTION
    try:
        qdrant_state = inspect_qdrant_read_only(args.qdrant_path, target_collection)
        xml_files = load_manifest_xml_files(args.manifest, args.pmc_directory)

        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from symptom_rag_analyzer.data.literature_chunker import LiteratureSectionChunker
        from symptom_rag_analyzer.data.literature_parser import PMCLiteratureParser
        from symptom_rag_analyzer.embeddings.literature_embedding_windows import (
            LiteratureEmbeddingWindowSplitter,
        )
        from symptom_rag_analyzer.embeddings.model import BiomedicalEmbeddingModel
        from symptom_rag_analyzer.ingestion.literature_embedding_indexer import (
            LiteratureEmbeddingIndexer,
        )

        embedding_model = BiomedicalEmbeddingModel()
        parser = PMCLiteratureParser()
        chunker = LiteratureSectionChunker(target_size=1000, overlap=150)
        splitter = LiteratureEmbeddingWindowSplitter(embedding_model)
        preflight_started = time.monotonic()
        totals = count_corpus_work(xml_files, parser, chunker, splitter)
        preflight_elapsed = time.monotonic() - preflight_started

        print(f"Target collection: {target_collection}")
        print("Qdrant safety: symptom_chunks exists; benchmark collection is not a target")
        print(f"symptom_chunks point count before run: {qdrant_state['symptom_count']}")
        print(
            f"literature_chunks point count before run: "
            f"{qdrant_state['literature_count']}"
        )
        print(f"Manifest-backed local PMC XML files: {len(xml_files)}")
        print(f"Successfully parsed articles in preflight: {totals['articles']}")
        print(f"Expected LiteratureChunks: {totals['chunks']}")
        print(f"Expected LiteratureEmbeddingWindows: {totals['windows']}")
        print(f"Preflight failures: {totals['failures']}")
        print(f"Preflight elapsed: {preflight_elapsed:.2f} seconds")
        print(f"Embedding batch size: {args.embedding_batch_size}")

        if args.dry_run:
            after_dry_run = inspect_qdrant_read_only(args.qdrant_path, target_collection)
            if after_dry_run != qdrant_state:
                raise RuntimeError("Qdrant state changed during dry run")
            if totals["failures"]:
                print("LITERATURE_FULL_INGESTION_DRY_RUN: FAIL")
                return 1
            print("Embeddings generated: 0")
            print("Qdrant writes: 0")
            print("Dry-run Qdrant safety: symptom_chunks unchanged; literature_chunks unchanged")
            print("\nLITERATURE_FULL_INGESTION_DRY_RUN: PASS")
            return 0

        if totals["failures"]:
            raise RuntimeError(
                "Preflight found failures; refusing to begin production writes"
            )
        indexer = LiteratureEmbeddingIndexer(
            embedding_batch_size=args.embedding_batch_size,
            progress_every_batches=args.progress_every_batches,
            manifest_file=args.manifest,
            pmc_directory=args.pmc_directory,
            qdrant_path=args.qdrant_path,
            collection_name=PRODUCTION_COLLECTION,
            embedding_model=embedding_model,
            window_splitter=splitter,
            parser=parser,
            chunker=chunker,
        )
        progress_indexer = ProgressLiteratureIndexer(indexer, totals)
        result = progress_indexer.index_corpus()
        final_count = indexer.vector_store.client.count(
            PRODUCTION_COLLECTION,
            exact=True,
        ).count
        symptom_count_after = indexer.vector_store.client.count(
            "symptom_chunks",
            exact=True,
        ).count

        print("\nFINAL INGESTION SUMMARY")
        print(f"Articles processed: {result.articles_processed}/{totals['articles']}")
        print(f"Chunks processed: {result.chunks_processed}/{totals['chunks']}")
        print(f"Windows generated: {result.embedding_windows}")
        print(f"Embeddings generated: {result.embeddings_generated}")
        print(f"Vectors upserted: {result.vectors_upserted}")
        print(f"Failures: {result.failures}")
        print(f"Elapsed time: {result.elapsed_seconds / 3600:.2f} hours")
        print(
            "Average throughput: "
            f"{result.embedding_windows / result.elapsed_seconds:.2f} windows/sec"
            if result.elapsed_seconds > 0
            else "Average throughput: 0.00 windows/sec"
        )
        print(f"Final literature_chunks point count: {final_count}")
        print(f"symptom_chunks point count after run: {symptom_count_after}")

        passed = (
            result.failures == 0
            and totals["failures"] == 0
            and result.articles_processed == totals["articles"]
            and result.chunks_processed == totals["chunks"]
            and result.embedding_windows == totals["windows"]
            and result.embedding_windows
            == result.embeddings_generated
            == result.vectors_upserted
            and result.collection_name == PRODUCTION_COLLECTION
            and target_collection == PRODUCTION_COLLECTION
            and target_collection not in FORBIDDEN_COLLECTIONS
            and symptom_count_after == qdrant_state["symptom_count"]
        )
        print(
            "\nLITERATURE_FULL_INGESTION: "
            + ("PASS" if passed else "FAIL")
        )
        return 0 if passed else 1
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        print("\nLITERATURE_FULL_INGESTION: FAIL")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
