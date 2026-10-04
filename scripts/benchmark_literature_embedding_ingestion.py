import csv
import ctypes
import os
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_fulltext_manifest.csv"
PMC_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "pmc"
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"
BENCHMARK_COLLECTION = "literature_chunks_benchmark"
PROTECTED_COLLECTIONS = ("symptom_chunks", "literature_chunks")
SAMPLE_CHUNK_COUNT = 1000
FULL_CORPUS_WINDOW_COUNT = 201_139


def load_xml_files():
    if not MANIFEST_FILE.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}")
    pmcids = set()
    with MANIFEST_FILE.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required = {"pmcid", "acquisition_route"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Manifest is missing required columns: " + ", ".join(sorted(missing))
            )
        for row in reader:
            if (row.get("acquisition_route") or "").strip().casefold() == "pmc":
                pmcid = (row.get("pmcid") or "").strip()
                if pmcid:
                    pmcids.add(pmcid)
    if not PMC_DIRECTORY.is_dir():
        raise FileNotFoundError(f"PMC XML directory not found: {PMC_DIRECTORY}")
    return [
        PMC_DIRECTORY / f"{pmcid}.xml"
        for pmcid in sorted(pmcids)
        if (PMC_DIRECTORY / f"{pmcid}.xml").is_file()
    ]


def select_chunks(xml_files, parser, chunker):
    records = []
    articles = set()
    for xml_path in xml_files:
        article = parser.parse_file(xml_path)
        for chunk in chunker.chunk_article(article):
            records.append((article, chunk))
            articles.add((article.pmid, article.pmcid or xml_path.stem))
            if len(records) == SAMPLE_CHUNK_COUNT:
                return records, len(articles)
    raise RuntimeError(
        f"Expected {SAMPLE_CHUNK_COUNT} chunks, found only {len(records)}"
    )


def collection_count(client, collection_name):
    if not client.collection_exists(collection_name):
        return None
    return client.count(collection_name, exact=True).count


def peak_process_memory_bytes():
    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.argtypes = []
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL
        succeeded = get_process_memory_info(
            get_current_process(),
            ctypes.byref(counters),
            counters.cb,
        )
        if not succeeded:
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def format_duration(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def retrieve_and_validate(store, point_ids, expected_payloads):
    retrieved = 0
    for start in range(0, len(point_ids), 256):
        batch_ids = point_ids[start:start + 256]
        points = store.retrieve_by_ids(batch_ids, with_vectors=True, with_payload=True)
        points_by_id = {str(point.id): point for point in points}
        if set(points_by_id) != set(batch_ids):
            raise RuntimeError("Qdrant readback IDs do not match deterministic window IDs")
        for point_id, point in points_by_id.items():
            if point.vector is None or len(point.vector) != 768:
                raise RuntimeError(f"Point {point_id} does not have a 768-dimensional vector")
            payload = point.payload or {}
            for key in ("text", "pmid", "pmcid", "section_path", "literature_chunk_index"):
                if payload.get(key) != expected_payloads[point_id].get(key):
                    raise RuntimeError(f"Point {point_id} has invalid {key} provenance")
        retrieved += len(points)
    return retrieved


def main():
    client = None
    try:
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
        from symptom_rag_analyzer.vector_store.qdrant_store import QdrantVectorStore

        parser = PMCLiteratureParser()
        chunker = LiteratureSectionChunker(target_size=1000, overlap=150)
        records, selected_articles = select_chunks(load_xml_files(), parser, chunker)
        if len(records) != SAMPLE_CHUNK_COUNT:
            raise RuntimeError("Selection did not return exactly 1,000 LiteratureChunks")

        embedding_model = BiomedicalEmbeddingModel()
        splitter = LiteratureEmbeddingWindowSplitter(embedding_model)
        client = QdrantClient(path=str(QDRANT_PATH))
        protected_before = {
            name: collection_count(client, name)
            for name in PROTECTED_COLLECTIONS
        }
        if protected_before["symptom_chunks"] is None:
            raise RuntimeError("symptom_chunks is missing; refusing to run benchmark")

        store = QdrantVectorStore(
            collection_name=BENCHMARK_COLLECTION,
            vector_size=embedding_model.embedding_dim,
            client=client,
        )
        benchmark_count_before = collection_count(client, BENCHMARK_COLLECTION)
        indexer = LiteratureEmbeddingIndexer(
            collection_name=BENCHMARK_COLLECTION,
            embedding_model=embedding_model,
            window_splitter=splitter,
            parser=parser,
            chunker=chunker,
            vector_store=store,
        )

        batch_times = []
        point_ids = []
        original_flush = indexer._flush_batch

        def timed_flush(batch, result):
            point_ids.extend(point_id for point_id, _window, _article in batch)
            started = time.perf_counter()
            try:
                return original_flush(batch, result)
            finally:
                batch_times.append(time.perf_counter() - started)

        indexer._flush_batch = timed_flush
        started = time.perf_counter()
        first_result = indexer.index_chunk_records(
            records,
            articles_processed=selected_articles,
        )
        elapsed_seconds = time.perf_counter() - started
        if first_result.chunks_processed != SAMPLE_CHUNK_COUNT:
            raise RuntimeError(f"Only processed {first_result.chunks_processed} chunks")
        if not first_result.is_successful:
            raise RuntimeError(f"Production indexer reported failure: {first_result}")
        if len(point_ids) != len(set(point_ids)):
            raise RuntimeError("Duplicate deterministic IDs were generated")
        if not (
            first_result.embedding_windows
            == first_result.embeddings_generated
            == first_result.vectors_upserted
        ):
            raise RuntimeError("Window, embedding, and upsert counts differ")

        expected_payloads = {}
        for article, chunk in records:
            for window in splitter.split_chunk(chunk):
                point_id = LiteratureEmbeddingIndexer.deterministic_window_id(window)
                expected_payloads[point_id] = LiteratureEmbeddingIndexer.make_payload(article, window)
        if set(expected_payloads) != set(point_ids):
            raise RuntimeError("Stored IDs do not match all selected embedding windows")
        readback_count = retrieve_and_validate(store, point_ids, expected_payloads)
        if readback_count != first_result.embeddings_generated:
            raise RuntimeError("Readback count does not match generated embeddings")
        benchmark_count_after_first = collection_count(client, BENCHMARK_COLLECTION)

        point_ids.clear()
        repeat_result = indexer.index_chunk_records(
            records,
            articles_processed=selected_articles,
        )
        if not repeat_result.is_successful:
            raise RuntimeError(f"Repeat indexer run failed: {repeat_result}")
        if point_ids != list(expected_payloads):
            raise RuntimeError("Deterministic rerun generated different IDs or ordering")
        benchmark_count_after_repeat = collection_count(client, BENCHMARK_COLLECTION)
        if benchmark_count_after_repeat != benchmark_count_after_first:
            raise RuntimeError("Deterministic rerun increased benchmark collection count")

        protected_after = {
            name: collection_count(client, name)
            for name in PROTECTED_COLLECTIONS
        }
        if protected_after != protected_before:
            raise RuntimeError(
                f"Production collections changed: before={protected_before}, after={protected_after}"
            )

        if elapsed_seconds <= 0:
            raise RuntimeError("Benchmark elapsed time is not positive")
        windows_per_second = first_result.embedding_windows / elapsed_seconds
        projected_seconds = FULL_CORPUS_WINDOW_COUNT / windows_per_second
        average_batch_seconds = sum(batch_times) / len(batch_times) if batch_times else 0.0
        maximum_batch_seconds = max(batch_times, default=0.0)
        peak_memory = peak_process_memory_bytes()

        print("LITERATURE EMBEDDING INGESTION PERFORMANCE BENCHMARK")
        print(f"Deterministic LiteratureChunks selected: {first_result.chunks_processed}")
        print(f"Articles represented: {selected_articles}")
        print(f"Production embedding batch size: {indexer.embedding_batch_size}")
        print(f"Total chunks processed: {first_result.chunks_processed}")
        print(f"Total embedding windows generated: {first_result.embedding_windows}")
        print(f"Total embeddings generated: {first_result.embeddings_generated}")
        print(f"Total vectors upserted: {first_result.vectors_upserted}")
        print(f"Elapsed time: {elapsed_seconds:.2f} seconds")
        print(f"Chunks/second: {first_result.chunks_processed / elapsed_seconds:.2f}")
        print(f"Windows/second: {windows_per_second:.2f}")
        print(
            "Embeddings/second: "
            f"{first_result.embeddings_generated / elapsed_seconds:.2f}"
        )
        print(f"Average batch time: {average_batch_seconds:.3f} seconds")
        print(f"Maximum observed batch time: {maximum_batch_seconds:.3f} seconds")
        print(
            "Estimated full-corpus runtime for 201,139 windows "
            f"(benchmark-based estimate): {format_duration(projected_seconds)} "
            f"({projected_seconds:.1f} seconds)"
        )
        print(f"Peak process working set: {peak_memory / (1024 ** 2):.2f} MiB")
        print(f"Read-back vectors validated: {readback_count}")
        print(f"Benchmark collection after first run: {benchmark_count_after_first}")
        print(f"Benchmark collection after rerun: {benchmark_count_after_repeat}")
        print(f"Production collection counts unchanged: {protected_after}")
        print(f"Qdrant collection used: {BENCHMARK_COLLECTION}")
        print("Duplicate IDs: 0")
        print("Embedding dimension: 768 for every generated vector")
        print("Failures: 0")
        print("\nLITERATURE_EMBEDDING_PERFORMANCE_BENCHMARK: PASS")
        return 0
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        print("\nLITERATURE_EMBEDDING_PERFORMANCE_BENCHMARK: FAIL")
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
