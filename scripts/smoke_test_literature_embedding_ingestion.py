import csv
import sys
from pathlib import Path

from qdrant_client import QdrantClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_fulltext_manifest.csv"
PMC_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "pmc"
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"
SAMPLE_CHUNK_COUNT = 50


def manifest_xml_files():
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
    return [
        PMC_DIRECTORY / f"{pmcid}.xml"
        for pmcid in sorted(pmcids)
        if (PMC_DIRECTORY / f"{pmcid}.xml").is_file()
    ]


def select_first_chunks(xml_files, parser, chunker):
    selected = []
    parsed_article_ids = set()
    for xml_path in xml_files:
        article = parser.parse_file(xml_path)
        article_chunks = chunker.chunk_article(article)
        if article_chunks:
            parsed_article_ids.add((article.pmid, article.pmcid or xml_path.stem))
        for chunk in article_chunks:
            selected.append((article, chunk))
            if len(selected) == SAMPLE_CHUNK_COUNT:
                return selected, len(parsed_article_ids)
    raise RuntimeError(
        f"Expected {SAMPLE_CHUNK_COUNT} chunks; found only {len(selected)}"
    )


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


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
            LITERATURE_COLLECTION,
            LiteratureEmbeddingIndexer,
        )
        from symptom_rag_analyzer.vector_store.qdrant_store import QdrantVectorStore

        xml_files = manifest_xml_files()
        article_parser = PMCLiteratureParser()
        article_chunker = LiteratureSectionChunker(target_size=1000, overlap=150)
        records, articles_selected = select_first_chunks(
            xml_files,
            article_parser,
            article_chunker,
        )
        embedding_model = BiomedicalEmbeddingModel()
        splitter = LiteratureEmbeddingWindowSplitter(embedding_model)
        all_windows = [
            (article, window)
            for article, chunk in records
            for window in splitter.split_chunk(chunk)
        ]
        point_ids = [
            LiteratureEmbeddingIndexer.deterministic_window_id(window)
            for _article, window in all_windows
        ]
        require(len(point_ids) == len(set(point_ids)), "Smoke sample contains duplicate IDs")

        client = QdrantClient(path=str(QDRANT_PATH))
        require(
            client.collection_exists("symptom_chunks"),
            "Existing symptom_chunks collection is missing; refusing smoke ingestion",
        )
        symptom_count_before = client.count("symptom_chunks", exact=True).count
        vector_store = QdrantVectorStore(
            collection_name=LITERATURE_COLLECTION,
            vector_size=embedding_model.embedding_dim,
            client=client,
        )
        require(
            vector_store.collection_name == "literature_chunks",
            "Smoke test did not select literature_chunks",
        )
        collection_before = client.count(LITERATURE_COLLECTION, exact=True).count

        indexer = LiteratureEmbeddingIndexer(
            embedding_batch_size=32,
            progress_every_batches=1,
            embedding_model=embedding_model,
            window_splitter=splitter,
            parser=article_parser,
            chunker=article_chunker,
            vector_store=vector_store,
            qdrant_path=QDRANT_PATH,
        )
        first_result = indexer.index_chunk_records(
            records,
            articles_processed=articles_selected,
        )
        require(first_result.is_successful, f"First ingestion failed: {first_result}")
        require(first_result.chunks_processed == SAMPLE_CHUNK_COUNT, "Did not process 50 chunks")
        require(first_result.embedding_windows == len(all_windows), "Window count mismatch")
        require(
            first_result.embedding_windows == first_result.embeddings_generated,
            "Generated embeddings do not match generated windows",
        )
        require(
            first_result.vectors_upserted == first_result.embedding_windows,
            "Upserted vector count does not match generated windows",
        )

        stored_points = vector_store.retrieve_by_ids(point_ids)
        require(len(stored_points) == len(point_ids), "Not all smoke vectors can be read back")
        for stored in stored_points:
            require(stored.vector is not None, "Retrieved point is missing its vector")
            require(
                len(stored.vector) == 768,
                f"Retrieved vector dimension is {len(stored.vector)}, expected 768",
            )
            payload = stored.payload or {}
            for required_key in (
                "text",
                "pmid",
                "pmcid",
                "doi",
                "title",
                "journal",
                "publication_date",
                "publication_year",
                "record_type",
                "article_type",
                "publication_types",
                "authors",
                "mesh_terms",
                "section_title",
                "normalized_section_title",
                "section_path",
                "category",
                "source",
                "literature_chunk_index",
                "embedding_window_index",
                "embedding_window_count",
                "source_file",
                "source_type",
                "chunk_metadata",
                "window_metadata",
            ):
                require(required_key in payload, f"Missing provenance payload field: {required_key}")
        retrieved_ids = {str(point.id) for point in stored_points}
        require(retrieved_ids == set(point_ids), "Read-back IDs do not match deterministic IDs")

        first_count_for_ids = len(stored_points)
        first_collection_count = client.count(LITERATURE_COLLECTION, exact=True).count
        require(
            first_collection_count >= collection_before,
            "Literature collection count unexpectedly decreased",
        )

        second_result = indexer.index_chunk_records(
            records,
            articles_processed=articles_selected,
        )
        require(second_result.is_successful, f"Repeat ingestion failed: {second_result}")
        second_points = vector_store.retrieve_by_ids(point_ids)
        second_collection_count = client.count(LITERATURE_COLLECTION, exact=True).count
        require(
            len(second_points) == first_count_for_ids,
            "Repeat upsert changed the number of points for the deterministic sample IDs",
        )
        require(
            second_collection_count == first_collection_count,
            "Repeat upsert increased collection count instead of replacing the same IDs",
        )
        require(
            client.count("symptom_chunks", exact=True).count == symptom_count_before,
            "symptom_chunks count changed during the literature smoke test",
        )

        print("LITERATURE EMBEDDING INGESTION SMOKE TEST")
        print(f"Literature chunks processed per run: {first_result.chunks_processed}")
        print(f"Articles represented: {first_result.articles_processed}")
        print(f"Embedding windows generated: {first_result.embedding_windows}")
        print(f"Embeddings generated: {first_result.embeddings_generated}")
        print(f"Vectors upserted: {first_result.vectors_upserted}")
        print(f"Repeat-run vectors upserted: {second_result.vectors_upserted}")
        print(f"Collection used: {vector_store.collection_name}")
        print(f"Vectors read back: {len(stored_points)}")
        print(f"Literature collection count after first run: {first_collection_count}")
        print(f"Literature collection count after repeat: {second_collection_count}")
        print(f"Symptom collection count unchanged: {symptom_count_before}")
        print("Duplicate deterministic IDs in sample: 0")
        print("\nLITERATURE_EMBEDDING_INGESTION_SMOKE_TEST: PASS")
        return 0
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        print("\nLITERATURE_EMBEDDING_INGESTION_SMOKE_TEST: FAIL")
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())