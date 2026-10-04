"""Export production literature embedding windows without generating vectors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


from symptom_rag_analyzer.data.literature_chunker import LiteratureSectionChunker
from symptom_rag_analyzer.data.literature_parser import PMCLiteratureParser
from symptom_rag_analyzer.embeddings.literature_embedding_windows import (
    LiteratureEmbeddingWindow,
    LiteratureEmbeddingWindowSplitter,
)
from symptom_rag_analyzer.embeddings.model import BiomedicalEmbeddingModel
from symptom_rag_analyzer.ingestion.literature_embedding_indexer import (
    LiteratureEmbeddingIndexer,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "pubmed_fulltext_manifest.csv"
DEFAULT_PMC_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "pmc"

MODEL_IDENTIFIER = (
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
)

SCHEMA_VERSION = 1


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export production literature embedding windows and Qdrant payloads "
            "without generating embeddings or writing to Qdrant."
        )
    )

    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pmc-directory", type=Path, default=DEFAULT_PMC_DIRECTORY)

    return parser


def load_manifest_xml_files(
    manifest_path: Path,
    pmc_directory: Path,
) -> list[Path]:
    """Resolve PMC XML files using the same manifest rules as production ingestion."""

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    pmcids = set()

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as manifest_file:
        reader = csv.DictReader(manifest_file)

        required_columns = {"pmcid", "acquisition_route"}
        missing = required_columns - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                "Manifest is missing required columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            if (
                row.get("acquisition_route") or ""
            ).strip().casefold() != "pmc":
                continue

            pmcid = (row.get("pmcid") or "").strip()

            if pmcid:
                pmcids.add(pmcid)

    if not pmc_directory.is_dir():
        raise FileNotFoundError(
            f"PMC XML directory not found: {pmc_directory}"
        )

    return [
        pmc_directory / f"{pmcid}.xml"
        for pmcid in sorted(pmcids)
        if (pmc_directory / f"{pmcid}.xml").is_file()
    ]


def load_production_window_splitter(
) -> tuple[LiteratureEmbeddingWindowSplitter, dict[str, Any]]:
    """Use the same initialized SentenceTransformer tokenizer as production."""

    from huggingface_hub import hf_hub_download

    embedding_model = BiomedicalEmbeddingModel(MODEL_IDENTIFIER)

    sentence_transformer = embedding_model.model

    tokenizer = sentence_transformer.tokenizer
    max_seq_length = sentence_transformer.max_seq_length

    transformer_module = sentence_transformer[0]

    revision = getattr(
        transformer_module.auto_model.config,
        "_commit_hash",
        None,
    )

    if not revision:
        raise ValueError(
            "Loaded SentenceTransformer has no resolved model revision"
        )

    sentence_config_path = Path(
        hf_hub_download(
            repo_id=MODEL_IDENTIFIER,
            filename="sentence_bert_config.json",
            revision=revision,
        )
    )

    sentence_config = json.loads(
        sentence_config_path.read_text(
            encoding="utf-8"
        )
    )

    pooling_config_path = Path(
        hf_hub_download(
            repo_id=MODEL_IDENTIFIER,
            filename="1_Pooling/config.json",
            revision=revision,
        )
    )

    pooling_config = json.loads(
        pooling_config_path.read_text(
            encoding="utf-8"
        )
    )

    vector_dimension = pooling_config.get(
        "word_embedding_dimension"
    )

    if not isinstance(vector_dimension, int) or vector_dimension <= 0:
        raise ValueError(
            "Model pooling config has no valid word_embedding_dimension"
        )

    if not tokenizer.is_fast:
        raise ValueError(
            "The production window splitter requires a fast tokenizer"
        )

    if not isinstance(max_seq_length, int) or max_seq_length <= 0:
        raise ValueError(
            "Production model max_seq_length must be a positive integer"
        )

    # The splitter only reads model.tokenizer and model.max_seq_length.
    tokenizer_host = type(
        "TokenizerHost",
        (),
        {
            "tokenizer": tokenizer,
            "max_seq_length": max_seq_length,
        },
    )()

    embedding_model_view = type(
        "EmbeddingModelView",
        (),
        {"model": tokenizer_host},
    )()

    splitter = LiteratureEmbeddingWindowSplitter(
        embedding_model_view
    )

    tokenizer_fingerprint = hashlib.sha256(
        tokenizer.backend_tokenizer.to_str().encode("utf-8")
    ).hexdigest()

    configuration = {
        "model_revision": revision,
        "vector_dimension": vector_dimension,
        "sentence_bert_config": sentence_config,
        "pooling_config": pooling_config,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": tokenizer.is_fast,
        "tokenizer_do_lower_case": getattr(
            tokenizer,
            "do_lower_case",
            None,
        ),
        "tokenizer_sha256": tokenizer_fingerprint,
        "max_seq_length": splitter.max_tokens,
        "special_token_count": splitter.special_token_count,
        "content_token_capacity": splitter.content_token_capacity,
    }

    return splitter, configuration


class _JsonlShardWriter:
    def __init__(
        self,
        output_dir: Path,
        shard_size: int,
    ) -> None:
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.handle = None
        self.current_count = 0
        self.shard_index = -1
        self.shard_hash = None
        self.shards: list[dict[str, Any]] = []

    def write(self, record: dict[str, Any]) -> None:
        if (
            self.handle is None
            or self.current_count == self.shard_size
        ):
            self._open_next_shard()

        line = (
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )

        self.handle.write(line)

        self.shard_hash.update(
            line.encode("utf-8")
        )

        self.current_count += 1

    def _open_next_shard(self) -> None:
        self._close_current_shard()

        self.shard_index += 1
        self.current_count = 0
        self.shard_hash = hashlib.sha256()

        shard_path = (
            self.output_dir
            / f"windows-{self.shard_index:05d}.jsonl"
        )

        self.handle = shard_path.open(
            "x",
            encoding="utf-8",
            newline="\n",
        )

    def _close_current_shard(self) -> None:
        if self.handle is None:
            return

        self.handle.close()

        self.shards.append(
            {
                "file": (
                    f"windows-{self.shard_index:05d}.jsonl"
                ),
                "record_count": self.current_count,
                "sha256": self.shard_hash.hexdigest(),
            }
        )

        self.handle = None

    def close(self) -> None:
        self._close_current_shard()


def _window_record(
    article: Any,
    window: LiteratureEmbeddingWindow,
) -> dict[str, Any]:
    return {
        "vector_id": LiteratureEmbeddingIndexer.deterministic_window_id(
            window
        ),
        "window": {
            "text": window.text,
            "pmid": window.pmid,
            "pmcid": window.pmcid,
            "doi": window.doi,
            "section_title": window.section_title,
            "normalized_section_title": (
                window.normalized_section_title
            ),
            "section_path": window.section_path,
            "category": window.category,
            "source": window.source,
            "literature_chunk_index": (
                window.literature_chunk_index
            ),
            "embedding_window_index": (
                window.embedding_window_index
            ),
            "embedding_window_count": (
                window.embedding_window_count
            ),
            "metadata": window.metadata,
        },
        "payload": LiteratureEmbeddingIndexer.make_payload(
            article,
            window,
        ),
    }


def export_windows(
    *,
    xml_files: list[Path],
    output_dir: Path,
    shard_size: int,
    limit: int | None,
    dry_run: bool,
    parser: Any,
    chunker: Any,
    window_splitter: LiteratureEmbeddingWindowSplitter,
    model_configuration: dict[str, Any],
) -> dict[str, Any]:
    """Stream corpus windows to shards while counting the complete corpus."""

    if shard_size <= 0:
        raise ValueError("--shard-size must be positive")

    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")

    if not dry_run:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}"
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    writer = (
        None
        if dry_run
        else _JsonlShardWriter(
            output_dir,
            shard_size,
        )
    )

    articles = 0
    chunks = 0
    windows_count = 0
    exported = 0
    duplicate_windows = 0

    seen_ids: set[str] = set()

    try:
        for xml_path in xml_files:
            article = parser.parse_file(xml_path)
            articles += 1

            for chunk in chunker.chunk_article(article):
                chunks += 1

                for window in window_splitter.split_chunk(
                    chunk
                ):
                    windows_count += 1

                    if (
                        limit is not None
                        and exported >= limit
                    ):
                        continue

                    record = _window_record(
                        article,
                        window,
                    )

                    vector_id = record["vector_id"]

                    # Some production windows produce the same
                    # deterministic UUID because the identity function
                    # does not include a section-instance identifier.
                    #
                    # Keep the first occurrence and skip later
                    # duplicates so every exported vector_id is unique.
                    if vector_id in seen_ids:
                        duplicate_windows += 1
                        continue

                    seen_ids.add(vector_id)

                    if writer is not None:
                        writer.write(record)

                    exported += 1

    finally:
        if writer is not None:
            writer.close()

    shards = (
        []
        if writer is None
        else writer.shards
    )

    manifest = {
        "schema": (
            "symptom-rag-literature-embedding-windows"
        ),
        "schema_version": SCHEMA_VERSION,
        "model_identifier": MODEL_IDENTIFIER,
        "model_revision": model_configuration[
            "model_revision"
        ],
        "expected_total_articles": articles,
        "expected_total_literature_chunks": chunks,
        "expected_total_embedding_windows": windows_count,
        "number_of_shards": (
            len(shards)
            if not dry_run
            else math.ceil(
                exported / shard_size
            )
        ),
        "shard_size": shard_size,
        "export_limit": limit,
        "total_exported_windows": exported,
        "duplicate_windows_skipped": duplicate_windows,
        "vector_dimension": model_configuration[
            "vector_dimension"
        ],
        "tokenizer_window_configuration": {
            key: value
            for key, value in model_configuration.items()
            if key not in {
                "model_revision",
                "vector_dimension",
            }
        },
        "creation_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "deterministic_ordering": (
            "manifest PMCIDs sorted lexicographically; "
            "parser article order; "
            "chunker output order; "
            "embedding_window_index ascending within "
            "each chunk"
        ),
        "shards": shards,
    }

    if not dry_run:
        manifest_path = (
            output_dir / "manifest.json"
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    return manifest


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)

    try:
        xml_files = load_manifest_xml_files(
            args.manifest,
            args.pmc_directory,
        )

        (
            window_splitter,
            model_configuration,
        ) = load_production_window_splitter()

        manifest = export_windows(
            xml_files=xml_files,
            output_dir=args.output_dir,
            shard_size=args.shard_size,
            limit=args.limit,
            dry_run=args.dry_run,
            parser=PMCLiteratureParser(),
            chunker=LiteratureSectionChunker(
                target_size=1000,
                overlap=150,
            ),
            window_splitter=window_splitter,
            model_configuration=model_configuration,
        )

    except Exception as error:
        print(
            f"ERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(
        f"Manifest-backed local PMC XML files: "
        f"{len(xml_files)}"
    )

    print(
        f"Expected LiteratureArticles: "
        f"{manifest['expected_total_articles']}"
    )

    print(
        f"Expected LiteratureChunks: "
        f"{manifest['expected_total_literature_chunks']}"
    )

    print(
        "Expected LiteratureEmbeddingWindows: "
        f"{manifest['expected_total_embedding_windows']}"
    )

    print(
        f"Exported windows: "
        f"{manifest['total_exported_windows']}"
    )

    print(
        f"Duplicate windows skipped: "
        f"{manifest['duplicate_windows_skipped']}"
    )

    print(
        f"Shards: "
        f"{manifest['number_of_shards']}"
    )

    print(
        "Embedding model instantiated: yes; "
        "embeddings generated: no"
    )

    print(
        "Qdrant accessed or modified: no"
    )

    if args.dry_run:
        print(
            "Dry run: no output files written"
        )
    else:
        print(
            f"Manifest: "
            f"{args.output_dir / 'manifest.json'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())