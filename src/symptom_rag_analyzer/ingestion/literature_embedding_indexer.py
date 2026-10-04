"""Incrementally embed literature windows into a dedicated Qdrant collection."""

from __future__ import annotations

import csv
import json
import sys
import time
import uuid
from collections.abc import Iterable, Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from symptom_rag_analyzer.data.literature_chunker import LiteratureSectionChunker
from symptom_rag_analyzer.data.literature_chunks import LiteratureChunk
from symptom_rag_analyzer.data.literature_models import LiteratureArticle
from symptom_rag_analyzer.data.literature_parser import PMCLiteratureParser
from symptom_rag_analyzer.embeddings.literature_embedding_windows import (
	LiteratureEmbeddingWindow,
	LiteratureEmbeddingWindowSplitter,
)
from symptom_rag_analyzer.embeddings.model import BiomedicalEmbeddingModel
from symptom_rag_analyzer.vector_store.qdrant_store import QdrantVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_fulltext_manifest.csv"
DEFAULT_PMC_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "pmc"
DEFAULT_QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"
LITERATURE_COLLECTION = "literature_chunks"


@dataclass
class LiteratureIndexingResult:
	articles_processed: int = 0
	chunks_processed: int = 0
	embedding_windows: int = 0
	embeddings_generated: int = 0
	vectors_upserted: int = 0
	failures: int = 0
	collection_name: str = LITERATURE_COLLECTION
	elapsed_seconds: float = 0.0

	@property
	def is_successful(self) -> bool:
		return (
			self.failures == 0
			and self.embedding_windows == self.embeddings_generated
			and self.embeddings_generated == self.vectors_upserted
		)


class LiteratureEmbeddingIndexer:
	"""Stream parsed literature chunks through windowing, embedding, and Qdrant."""

	def __init__(
		self,
		*,
		embedding_batch_size: int = 32,
		progress_every_batches: int = 10,
		manifest_file: str | Path = DEFAULT_MANIFEST_FILE,
		pmc_directory: str | Path = DEFAULT_PMC_DIRECTORY,
		qdrant_path: str | Path = DEFAULT_QDRANT_PATH,
		collection_name: str = LITERATURE_COLLECTION,
		embedding_model: BiomedicalEmbeddingModel | None = None,
		window_splitter: LiteratureEmbeddingWindowSplitter | None = None,
		parser: PMCLiteratureParser | None = None,
		chunker: LiteratureSectionChunker | None = None,
		vector_store: QdrantVectorStore | None = None,
	) -> None:
		if embedding_batch_size <= 0:
			raise ValueError("embedding_batch_size must be positive")
		if progress_every_batches <= 0:
			raise ValueError("progress_every_batches must be positive")
		if collection_name == "symptom_chunks":
			raise ValueError("literature indexing cannot target symptom_chunks")

		self.embedding_model = embedding_model or BiomedicalEmbeddingModel()
		self.window_splitter = window_splitter or LiteratureEmbeddingWindowSplitter(
			self.embedding_model
		)
		self.parser = parser or PMCLiteratureParser()
		self.chunker = chunker or LiteratureSectionChunker(target_size=1000, overlap=150)
		self.vector_store = vector_store or QdrantVectorStore(
			collection_name=collection_name,
			vector_size=self.embedding_model.embedding_dim,
			path=qdrant_path,
		)
		if self.vector_store.collection_name == "symptom_chunks":
			raise ValueError("literature indexing cannot target symptom_chunks")
		if self.vector_store.collection_name != collection_name:
			raise ValueError(
				f"Literature indexing was configured for {collection_name!r}, "
				f"but the vector store targets {self.vector_store.collection_name!r}"
			)
		if self.vector_store.vector_size != self.embedding_model.embedding_dim:
			raise ValueError("Qdrant vector size must match the embedding model dimension")

		self.embedding_batch_size = embedding_batch_size
		self.progress_every_batches = progress_every_batches
		self.manifest_file = Path(manifest_file)
		self.pmc_directory = Path(pmc_directory)

	@staticmethod
	def deterministic_window_id(window: LiteratureEmbeddingWindow) -> str:
		"""Return a stable UUID incorporating article, section, chunk, and window identity."""
		identity = {
			"pmid": window.pmid,
			"pmcid": window.pmcid or "",
			"section_path": window.section_path,
			"section_title": window.section_title,
			"normalized_section_title": window.normalized_section_title,
			"category": window.category,
			"source": window.source,
			"literature_chunk_index": window.literature_chunk_index,
			"embedding_window_index": window.embedding_window_index,
			"text": window.text,
		}
		canonical_identity = json.dumps(
			identity,
			sort_keys=True,
			separators=(",", ":"),
			ensure_ascii=False,
		)
		return str(uuid.uuid5(uuid.NAMESPACE_URL, canonical_identity))

	@staticmethod
	def make_payload(
		article: LiteratureArticle,
		window: LiteratureEmbeddingWindow,
	) -> dict[str, Any]:
		return {
			"text": window.text,
			"pmid": window.pmid,
			"pmcid": window.pmcid,
			"doi": window.doi,
			"title": article.title,
			"journal": article.journal,
			"publication_date": article.publication_date,
			"publication_year": article.publication_year,
			"record_type": article.record_type,
			"article_type": article.article_type,
			"publication_types": list(article.publication_types),
			"authors": list(article.authors),
			"mesh_terms": list(article.mesh_terms),
			"section_title": window.section_title,
			"normalized_section_title": window.normalized_section_title,
			"section_path": list(window.section_path),
			"category": window.category,
			"source": window.source,
			"literature_chunk_index": window.literature_chunk_index,
			"embedding_window_index": window.embedding_window_index,
			"embedding_window_count": window.embedding_window_count,
			"source_file": article.source_file,
			"source_type": article.source_type,
			"chunk_metadata": deepcopy(window.metadata),
			"window_metadata": {
				**deepcopy(window.metadata),
				"literature_chunk_index": window.literature_chunk_index,
				"embedding_window_index": window.embedding_window_index,
				"embedding_window_count": window.embedding_window_count,
			},
		}

	def index_corpus(self) -> LiteratureIndexingResult:
		"""Parse the authoritative local PMC manifest and index it incrementally."""
		result = LiteratureIndexingResult(collection_name=self.vector_store.collection_name)
		started = time.monotonic()
		try:
			xml_files = self._manifest_xml_files()
		except Exception as error:
			result.failures += 1
			print(f"Manifest failure: {type(error).__name__}: {error}", file=sys.stderr)
			result.elapsed_seconds = time.monotonic() - started
			return result

		self._index_records(self._iter_chunk_records(xml_files, result), result, started)
		return result

	def index_chunk_records(
		self,
		records: Iterable[tuple[LiteratureArticle, LiteratureChunk]],
		*,
		articles_processed: int = 0,
	) -> LiteratureIndexingResult:
		"""Index supplied article/chunk pairs; useful for bounded smoke runs."""
		result = LiteratureIndexingResult(
			articles_processed=articles_processed,
			collection_name=self.vector_store.collection_name,
		)
		self._index_records(records, result, time.monotonic())
		return result

	def _manifest_xml_files(self) -> list[Path]:
		if not self.manifest_file.is_file():
			raise FileNotFoundError(f"Manifest not found: {self.manifest_file}")
		pmcids = set()
		with self.manifest_file.open("r", encoding="utf-8-sig", newline="") as manifest_file:
			reader = csv.DictReader(manifest_file)
			required = {"pmcid", "acquisition_route"}
			missing = required - set(reader.fieldnames or [])
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

		if not self.pmc_directory.is_dir():
			raise FileNotFoundError(f"PMC XML directory not found: {self.pmc_directory}")
		return [
			self.pmc_directory / f"{pmcid}.xml"
			for pmcid in sorted(pmcids)
			if (self.pmc_directory / f"{pmcid}.xml").is_file()
		]

	def _iter_chunk_records(
		self,
		xml_files: Sequence[Path],
		result: LiteratureIndexingResult,
	) -> Iterator[tuple[LiteratureArticle, LiteratureChunk]]:
		for xml_path in xml_files:
			try:
				article = self.parser.parse_file(xml_path)
			except Exception as error:
				result.failures += 1
				print(
					f"Article parse failure PMCID={xml_path.stem}: "
					f"{type(error).__name__}: {error}",
					file=sys.stderr,
				)
				continue
			result.articles_processed += 1
			try:
				chunks = self.chunker.chunk_article(article)
			except Exception as error:
				result.failures += 1
				print(
					f"Article chunk failure PMID={article.pmid} PMCID={article.pmcid or xml_path.stem}: "
					f"{type(error).__name__}: {error}",
					file=sys.stderr,
				)
				continue
			for chunk in chunks:
				yield article, chunk

	def _index_records(
		self,
		records: Iterable[tuple[LiteratureArticle, LiteratureChunk]],
		result: LiteratureIndexingResult,
		started: float,
	) -> None:
		batch = []
		seen_point_ids = set()
		batch_number = 0
		for article, chunk in records:
			result.chunks_processed += 1
			try:
				windows = self.window_splitter.split_chunk(chunk)
			except Exception as error:
				result.failures += 1
				print(
					f"Window failure PMID={article.pmid} PMCID={article.pmcid or '(missing)'} "
					f"chunk_index={chunk.chunk_index}: {type(error).__name__}: {error}",
					file=sys.stderr,
				)
				continue

			result.embedding_windows += len(windows)
			for window in windows:
				point_id = self.deterministic_window_id(window)
				if point_id in seen_point_ids:
					result.failures += 1
					print(
						f"Duplicate deterministic window ID {point_id} for "
						f"PMID={article.pmid} PMCID={article.pmcid or '(missing)'}",
						file=sys.stderr,
					)
					continue
				seen_point_ids.add(point_id)
				batch.append((point_id, window, article))
				if len(batch) >= self.embedding_batch_size:
					self._flush_batch(batch, result)
					batch_number += 1
					self._report_progress(batch_number, result, started)

		if batch:
			self._flush_batch(batch, result)
			batch_number += 1
			self._report_progress(batch_number, result, started, force=True)

		result.elapsed_seconds = time.monotonic() - started
		if (
			result.embedding_windows != result.embeddings_generated
			or result.embeddings_generated != result.vectors_upserted
		):
			result.failures += 1
			print(
				"Count mismatch: "
			f"windows={result.embedding_windows}, "
			f"embeddings={result.embeddings_generated}, "
			f"vectors_upserted={result.vectors_upserted}",
				file=sys.stderr,
			)

	def _flush_batch(self, batch, result: LiteratureIndexingResult) -> None:
		point_ids = [point_id for point_id, _window, _article in batch]
		texts = [window.text for _point_id, window, _article in batch]
		payloads = [
			self.make_payload(article, window)
			for _point_id, window, article in batch
		]
		first_window = batch[0][1]
		try:
			embeddings = np.asarray(
				self.embedding_model.embed_batch(texts),
				dtype=np.float32,
			)
			if embeddings.shape != (len(batch), self.embedding_model.embedding_dim):
				raise ValueError(
					f"Expected embeddings with shape "
					f"({len(batch)}, {self.embedding_model.embedding_dim}), "
					f"got {embeddings.shape}"
				)
			result.embeddings_generated += len(embeddings)
			upserted = self.vector_store.upsert_vectors(
				point_ids,
				embeddings,
				payloads,
			)
			if upserted != len(batch):
				raise RuntimeError(
					f"Qdrant acknowledged {upserted} of {len(batch)} vectors"
				)
			result.vectors_upserted += upserted
		except Exception as error:
			result.failures += 1
			print(
				f"Embedding/upsert batch failure PMID={first_window.pmid} "
				f"PMCID={first_window.pmcid or '(missing)'} "
				f"batch_size={len(batch)}: {type(error).__name__}: {error}",
				file=sys.stderr,
			)
		finally:
			batch.clear()

	def _report_progress(
		self,
		batch_number: int,
		result: LiteratureIndexingResult,
		started: float,
		*,
		force: bool = False,
	) -> None:
		if not force and batch_number % self.progress_every_batches:
			return
		elapsed = time.monotonic() - started
		windows_per_second = result.vectors_upserted / elapsed if elapsed > 0 else 0.0
		print(
			f"Progress: articles={result.articles_processed}, "
			f"chunks={result.chunks_processed}, "
			f"windows={result.embedding_windows}, "
			f"embeddings={result.embeddings_generated}, "
			f"vectors_upserted={result.vectors_upserted}, "
			f"elapsed={elapsed:.1f}s, rate={windows_per_second:.2f} vectors/s"
		)