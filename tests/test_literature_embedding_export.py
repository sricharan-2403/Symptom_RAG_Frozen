import csv
import hashlib
import json
import re
from dataclasses import asdict
from types import SimpleNamespace

from symptom_rag_analyzer.data.literature_chunks import LiteratureChunk
from symptom_rag_analyzer.data.literature_models import LiteratureArticle
from symptom_rag_analyzer.embeddings import model as embedding_model_module
from symptom_rag_analyzer.embeddings.literature_embedding_windows import (
	LiteratureEmbeddingWindow,
	LiteratureEmbeddingWindowSplitter,
)
from symptom_rag_analyzer.ingestion.literature_embedding_indexer import (
	LiteratureEmbeddingIndexer,
)
from symptom_rag_analyzer.vector_store.qdrant_store import QdrantVectorStore
from scripts import export_literature_embedding_windows as exporter
from scripts.export_literature_embedding_windows import export_windows


class _Encoding(dict):
	def __init__(self, values, word_ids):
		super().__init__(values)
		self._word_ids = word_ids

	def word_ids(self):
		return self._word_ids


class _WhitespaceTokenizer:
	def __init__(self, do_lower_case=True):
		self.token_ids = {}
		self.do_lower_case = do_lower_case
		self.is_fast = True
		self.backend_tokenizer = SimpleNamespace(
			to_str=lambda: f"whitespace:{self.do_lower_case}"
		)

	def num_special_tokens_to_add(self, pair=False):
		return 2

	def __call__(
		self,
		text,
		*,
		add_special_tokens=True,
		return_offsets_mapping=False,
		**_kwargs,
	):
		matches = list(re.finditer(r"\S+", text))
		content_ids = [
			self.token_ids.setdefault(
				match.group().lower() if self.do_lower_case else match.group(),
				len(self.token_ids) + 1,
			)
			for match in matches
		]
		offsets = [(match.start(), match.end()) for match in matches]
		word_ids = list(range(len(matches)))
		input_ids = [101, *content_ids, 102] if add_special_tokens else content_ids
		encoding_word_ids = [None, *word_ids, None] if add_special_tokens else word_ids
		result = _Encoding({"input_ids": input_ids}, encoding_word_ids)
		if return_offsets_mapping:
			result["offset_mapping"] = (
				[(0, 0), *offsets, (0, 0)] if add_special_tokens else offsets
			)
		return result


def _article(pmcid):
	return LiteratureArticle(
		pmid=pmcid.removeprefix("PMC"),
		pmcid=pmcid,
		doi=f"10.1234/{pmcid}",
		title=f"Title {pmcid}",
		journal="Example Journal",
		publication_date="2024-01-02",
		publication_year=2024,
		record_type="article",
		article_type="research-article",
		publication_types=["Journal Article"],
		authors=["A. Author"],
		mesh_terms=["Example Term"],
		source_file=f"{pmcid}.xml",
	)


def _chunk(article, text, chunk_index):
	return LiteratureChunk(
		text=text,
		pmid=article.pmid,
		pmcid=article.pmcid,
		doi=article.doi,
		section_title="Results",
		normalized_section_title="results",
		section_path=["Results"],
		category="results",
		source="body",
		chunk_index=chunk_index,
		metadata={"source_file": article.source_file},
	)


class _Parser:
	def parse_file(self, path):
		return _article(path.stem)


class _Chunker:
	def chunk_article(self, article):
		return [_chunk(article, "Alpha BETA Gamma Delta", 0)]


def _components():
	tokenizer = _WhitespaceTokenizer()
	model_view = SimpleNamespace(
		model=SimpleNamespace(tokenizer=tokenizer, max_seq_length=4)
	)
	return LiteratureEmbeddingWindowSplitter(model_view), tokenizer


def _configuration():
	return {
		"model_revision": "unit-test-revision",
		"vector_dimension": 768,
		"sentence_bert_config": {"max_seq_length": 4, "do_lower_case": True},
		"pooling_config": {"word_embedding_dimension": 768},
		"tokenizer_class": "WhitespaceTokenizer",
		"tokenizer_is_fast": True,
		"tokenizer_do_lower_case": True,
		"tokenizer_sha256": "unit-test-tokenizer",
		"max_seq_length": 4,
		"special_token_count": 2,
		"content_token_capacity": 2,
	}


def _export(output_dir, *, limit=None, shard_size=2, dry_run=False):
	window_splitter, _tokenizer = _components()
	article_paths = [output_dir.parent / "PMC001.xml", output_dir.parent / "PMC002.xml"]
	for path in article_paths:
		path.touch(exist_ok=True)
	return export_windows(
		xml_files=article_paths,
		output_dir=output_dir,
		shard_size=shard_size,
		limit=limit,
		dry_run=dry_run,
		parser=_Parser(),
		chunker=_Chunker(),
		window_splitter=window_splitter,
		model_configuration=_configuration(),
	)


def _read_records(output_dir):
	records = []
	for shard in sorted(output_dir.glob("windows-*.jsonl")):
		with shard.open(encoding="utf-8") as shard_file:
			records.extend(json.loads(line) for line in shard_file)
	return records


def test_export_matches_production_window_objects_and_expected_count(tmp_path):
	output_dir = tmp_path / "export"
	manifest = _export(output_dir, limit=5, shard_size=2)
	window_splitter, _tokenizer = _components()
	article = _article("PMC001")
	expected_windows = window_splitter.split_chunk(
		_chunk(article, "Alpha BETA Gamma Delta", 0)
	)
	actual = _read_records(output_dir)

	assert manifest["expected_total_articles"] == 2
	assert manifest["expected_total_literature_chunks"] == 2
	assert manifest["expected_total_embedding_windows"] == 4
	assert manifest["total_exported_windows"] == len(actual) == 4
	assert manifest["number_of_shards"] == 2
	assert [record["window"] for record in actual[:2]] == [
		{**asdict(window), "section_path": window.section_path}
		for window in expected_windows
	]
	assert [record["window"]["text"] for record in actual[:2]] == [
		window.text for window in expected_windows
	]
	assert [record["vector_id"] for record in actual[:2]] == [
		LiteratureEmbeddingIndexer.deterministic_window_id(window)
		for window in expected_windows
	]
	assert [record["payload"] for record in actual[:2]] == [
		LiteratureEmbeddingIndexer.make_payload(article, window)
		for window in expected_windows
	]
	assert json.loads((output_dir / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_rerunning_export_preserves_window_ids_and_text_and_has_no_duplicates(tmp_path):
	first_dir = tmp_path / "first"
	second_dir = tmp_path / "second"
	_export(first_dir, shard_size=3)
	_export(second_dir, shard_size=3)
	first = _read_records(first_dir)
	second = _read_records(second_dir)
	first_ids = [record["vector_id"] for record in first]
	second_ids = [record["vector_id"] for record in second]
	first_texts = [record["window"]["text"] for record in first]
	second_texts = [record["window"]["text"] for record in second]

	assert first_ids == second_ids
	assert first_texts == second_texts
	assert len(first_ids) == len(set(first_ids))
	assert len(first) == 4


def test_dry_run_counts_full_corpus_without_writing_or_side_effects(
	tmp_path,
	monkeypatch,
):
	def forbidden_call(*_args, **_kwargs):
		raise AssertionError("dry run must not generate embeddings or access Qdrant")

	monkeypatch.setattr(embedding_model_module.BiomedicalEmbeddingModel, "embed_text", forbidden_call)
	monkeypatch.setattr(embedding_model_module.BiomedicalEmbeddingModel, "embed_batch", forbidden_call)
	monkeypatch.setattr(QdrantVectorStore, "__init__", forbidden_call)
	output_dir = tmp_path / "dry-run-output"
	manifest = _export(output_dir, dry_run=True)

	assert manifest["expected_total_embedding_windows"] == 4
	assert manifest["total_exported_windows"] == 4
	assert manifest["number_of_shards"] == 2
	assert not output_dir.exists()


def test_export_never_instantiates_embedding_or_qdrant_and_cannot_change_collections(
	tmp_path,
	monkeypatch,
):
	def forbidden_call(*_args, **_kwargs):
		raise AssertionError("embedding generation or Qdrant access is forbidden")

	monkeypatch.setattr(embedding_model_module.BiomedicalEmbeddingModel, "embed_text", forbidden_call)
	monkeypatch.setattr(embedding_model_module.BiomedicalEmbeddingModel, "embed_batch", forbidden_call)
	monkeypatch.setattr(QdrantVectorStore, "__init__", forbidden_call)
	collection_dir = tmp_path / "qdrant"
	for collection in ("symptom_chunks", "literature_chunks", "literature_chunks_benchmark"):
		path = collection_dir / collection
		path.mkdir(parents=True)
		(path / "sentinel").write_text(collection, encoding="utf-8")
	before = {
		path.relative_to(collection_dir): path.read_bytes()
		for path in collection_dir.rglob("*")
		if path.is_file()
	}

	_export(tmp_path / "export")

	after = {
		path.relative_to(collection_dir): path.read_bytes()
		for path in collection_dir.rglob("*")
		if path.is_file()
	}
	assert after == before


def test_exporter_uses_production_tokenizer_and_fingerprint(monkeypatch, tmp_path):
	tokenizer = _WhitespaceTokenizer(do_lower_case=True)
	production_model = SimpleNamespace(
		model=SimpleNamespace(
			tokenizer=tokenizer,
			max_seq_length=100,
			__getitem__=lambda _self, _index: SimpleNamespace(
				auto_model=SimpleNamespace(
					config=SimpleNamespace(_commit_hash="production-revision")
				)
			),
		),
		embedding_dim=768,
	)

	class ProductionSentenceTransformer(SimpleNamespace):
		def __getitem__(self, index):
			return SimpleNamespace(
				auto_model=SimpleNamespace(
					config=SimpleNamespace(_commit_hash="production-revision")
				)
			)

	production_model.model = ProductionSentenceTransformer(
		tokenizer=tokenizer, max_seq_length=100
	)
	monkeypatch.setattr(exporter, "BiomedicalEmbeddingModel", lambda _name: production_model)

	def fake_hub_download(*, filename, **_kwargs):
		content = (
			'{"max_seq_length": 100, "do_lower_case": false}'
			if filename == "sentence_bert_config.json"
			else '{"word_embedding_dimension": 768}'
		)
		path = tmp_path / filename.replace("/", "_")
		path.write_text(content, encoding="utf-8")
		return str(path)

	monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hub_download)
	splitter, configuration = exporter.load_production_window_splitter()
	production_fingerprint = hashlib.sha256(
		production_model.model.tokenizer.backend_tokenizer.to_str().encode("utf-8")
	).hexdigest()

	assert splitter.tokenizer is production_model.model.tokenizer
	assert splitter.max_tokens == production_model.model.max_seq_length == 100
	assert configuration["tokenizer_sha256"] == production_fingerprint
	assert configuration["tokenizer_do_lower_case"] is True
	assert tokenizer("Alpha")["input_ids"] == tokenizer("alpha")["input_ids"]
	assert configuration["content_token_capacity"] == 98
	assert configuration["special_token_count"] == 2


def test_manifest_csv_resolver_uses_sorted_manifest_pmcids(tmp_path):
	from scripts.export_literature_embedding_windows import load_manifest_xml_files

	manifest_path = tmp_path / "manifest.csv"
	pmc_directory = tmp_path / "pmc"
	pmc_directory.mkdir()
	with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
		writer = csv.DictWriter(manifest_file, fieldnames=["pmcid", "acquisition_route"])
		writer.writeheader()
		writer.writerows(
			[
				{"pmcid": "PMC2", "acquisition_route": "PMC"},
				{"pmcid": "PMC1", "acquisition_route": "pmc"},
				{"pmcid": "PMC3", "acquisition_route": "pubmed"},
			]
		)
	for pmcid in ("PMC1", "PMC2"):
		(pmc_directory / f"{pmcid}.xml").touch()

	assert load_manifest_xml_files(manifest_path, pmc_directory) == [
		pmc_directory / "PMC1.xml",
		pmc_directory / "PMC2.xml",
	]