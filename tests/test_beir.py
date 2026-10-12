"""Tests for BEIR dataset conversion.

Builds a miniature BEIR archive on disk, so conversion is tested without
network access: title joining, split selection, qrels parsing, round-tripping
through the harness's own loader, and checksum verification.
"""

import json
import zipfile

import pytest

from retrieval_eval import beir, data

CORPUS = [
    {"_id": "d1", "title": "Aspirin", "text": "reduces fever.", "metadata": {}},
    {"_id": "d2", "title": "", "text": "  untitled abstract ", "metadata": {}},
    {"_id": "d3", "title": "Statins", "text": "lower cholesterol.", "metadata": {}},
]
QUERIES = [
    {"_id": "q1", "text": "aspirin and fever", "metadata": {}},
    {"_id": "q2", "text": "a training claim", "metadata": {}},
    {"_id": "q3", "text": "statins and cholesterol", "metadata": {}},
]
TEST_QRELS = "query-id\tcorpus-id\tscore\nq1\td1\t1\nq3\td3\t1\nq3\td2\t1\n"
TRAIN_QRELS = "query-id\tcorpus-id\tscore\nq2\td2\t1\n"


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "mini.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mini/corpus.jsonl", "".join(json.dumps(r) + "\n" for r in CORPUS))
        zf.writestr(
            "mini/queries.jsonl", "".join(json.dumps(r) + "\n" for r in QUERIES)
        )
        zf.writestr("mini/qrels/test.tsv", TEST_QRELS)
        zf.writestr("mini/qrels/train.tsv", TRAIN_QRELS)
    return path


def test_join_title_matches_beir():
    assert beir.join_title("Aspirin", "reduces fever.") == "Aspirin reduces fever."
    assert beir.join_title("", "  body only ") == "body only"


def test_read_qrels_skips_header_and_parses_relevance():
    assert beir.read_qrels(TEST_QRELS.splitlines()) == [
        {"query_id": "q1", "doc_id": "d1", "relevance": 1},
        {"query_id": "q3", "doc_id": "d3", "relevance": 1},
        {"query_id": "q3", "doc_id": "d2", "relevance": 1},
    ]


def test_convert_keeps_only_queries_judged_in_the_split(archive):
    _, queries, labels = beir.convert_archive(archive, "test")
    assert [q["query_id"] for q in queries] == ["q1", "q3"]  # q2 is train-only
    assert {label["query_id"] for label in labels} == {"q1", "q3"}


def test_convert_keeps_corpus_order_and_joins_titles(archive):
    documents, _, _ = beir.convert_archive(archive, "test")
    assert documents == [
        {"doc_id": "d1", "text": "Aspirin reduces fever."},
        {"doc_id": "d2", "text": "untitled abstract"},
        {"doc_id": "d3", "text": "Statins lower cholesterol."},
    ]


def test_missing_split_is_reported(archive):
    with pytest.raises(ValueError, match=r"qrels/dev\.tsv"):
        beir.convert_archive(archive, "dev")


def test_written_dataset_loads_through_the_harness(archive, tmp_path):
    out = tmp_path / "out"
    documents, queries, labels = beir.convert_archive(archive, "test")
    beir.write_dataset(out, documents, queries, labels, {"dataset": "mini"})
    dataset = data.load_dataset(
        str(out / "documents.jsonl"),
        str(out / "queries.jsonl"),
        str(out / "labels.jsonl"),
    )
    assert dataset.relevant_doc_ids("q3") == {"d2", "d3"}
    assert json.loads((out / "manifest.json").read_text()) == {"dataset": "mini"}


def test_writing_twice_produces_identical_files(archive, tmp_path):
    converted = beir.convert_archive(archive, "test")
    for out in (tmp_path / "a", tmp_path / "b"):
        beir.write_dataset(out, *converted, {"dataset": "mini"})
    for name in ("documents.jsonl", "queries.jsonl", "labels.jsonl"):
        assert (tmp_path / "a" / name).read_bytes() == (
            tmp_path / "b" / name
        ).read_bytes()


def test_checksum_mismatch_is_rejected(tmp_path):
    fake = tmp_path / "scifact.zip"
    fake.write_bytes(b"not the real archive")
    with pytest.raises(ValueError, match="does not match"):
        beir.verify_archive(fake, "scifact")


def test_datasets_without_a_published_checksum_are_accepted(tmp_path):
    fake = tmp_path / "other.zip"
    fake.write_bytes(b"anything")
    beir.verify_archive(fake, "other")  # no exception
