"""Tests for dataset loading and validation.

Covers required fields, duplicate IDs, and referential integrity between
labels and the documents and queries they reference.
"""

import pytest

from retrieval_eval import data


def _write(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def test_load_documents_ok(tmp_path):
    path = _write(
        tmp_path,
        "docs.jsonl",
        [
            '{"doc_id": "d1", "text": "alpha"}',
            '{"doc_id": "d2", "text": "beta"}',
        ],
    )
    docs = data.load_documents(path)
    assert [d.doc_id for d in docs] == ["d1", "d2"]
    assert docs[0].text == "alpha"


def test_load_documents_rejects_duplicate_ids(tmp_path):
    path = _write(
        tmp_path,
        "docs.jsonl",
        [
            '{"doc_id": "d1", "text": "alpha"}',
            '{"doc_id": "d1", "text": "beta"}',
        ],
    )
    with pytest.raises(ValueError):
        data.load_documents(path)


def test_load_documents_rejects_missing_field(tmp_path):
    path = _write(tmp_path, "docs.jsonl", ['{"doc_id": "d1"}'])
    with pytest.raises(ValueError):
        data.load_documents(path)


def _tiny_dataset(tmp_path, label_lines):
    docs = _write(
        tmp_path,
        "docs.jsonl",
        [
            '{"doc_id": "d1", "text": "alpha"}',
            '{"doc_id": "d2", "text": "beta"}',
        ],
    )
    queries = _write(
        tmp_path,
        "queries.jsonl",
        [
            '{"query_id": "q1", "text": "find alpha"}',
        ],
    )
    labels = _write(tmp_path, "labels.jsonl", label_lines)
    return docs, queries, labels


def test_load_dataset_ok_and_relevant_doc_ids(tmp_path):
    docs, queries, labels = _tiny_dataset(
        tmp_path,
        [
            '{"query_id": "q1", "doc_id": "d1", "relevance": 1}',
        ],
    )
    ds = data.load_dataset(docs, queries, labels)
    assert ds.relevant_doc_ids("q1") == {"d1"}


def test_validate_rejects_label_to_missing_doc(tmp_path):
    docs, queries, labels = _tiny_dataset(
        tmp_path,
        [
            '{"query_id": "q1", "doc_id": "ZZZ", "relevance": 1}',
        ],
    )
    with pytest.raises(ValueError):
        data.load_dataset(docs, queries, labels)


def test_validate_rejects_label_to_missing_query(tmp_path):
    docs, queries, labels = _tiny_dataset(
        tmp_path,
        [
            '{"query_id": "qX", "doc_id": "d1", "relevance": 1}',
        ],
    )
    with pytest.raises(ValueError):
        data.load_dataset(docs, queries, labels)
