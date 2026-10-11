"""End-to-end test of the evaluation pipeline.

Exercises data loading, retrieval, metrics, and aggregation together. A dummy
embedder stands in for the real model so the test runs offline and fast.
"""

import numpy as np
import pytest

from retrieval_eval import evaluation
from retrieval_eval.config import (
    Config,
    DatasetConfig,
    EmbeddingConfig,
    OutputConfig,
    RetrievalConfig,
)
from retrieval_eval.embeddings import Embedder


class DummyEmbedder(Embedder):
    """Maps known texts to one-hot vectors; everything else to a neutral vector."""

    def encode(self, texts):
        table = {"cat": [1.0, 0.0], "dog": [0.0, 1.0]}
        return np.array([table.get(t.strip().lower(), [0.5, 0.5]) for t in texts])


class ExplodingEmbedder(Embedder):
    """Fails if called, to prove a run stopped before any embedding happened."""

    def encode(self, texts):
        raise AssertionError("encode should not have been called")


def _config(
    tmp_path,
    queries='{"query_id": "q1", "text": "cat"}\n',
    labels='{"query_id": "q1", "doc_id": "d1", "relevance": 1}\n',
    top_k=1,
    similarity="cosine",
):
    docs_path = tmp_path / "docs.jsonl"
    docs_path.write_text(
        '{"doc_id": "d1", "text": "cat"}\n{"doc_id": "d2", "text": "dog"}\n',
        encoding="utf-8",
    )
    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_text(queries, encoding="utf-8")
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(labels, encoding="utf-8")
    return Config(
        dataset=DatasetConfig(str(docs_path), str(queries_path), str(labels_path)),
        retrieval=RetrievalConfig(top_k=top_k, similarity=similarity),
        embedding=EmbeddingConfig(model_name="dummy"),
        output=OutputConfig(run_dir=str(tmp_path / "run")),
    )


def test_run_evaluation_perfect_retrieval(tmp_path):
    result = evaluation.run_evaluation(_config(tmp_path), embedder=DummyEmbedder())
    m = result.metrics
    expected_keys = {
        "k",
        "num_queries_evaluated",
        "precision_at_k",
        "recall_at_k",
        "mrr",
        "ndcg_at_k",
    }
    assert expected_keys.issubset(m.keys())
    assert m["num_queries_evaluated"] == 1
    assert m["recall_at_k"] == 1.0
    assert m["precision_at_k"] == 1.0
    assert m["mrr"] == 1.0


def test_unlabelled_query_counts_in_total_but_not_in_averages(tmp_path):
    config = _config(
        tmp_path,
        queries=(
            '{"query_id": "q1", "text": "cat"}\n{"query_id": "q2", "text": "dog"}\n'
        ),
    )
    m = evaluation.run_evaluation(config, embedder=DummyEmbedder()).metrics
    assert m["num_queries"] == 2
    assert m["num_queries_evaluated"] == 1
    # The unlabelled query must not drag the averages toward zero.
    assert m["recall_at_k"] == 1.0


def test_reciprocal_rank_is_cut_at_k(tmp_path):
    # The relevant doc d2 ranks second for "cat", just below k=1.
    config = _config(
        tmp_path, labels='{"query_id": "q1", "doc_id": "d2", "relevance": 1}\n'
    )
    m = evaluation.run_evaluation(config, embedder=DummyEmbedder()).metrics
    assert m["mrr"] == 0.0  # MRR@k: rank 2 with k=1 contributes 0, not 1/2


def test_unsupported_similarity_is_rejected_before_any_work(tmp_path):
    config = _config(tmp_path, similarity="dot")
    with pytest.raises(ValueError, match="unsupported similarity 'dot'"):
        evaluation.run_evaluation(config, embedder=ExplodingEmbedder())
