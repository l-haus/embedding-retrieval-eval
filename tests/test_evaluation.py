"""End-to-end test of the evaluation pipeline.

Exercises data loading, retrieval, metrics, and aggregation together. A dummy
embedder stands in for the real model so the test runs offline and fast.
"""

import numpy as np

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


def _config(tmp_path):
    docs = tmp_path / "docs.jsonl"
    docs.write_text(
        '{"doc_id": "d1", "text": "cat"}\n{"doc_id": "d2", "text": "dog"}\n',
        encoding="utf-8",
    )
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"query_id": "q1", "text": "cat"}\n', encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        '{"query_id": "q1", "doc_id": "d1", "relevance": 1}\n', encoding="utf-8"
    )
    return Config(
        dataset=DatasetConfig(str(docs), str(queries), str(labels)),
        retrieval=RetrievalConfig(top_k=1, similarity="cosine"),
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
