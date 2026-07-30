"""Tests for top-k retrieval.

Uses hand-built vectors so similarity ordering is checkable by inspection,
including zero-norm rows and corpora smaller than k.
"""

import numpy as np
import pytest

from retrieval_eval import retrieval

DOC_IDS = ["d1", "d2", "d3"]
DOC_EMB = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def test_cosine_top_k_orders_by_similarity():
    queries = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = retrieval.cosine_top_k(queries, DOC_EMB, DOC_IDS, k=2)
    assert out[0] == ["d1", "d3"]  # [1,0] closest to d1, then d3 (45 deg)
    assert out[1] == ["d2", "d3"]


def test_cosine_top_k_respects_k_and_corpus_size():
    queries = np.array([[1.0, 0.0]])
    out = retrieval.cosine_top_k(queries, DOC_EMB, DOC_IDS, k=10)
    assert len(out[0]) == 3  # never more than the corpus size
    assert out[0][0] == "d1"


def test_normalize_unit_rows():
    m = retrieval.normalize(np.array([[3.0, 4.0]]))
    assert m[0] == pytest.approx([0.6, 0.8])


def test_normalize_handles_zero_row():
    m = retrieval.normalize(np.array([[0.0, 0.0]]))
    assert np.all(np.isfinite(m))  # no NaNs from divide-by-zero
