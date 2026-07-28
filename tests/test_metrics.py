"""Tests for retrieval metrics.

Each metric is checked against small hand-computed examples, including the
edge cases the definitions gloss over: fewer than k results retrieved,
queries with no relevant documents, and relevance sets larger than k.
"""

import pytest

from retrieval_eval import metrics

RETRIEVED = ["d1", "d2", "d3", "d4", "d5"]
RELEVANT = {"d2", "d4"}


def test_precision_at_k_divides_by_k():
    assert metrics.precision_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(1 / 3)
    assert metrics.precision_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(2 / 5)


def test_precision_at_k_when_fewer_than_k_retrieved():
    # Only 2 retrieved, 1 relevant, but we still divide by k=5.
    assert metrics.precision_at_k(["d1", "d2"], {"d2"}, 5) == pytest.approx(1 / 5)


def test_recall_at_k():
    assert metrics.recall_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(1 / 2)
    assert metrics.recall_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(1.0)


def test_recall_with_empty_relevant_is_zero():
    assert metrics.recall_at_k(RETRIEVED, set(), 3) == 0.0


def test_reciprocal_rank():
    assert metrics.reciprocal_rank(RETRIEVED, {"d2"}) == pytest.approx(1 / 2)
    assert metrics.reciprocal_rank(RETRIEVED, {"d1"}) == pytest.approx(1.0)


def test_reciprocal_rank_no_hit_is_zero():
    assert metrics.reciprocal_rank(RETRIEVED, {"zzz"}) == 0.0


def test_mean_reciprocal_rank():
    assert metrics.mean_reciprocal_rank([1.0, 0.5, 0.0]) == pytest.approx(0.5)
    assert metrics.mean_reciprocal_rank([]) == 0.0


def test_ndcg_at_k_value():
    # See docstring example: DCG=1/log2(3), IDCG=1+1/log2(3).
    got = metrics.ndcg_at_k(["d1", "d2", "d3"], {"d2": 1, "d4": 1}, 3)
    assert got == pytest.approx(0.38685, abs=1e-4)


def test_ndcg_perfect_ordering_is_one():
    got = metrics.ndcg_at_k(["d2", "d4", "d1"], {"d2": 1, "d4": 1}, 3)
    assert got == pytest.approx(1.0)


def test_ndcg_zero_when_no_gain():
    assert metrics.ndcg_at_k(["d1", "d3"], {"d2": 1}, 2) == 0.0


def test_ndcg_idcg_truncates_to_k():
    # Perfect top-3 should score 1.0 even when more relevant docs exist beyond k.
    got = metrics.ndcg_at_k(
        ["d1", "d2", "d3"],
        {"d1": 1, "d2": 1, "d3": 1, "d4": 1, "d5": 1},
        3,
    )
    assert got == pytest.approx(1.0)
