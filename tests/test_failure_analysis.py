"""Tests for failure classification.

Checks that each query outcome maps to the right category: no labels to judge
against, a relevant document inside the top-k, one ranked below k, or none
retrieved at all.
"""

from retrieval_eval import failure_analysis as fa


def test_no_labels():
    assert fa.classify_query(["d1", "d2"], set(), k=2) == "no_labels"


def test_ok_when_relevant_in_top_k():
    assert fa.classify_query(["d1", "d2", "d3"], {"d2"}, k=2) == "ok"


def test_relevant_ranked_low():
    assert fa.classify_query(["d1", "d2", "d3"], {"d3"}, k=2) == "relevant_ranked_low"


def test_no_relevant_retrieved():
    assert fa.classify_query(["d1", "d2"], {"zzz"}, k=2) == "no_relevant_retrieved"
