"""Tests for failure classification and failure-case collection.

Covers the four query outcome categories, and the failure records built from
them: which queries are reported, the schema of each record, and the ordering
guarantees the JSONL output depends on.
"""

from retrieval_eval import failure_analysis as fa
from retrieval_eval.data import Dataset, Document, Query, RelevanceLabel

RECORD_KEYS = {"query_id", "query_text", "category", "retrieved", "missed_doc_ids"}


def test_no_labels():
    assert fa.classify_query(["d1", "d2"], set(), k=2) == "no_labels"


def test_ok_when_relevant_in_top_k():
    assert fa.classify_query(["d1", "d2", "d3"], {"d2"}, k=2) == "ok"


def test_relevant_ranked_low():
    assert fa.classify_query(["d1", "d2", "d3"], {"d3"}, k=2) == "relevant_ranked_low"


def test_no_relevant_retrieved():
    assert fa.classify_query(["d1", "d2"], {"zzz"}, k=2) == "no_relevant_retrieved"


def _dataset() -> Dataset:
    """Four queries, one per outcome category.

    q1 ok, q2 relevant ranked low, q3 nothing relevant retrieved, q4 unlabelled.
    """
    return Dataset(
        documents=[Document(f"d{i}", f"text {i}") for i in range(1, 6)],
        queries=[
            Query("q1", "found in top-k"),
            Query("q2", "found below k"),
            Query("q3", "not retrieved at all"),
            Query("q4", "no labels"),
        ],
        labels=[
            RelevanceLabel("q1", "d1", 1),
            RelevanceLabel("q2", "d3", 1),
            RelevanceLabel("q2", "d5", 1),
            RelevanceLabel("q3", "d4", 1),
        ],
    )


RETRIEVED = {
    "q1": ["d1", "d2", "d3"],
    "q2": ["d1", "d2", "d3"],
    "q3": ["d1", "d2", "d3"],
    "q4": ["d1", "d2", "d3"],
}


def test_find_failure_cases_reports_every_non_ok_query():
    cases = fa.find_failure_cases(_dataset(), RETRIEVED, k=2)
    assert [case["query_id"] for case in cases] == ["q2", "q3", "q4"]


def test_find_failure_cases_assigns_the_right_category():
    cases = {
        c["query_id"]: c for c in fa.find_failure_cases(_dataset(), RETRIEVED, k=2)
    }
    assert cases["q2"]["category"] == fa.Category.RELEVANT_RANKED_LOW
    assert cases["q3"]["category"] == fa.Category.NO_RELEVANT_RETRIEVED
    assert cases["q4"]["category"] == fa.Category.NO_LABELS


def test_find_failure_cases_uses_a_fixed_record_schema():
    for case in fa.find_failure_cases(_dataset(), RETRIEVED, k=2):
        assert set(case) == RECORD_KEYS


def test_find_failure_cases_reports_missed_docs_sorted():
    cases = {
        c["query_id"]: c for c in fa.find_failure_cases(_dataset(), RETRIEVED, k=2)
    }
    # q2's relevant docs are d3 (below k) and d5 (absent); both count as missed.
    assert cases["q2"]["missed_doc_ids"] == ["d3", "d5"]
    # Sets iterate in arbitrary order, so the output must be sorted explicitly.
    assert cases["q3"]["missed_doc_ids"] == ["d4"]
    assert cases["q4"]["missed_doc_ids"] == []


def test_find_failure_cases_truncates_retrieved_to_k():
    cases = fa.find_failure_cases(_dataset(), RETRIEVED, k=2)
    assert all(case["retrieved"] == ["d1", "d2"] for case in cases)


def test_find_failure_cases_carries_query_text():
    cases = {
        c["query_id"]: c for c in fa.find_failure_cases(_dataset(), RETRIEVED, k=2)
    }
    assert cases["q2"]["query_text"] == "found below k"
