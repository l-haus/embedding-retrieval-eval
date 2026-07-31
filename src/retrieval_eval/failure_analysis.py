"""Failure-case analysis.

Aggregate metrics say whether retrieval worked; they do not say why it did
not. This module sorts each query outcome into coarse, explainable categories
so failures can be inspected individually rather than only counted.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from .data import Dataset


class Category(StrEnum):
    """Outcome categories for a single query.

    Members are strings, so they compare equal to their values and serialize
    directly to JSON.
    """

    NO_LABELS = "no_labels"
    OK = "ok"
    RELEVANT_RANKED_LOW = "relevant_ranked_low"
    NO_RELEVANT_RETRIEVED = "no_relevant_retrieved"


def classify_query(retrieved: Sequence[str], relevant: set[str], k: int) -> Category:
    """Return one category for a query's outcome.

    `NO_LABELS`             -> `relevant` is empty, so the query can't be judged
    `OK`                    -> at least one relevant doc is in the top-k
    `RELEVANT_RANKED_LOW`   -> a relevant doc was retrieved, but below k
    `NO_RELEVANT_RETRIEVED` -> no relevant doc anywhere in `retrieved`

    `retrieved` should be the full ranked candidate list, not a list already
    truncated to k, or `RELEVANT_RANKED_LOW` can never be returned.
    The checks are ordered by precedence: a query with relevant documents both
    inside and below the cut counts as `OK`.
    """
    if not relevant:
        return Category.NO_LABELS
    if any(doc in relevant for doc in retrieved[:k]):
        return Category.OK
    if any(doc in relevant for doc in retrieved):
        return Category.RELEVANT_RANKED_LOW
    return Category.NO_RELEVANT_RETRIEVED


def find_failure_cases(
    dataset: Dataset,
    retrieved_by_query: dict[str, list[str]],
    k: int,
) -> list[dict]:
    """Build one record per query whose outcome was not `OK`.

    Unlabelled queries are reported rather than skipped: "these queries could
    not be judged" is itself a finding worth seeing in the report.

    `retrieved_by_query` holds the full ranked candidate list per query, so
    `classify_query` can tell a relevant document ranked below k from one that
    was never retrieved. Records store only the top-k.

    Each record carries the same keys — query_id, query_text, category,
    retrieved, missed_doc_ids — so the output is a uniform list of plain dicts
    that serializes straight to JSONL. `missed_doc_ids` is sorted, since it
    derives from set arithmetic and would otherwise vary between runs.
    """
    cases: list[dict] = []
    for query in dataset.queries:
        retrieved = retrieved_by_query.get(query.query_id, [])
        relevant = dataset.relevant_doc_ids(query.query_id)
        category = classify_query(retrieved, relevant, k)
        if category is Category.OK:
            continue

        top_k = list(retrieved[:k])
        cases.append(
            {
                "query_id": query.query_id,
                "query_text": query.text,
                "category": category,
                "retrieved": top_k,
                "missed_doc_ids": sorted(relevant - set(top_k)),
            }
        )
    return cases
