"""Retrieval metrics.

These functions define the *contract* the tests pin down. Read the conventions
in each docstring carefully — they resolve the usual ambiguities (what to divide
by, how to handle empty inputs).

Implement: `precision_at_k`, `recall_at_k`, `reciprocal_rank`,
`mean_reciprocal_rank`, `ndcg_at_k`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import log2


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant.

    Convention: divide by `k` (NOT by the number retrieved). So if fewer than k
    items are retrieved, the "missing" slots count against precision.

    Example: retrieved=[d1,d2,d3,d4,d5], relevant={d2,d4}, k=3 -> 1/3.
    """
    result = len(set(retrieved[:k]) & relevant) / k
    return result


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of all relevant docs that appear in the top-k.

    Convention: divide by len(relevant). If `relevant` is empty, return 0.0
    (such queries should usually be excluded from aggregate recall upstream).

    Example: retrieved=[d1,d2,d3,d4,d5], relevant={d2,d4}, k=3 -> 1/2.
    """
    if len(relevant) == 0:
        return 0.0

    result = len(set(retrieved[:k]) & relevant) / len(relevant)

    return result


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1 / (rank of the first relevant item), ranks starting at 1.

    Return 0.0 if no relevant item appears in `retrieved`.

    Example: retrieved=[d1,d2,d3], relevant={d2} -> 1/2.
    """
    rank = 0.0
    for r, item in enumerate(retrieved, start=1):
        if item in relevant:
            rank = r
            break
    else:
        return rank

    return 1 / rank


def mean_reciprocal_rank(rr_values: Iterable[float]) -> float:
    """Mean of per-query reciprocal ranks. Return 0.0 for an empty input."""
    if len(rr_values) == 0:
        return 0.0

    result = sum(rr_values) / len(rr_values)
    return result


def ndcg_at_k(retrieved: Sequence[str], gains: Mapping[str, float], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k (graded relevance).

    Definitions (ranks i start at 1):
        DCG@k  = sum over top-k of  gain(item) / log2(i + 1)
        IDCG@k = DCG@k for the ideal ordering (gains sorted descending)
        nDCG@k = DCG@k / IDCG@k, or 0.0 if IDCG@k == 0.

    `gains` maps doc_id -> relevance grade; missing doc_ids have gain 0.

    Example: retrieved=[d1,d2,d3], gains={d2:1,d4:1}, k=3 -> ~0.3869.
    """
    dcg = 0
    for i, doc in enumerate(retrieved[:k], start=1):
        dcg += gains.get(doc, 0) / log2(i + 1)

    idcg = 0
    for i, gain in enumerate(sorted(gains.values(), reverse=True)[:k], start=1):
        idcg += gain / log2(i + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg
