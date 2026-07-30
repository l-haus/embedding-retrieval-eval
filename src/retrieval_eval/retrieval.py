"""Top-k retrieval over embedding vectors.

Pure NumPy, so it can be unit tested with hand-built vectors. Cosine
similarity over L2-normalized rows is enough for a correct baseline; an
approximate-nearest-neighbour backend such as FAISS would slot in behind the
same interface.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row.

    Rows with zero norm stay all-zeros rather than producing NaNs.
    """
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    return matrix / norm


def cosine_top_k(
    query_emb: np.ndarray,
    doc_emb: np.ndarray,
    doc_ids: Sequence[str],
    k: int,
) -> list[list[str]]:
    """Rank documents by cosine similarity for each query.

    Args:
        query_emb: shape (num_queries, dim)
        doc_emb:   shape (num_docs, dim)
        doc_ids:   length num_docs, aligned with rows of doc_emb
        k:         number of doc_ids to return per query

    Returns:
        For each query, the list of the top-k doc_ids in descending similarity
        order (length == min(k, num_docs)).
    """
    n_query_emb = normalize(query_emb)
    n_doc_emb = normalize(doc_emb)
    scores = n_query_emb @ n_doc_emb.T
    ranked = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return [[doc_ids[i] for i in row] for row in ranked]
