"""Evaluation orchestration.

Ties the pipeline together: embed documents and queries, retrieve top-k,
compute per-query metrics, and aggregate them. IO stays at the edges so a full
run can be driven with an injected embedder and no network access.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import data, failure_analysis, metrics, retrieval
from .config import Config
from .embeddings import Embedder, get_embedder

METRIC_KEYS = ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k")


@dataclass
class RunResult:
    metrics: dict[str, float]
    per_query: list[dict] = field(default_factory=list)
    config: Config | None = None
    failure_cases: list[dict] = field(default_factory=list)


def run_evaluation(config: Config, embedder: Embedder | None = None) -> RunResult:
    """Run a full evaluation and return a `RunResult`.

    Documents are ranked to `retrieval.candidate_depth` but scored at
    `retrieval.top_k`. The deeper list is what lets failure analysis separate a
    relevant document ranked just below the cut from one never retrieved.

    Every metric is cut at `top_k`, including reciprocal rank: a relevant
    document first appearing below the cut contributes 0, not 1/rank, so the
    reported MRR is MRR@k.

    Aggregate metrics average only over queries that have at least one relevant
    document; unjudged queries would otherwise drag every mean toward zero.
    They are excluded from `num_queries_evaluated` but counted in
    `num_queries`, and still appear as failure cases.

    `embedder` is injectable so the pipeline can be exercised without loading a
    model or touching the network.

    Raises `ValueError` for a similarity other than cosine, before any data is
    loaded or embedded.
    """
    if config.retrieval.similarity != "cosine":
        raise ValueError(
            f"unsupported similarity {config.retrieval.similarity!r}; "
            "only 'cosine' is implemented"
        )

    dataset = data.load_dataset(
        config.dataset.documents_path,
        config.dataset.queries_path,
        config.dataset.labels_path,
    )

    if embedder is None:
        embedder = get_embedder(config.embedding.model_name)

    k = config.retrieval.top_k
    depth = max(k, config.retrieval.candidate_depth)

    doc_ids = [document.doc_id for document in dataset.documents]
    doc_emb = embedder.encode([document.text for document in dataset.documents])
    query_emb = embedder.encode([query.text for query in dataset.queries])

    ranked = retrieval.cosine_top_k(query_emb, doc_emb, doc_ids, k=depth)
    retrieved_by_query = {
        query.query_id: candidates
        for query, candidates in zip(dataset.queries, ranked, strict=True)
    }

    per_query: list[dict] = []
    for query in dataset.queries:
        candidates = retrieved_by_query[query.query_id]
        relevant = dataset.relevant_doc_ids(query.query_id)
        if not relevant:
            continue

        gains = dataset.gains(query.query_id)
        per_query.append(
            {
                "query_id": query.query_id,
                "query_text": query.text,
                "retrieved": candidates[:k],
                "precision_at_k": metrics.precision_at_k(candidates, relevant, k),
                "recall_at_k": metrics.recall_at_k(candidates, relevant, k),
                "reciprocal_rank": metrics.reciprocal_rank(candidates[:k], relevant),
                "ndcg_at_k": metrics.ndcg_at_k(candidates, gains, k),
            }
        )

    aggregate = aggregate_metrics(per_query, k, num_queries=len(dataset.queries))
    failure_cases = failure_analysis.find_failure_cases(
        dataset, retrieved_by_query, k=k
    )
    return RunResult(
        metrics=aggregate,
        per_query=per_query,
        config=config,
        failure_cases=failure_cases,
    )


def aggregate_metrics(
    per_query: list[dict], k: int, num_queries: int
) -> dict[str, float]:
    """Average the per-query rows into the aggregate metrics block.

    `num_queries` is the total in the dataset, labelled or not, so a reader can
    see how many queries the averages leave out.

    Returns zeros for every metric when no query could be evaluated, so the
    shape of the metrics block never depends on the data.
    """
    count = len(per_query)
    aggregate: dict[str, float] = {
        "k": k,
        "num_queries": num_queries,
        "num_queries_evaluated": count,
    }

    if count == 0:
        return aggregate | dict.fromkeys(METRIC_KEYS, 0.0)

    aggregate["precision_at_k"] = sum(r["precision_at_k"] for r in per_query) / count
    aggregate["recall_at_k"] = sum(r["recall_at_k"] for r in per_query) / count
    aggregate["mrr"] = metrics.mean_reciprocal_rank(
        [r["reciprocal_rank"] for r in per_query]
    )
    aggregate["ndcg_at_k"] = sum(r["ndcg_at_k"] for r in per_query) / count
    return aggregate
