"""Tests for markdown report generation.

Covers the report's sections, the failure summary's counts, deterministic
output, and the switch between one ranked table on small runs and separate
best and worst tables on larger ones.
"""

import json

import numpy as np

from retrieval_eval import evaluation, reporting
from retrieval_eval.config import (
    Config,
    DatasetConfig,
    EmbeddingConfig,
    OutputConfig,
    RetrievalConfig,
)
from retrieval_eval.embeddings import Embedder
from retrieval_eval.evaluation import RunResult

DOC_VECTORS = {
    "alpha": [1.0, 0.0],
    "beta": [0.9, 0.1],
    "gamma": [0.8, 0.2],
    "delta": [0.7, 0.3],
    "epsilon": [0.0, 1.0],
}


class RankingEmbedder(Embedder):
    """Every query ranks d1 > d2 > d3 > d4 > d5; d5 is orthogonal to queries."""

    def encode(self, texts):
        return np.array([DOC_VECTORS.get(t, [1.0, 0.0]) for t in texts])


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _four_outcome_run(tmp_path):
    """q1 ok, q2 ranked below k, q3 never retrieved, q4 unlabelled."""
    docs = tmp_path / "docs.jsonl"
    _write_jsonl(
        docs,
        [{"doc_id": f"d{i}", "text": t} for i, t in enumerate(DOC_VECTORS, 1)],
    )
    queries = tmp_path / "queries.jsonl"
    _write_jsonl(
        queries, [{"query_id": f"q{i}", "text": f"query {i}"} for i in range(1, 5)]
    )
    labels = tmp_path / "labels.jsonl"
    _write_jsonl(
        labels,
        [
            {"query_id": "q1", "doc_id": "d1", "relevance": 1},  # rank 1: ok
            {"query_id": "q2", "doc_id": "d3", "relevance": 1},  # rank 3: below k
            {"query_id": "q3", "doc_id": "d5", "relevance": 1},  # rank 5: past depth
        ],
    )
    config = Config(
        dataset=DatasetConfig(str(docs), str(queries), str(labels)),
        retrieval=RetrievalConfig(top_k=2, candidate_depth=4),
        embedding=EmbeddingConfig(model_name="dummy"),
        output=OutputConfig(run_dir=str(tmp_path / "run")),
    )
    return evaluation.run_evaluation(config, embedder=RankingEmbedder()), config


def test_report_has_every_section(tmp_path):
    report = reporting.render_markdown_report(*_four_outcome_run(tmp_path))
    for heading in (
        "# Retrieval Evaluation Report",
        "## Config",
        "## Inputs",
        "## Metrics",
        "## Best and worst queries",
        "## Failure cases",
        "## Limitations",
    ):
        assert heading in report


def test_failure_summary_separates_misses_from_unlabelled_queries(tmp_path):
    report = reporting.render_markdown_report(*_four_outcome_run(tmp_path))
    assert (
        "2 of 3 labelled queries did not retrieve a relevant document within top-k:"
        in report
    )
    assert "1 of 4 queries had no relevance labels and could not be judged." in report


def test_failure_summary_splits_near_misses_from_misses(tmp_path):
    # top_k 2, candidate_depth 4: q2's document is at rank 3, q3's at rank 5.
    report = reporting.render_markdown_report(*_four_outcome_run(tmp_path))
    assert (
        "- 1 ranked below the top 2 but within the top 4 (`relevant_ranked_low`)"
        in report
    )
    assert "- 1 not within the top 4 at all (`no_relevant_retrieved`)" in report


def test_unpinned_revision_is_flagged(tmp_path):
    report = reporting.render_markdown_report(*_four_outcome_run(tmp_path))
    assert "| model revision | unpinned (latest at run time) |" in report


def test_pinned_revision_is_shown(tmp_path):
    run, config = _four_outcome_run(tmp_path)
    config.embedding.revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    report = reporting.render_markdown_report(run, config)
    assert "| model revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |" in report


def test_truncation_not_measured_is_said_plainly(tmp_path):
    report = reporting.render_markdown_report(*_four_outcome_run(tmp_path))
    assert "| documents | 5 | not measured |" in report


def test_truncation_counts_and_shares_are_shown(tmp_path):
    run, config = _four_outcome_run(tmp_path)
    run.diagnostics |= {"documents_truncated": 2, "queries_truncated": 0}
    report = reporting.render_markdown_report(run, config)
    assert "| documents | 5 | 2 (40.0%) |" in report
    assert "| queries | 4 | 0 (0.0%) |" in report
    assert "scored on its opening only" in report


def test_failure_summary_when_every_labelled_query_succeeds(tmp_path):
    run, config = _four_outcome_run(tmp_path)
    run.failure_cases = [c for c in run.failure_cases if c["query_id"] == "q4"]
    report = reporting.render_markdown_report(run, config)
    assert "Every labelled query retrieved a relevant document" in report
    assert "1 of 4 queries had no relevance labels" in report


def test_report_is_identical_across_renders(tmp_path):
    first = reporting.render_markdown_report(*_four_outcome_run(tmp_path))
    second = reporting.render_markdown_report(*_four_outcome_run(tmp_path))
    assert first == second


def test_small_runs_list_scored_queries_in_one_table(tmp_path):
    report = reporting.render_markdown_report(*_four_outcome_run(tmp_path))
    assert "All 3 scored queries, ranked by ndcg_at_k." in report
    assert "### Best" not in report
    assert "### Worst" not in report


def _row(index: int, ndcg: float) -> dict:
    return {
        "query_id": f"q{index:02d}",
        "query_text": f"query {index}",
        "retrieved": [],
        "precision_at_k": ndcg,
        "recall_at_k": ndcg,
        "reciprocal_rank": ndcg,
        "ndcg_at_k": ndcg,
    }


def test_large_runs_split_best_and_worst(tmp_path):
    _, config = _four_outcome_run(tmp_path)
    rows = [_row(i, i / 12) for i in range(1, 13)]
    run = RunResult(
        metrics=evaluation.aggregate_metrics(rows, k=2, num_queries=12),
        per_query=rows,
        config=config,
    )
    report = reporting.render_markdown_report(run, config)
    best = report.index("### Best")
    worst = report.index("### Worst")
    assert best < report.index("`q12`") < worst  # highest nDCG listed first
    assert worst < report.index("`q01`")  # lowest nDCG under worst


def test_write_report_creates_parent_directories(tmp_path):
    run, config = _four_outcome_run(tmp_path)
    destination = tmp_path / "nested" / "deeper" / "report.md"
    reporting.write_report(run, config, str(destination))
    assert destination.read_text(encoding="utf-8").startswith(
        "# Retrieval Evaluation Report"
    )
