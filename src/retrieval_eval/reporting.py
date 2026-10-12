"""Markdown report generation.

Renders a RunResult into a readable report: the config that produced it,
aggregate metrics, best and worst queries, failure cases, and the limitations
of the evaluation.
"""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .evaluation import RunResult
from .failure_analysis import Category

RANKING_METRIC = "ndcg_at_k"
MAX_LISTED_QUERIES = 5
MAX_LISTED_FAILURES = 20


def _format_number(value: float | int) -> str:
    """Render integers plainly and floats to four decimal places."""
    return str(value) if isinstance(value, int) else f"{value:.4f}"


def _rank_queries(per_query: list[dict]) -> list[dict]:
    """Sort per-query rows best-first, breaking ties by query_id.

    The tie-break keeps the report stable when many queries score identically,
    which is common on small evaluation sets.
    """
    return sorted(
        per_query,
        key=lambda row: (-row[RANKING_METRIC], row["query_id"]),
    )


def _query_table(rows: list[dict]) -> list[str]:
    """Render per-query rows as a markdown table."""
    lines = [
        "| query_id | query | precision@k | recall@k | RR | nDCG@k |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| `{row['query_id']}` | {row['query_text']} "
        f"| {row['precision_at_k']:.4f} | {row['recall_at_k']:.4f} "
        f"| {row['reciprocal_rank']:.4f} | {row['ndcg_at_k']:.4f} |"
        for row in rows
    ]
    return lines


def _revision_cell(revision: str | None) -> str:
    """Show the pinned revision, or say plainly that the run was unpinned."""
    if revision is None:
        return "unpinned (latest at run time)"
    return f"`{revision}`"


def _truncated_cell(truncated: int | None, count: int) -> str:
    if truncated is None:
        return "not measured"
    share = truncated / count if count else 0.0
    return f"{truncated} ({share:.1%})"


def _inputs_section(diagnostics: dict[str, int | None]) -> list[str]:
    """Describe the inputs: how many there were and how many were truncated."""
    lines = ["## Inputs", ""]
    if not diagnostics:
        return [*lines, "Input diagnostics were not recorded for this run.", ""]

    num_documents = diagnostics["num_documents"]
    num_queries = diagnostics["num_queries"]
    lines += [
        "| input | count | truncated |",
        "|---|---|---|",
        f"| documents | {num_documents} "
        f"| {_truncated_cell(diagnostics['documents_truncated'], num_documents)} |",
        f"| queries | {num_queries} "
        f"| {_truncated_cell(diagnostics['queries_truncated'], num_queries)} |",
        "",
    ]
    if diagnostics["documents_truncated"] or diagnostics["queries_truncated"]:
        lines += [
            "Text past the model's input limit is dropped before embedding, so a "
            "truncated document is scored on its opening only.",
            "",
        ]
    return lines


def _failure_summary(run_result: RunResult, config: Config) -> list[str]:
    """Summarize failures, keeping retrieval misses apart from unjudged queries.

    A query with no relevance labels is not a retrieval failure: it could not
    be judged at all. Each count is stated against its own total, labelled
    queries for misses and all queries for unlabelled ones.

    Misses are split by how far off they were: ranked below the cut but within
    the candidate depth, or not within the candidate depth at all. The first
    kind is what a reranker can recover; the second is not.
    """
    categories = [case["category"] for case in run_result.failure_cases]
    unlabelled = categories.count(Category.NO_LABELS)
    ranked_low = categories.count(Category.RELEVANT_RANKED_LOW)
    not_retrieved = categories.count(Category.NO_RELEVANT_RETRIEVED)
    missed = ranked_low + not_retrieved
    labelled = run_result.metrics["num_queries_evaluated"]
    total = run_result.metrics["num_queries"]
    k = config.retrieval.top_k
    depth = max(k, config.retrieval.candidate_depth)

    if missed:
        lines = [
            f"{missed} of {labelled} labelled queries did not retrieve a relevant "
            "document within top-k:",
            "",
            f"- {ranked_low} ranked below the top {k} but within the top {depth} "
            "(`relevant_ranked_low`)",
            f"- {not_retrieved} not within the top {depth} at all "
            "(`no_relevant_retrieved`)",
        ]
    else:
        lines = ["Every labelled query retrieved a relevant document within top-k."]
    if unlabelled:
        # The blank line keeps markdown from folding this into the list above.
        lines += [
            "",
            f"{unlabelled} of {total} queries had no relevance labels and could "
            "not be judged.",
        ]
    return [*lines, ""]


def render_markdown_report(run_result: RunResult, config: Config) -> str:
    """Return the run as a markdown report.

    Contains only values derived from the run, so two runs over identical
    inputs produce identical text.
    """
    lines: list[str] = ["# Retrieval Evaluation Report", ""]

    lines += [
        "## Config",
        "",
        "| setting | value |",
        "|---|---|",
        f"| embedding model | `{config.embedding.model_name}` |",
        f"| model revision | {_revision_cell(config.embedding.revision)} |",
        f"| top_k | {config.retrieval.top_k} |",
        f"| candidate_depth | {config.retrieval.candidate_depth} |",
        f"| similarity | {config.retrieval.similarity} |",
        f"| documents | `{config.dataset.documents_path}` |",
        f"| queries | `{config.dataset.queries_path}` |",
        f"| labels | `{config.dataset.labels_path}` |",
        "",
    ]

    lines += _inputs_section(run_result.diagnostics)

    lines += ["## Metrics", "", "| metric | value |", "|---|---|"]
    lines += [
        f"| {name} | {_format_number(value)} |"
        for name, value in run_result.metrics.items()
    ]
    lines.append("")

    ranked = _rank_queries(run_result.per_query)
    lines += ["## Best and worst queries", ""]
    if not ranked:
        lines += ["No query had relevant labels, so none could be scored.", ""]
    elif len(ranked) <= 2 * MAX_LISTED_QUERIES:
        # Splitting into best and worst would list the same rows twice.
        lines += [f"All {len(ranked)} scored queries, ranked by {RANKING_METRIC}.", ""]
        lines += _query_table(ranked)
        lines.append("")
    else:
        lines += [f"Ranked by {RANKING_METRIC}.", "", "### Best", ""]
        lines += _query_table(ranked[:MAX_LISTED_QUERIES])
        lines += ["", "### Worst", ""]
        lines += _query_table(ranked[-MAX_LISTED_QUERIES:][::-1])
        lines.append("")

    lines += ["## Failure cases", ""]
    lines += _failure_summary(run_result, config)
    if run_result.failure_cases:
        shown = run_result.failure_cases[:MAX_LISTED_FAILURES]
        lines += [
            "| query_id | query | category | retrieved | missed |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| `{case['query_id']}` | {case['query_text']} "
            f"| {case['category']} "
            f"| {', '.join(f'`{d}`' for d in case['retrieved']) or '—'} "
            f"| {', '.join(f'`{d}`' for d in case['missed_doc_ids']) or '—'} |"
            for case in shown
        ]
        if len(shown) < len(run_result.failure_cases):
            lines += [
                "",
                f"Showing {len(shown)} of {len(run_result.failure_cases)}; "
                "the full set is in `failure_cases.jsonl`.",
            ]
        lines.append("")

    lines += [
        "## Limitations",
        "",
        "These numbers describe one configuration over one labelled dataset.",
        "Before drawing conclusions from them, read `docs/limitations.md` —",
        "in particular that unlabelled retrieved documents count as",
        "non-relevant, that no significance testing is performed, and that",
        "text is used exactly as provided with no normalization.",
        "",
    ]

    return "\n".join(lines)


def write_report(run_result: RunResult, config: Config, path: str) -> None:
    """Render the report and write it to `path`, creating parent directories."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_markdown_report(run_result, config), encoding="utf-8")
