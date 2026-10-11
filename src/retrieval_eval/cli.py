"""Command-line interface.

Exposes the harness as `retrieval-eval run --config <path>`: load a config,
run an evaluation, and write metrics and a markdown report into the
configured run directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .evaluation import RunResult, run_evaluation
from .reporting import write_report


def _write_outputs(result: RunResult, run_dir: Path) -> None:
    """Write metrics, per-query rows, failure cases, and the report."""
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(run_dir / "retrieval_results.jsonl", result.per_query)
    _write_jsonl(run_dir / "failure_cases.jsonl", result.failure_cases)

    if result.config is not None:
        write_report(result, result.config, str(run_dir / "report.md"))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write one compact JSON object per line."""
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = run_evaluation(config)

    run_dir = Path(config.output.run_dir)
    _write_outputs(result, run_dir)

    metrics = result.metrics
    print(f"Evaluated {metrics['num_queries_evaluated']} queries at k={metrics['k']}")
    for name in ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k"):
        print(f"  {name:<16} {metrics[name]:.4f}")
    print(f"  {'failure cases':<16} {len(result.failure_cases)}")
    print(f"Wrote {run_dir}/metrics.json and {run_dir}/report.md")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, dispatch, and return a process exit code.

    Configuration and dataset problems surface as `ValueError` from the layers
    below; they are caught here and reported on stderr rather than as a
    traceback, since a malformed input file is a user error, not a crash.
    """
    parser = argparse.ArgumentParser(
        prog="retrieval-eval",
        description="Evaluate an embedding-based retrieval configuration.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run an evaluation")
    run_parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="path to a YAML config (see configs/example.yaml)",
    )
    run_parser.set_defaults(handler=_run)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(status=2, message=f"{parser.prog}: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
