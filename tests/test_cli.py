"""Tests for the command-line interface.

Runs `retrieval-eval run` end to end with a dummy embedder swapped in for the
real model, and checks that user errors exit cleanly with a readable message
instead of a traceback.
"""

import json

import numpy as np
import pytest
import yaml

from retrieval_eval import cli, evaluation
from retrieval_eval.embeddings import Embedder

OUTPUT_FILES = (
    "metrics.json",
    "retrieval_results.jsonl",
    "failure_cases.jsonl",
    "report.md",
)


class DummyEmbedder(Embedder):
    def encode(self, texts):
        table = {"cat": [1.0, 0.0], "dog": [0.0, 1.0]}
        return np.array([table.get(t, [0.5, 0.5]) for t in texts])


@pytest.fixture
def no_model(monkeypatch):
    """Replace the real embedding backend so no model is downloaded."""
    monkeypatch.setattr(evaluation, "get_embedder", lambda name: DummyEmbedder())


def _write_config(tmp_path, **overrides) -> str:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "docs.jsonl").write_text(
        '{"doc_id": "d1", "text": "cat"}\n{"doc_id": "d2", "text": "dog"}\n',
        encoding="utf-8",
    )
    (data_dir / "queries.jsonl").write_text(
        '{"query_id": "q1", "text": "cat"}\n', encoding="utf-8"
    )
    (data_dir / "labels.jsonl").write_text(
        '{"query_id": "q1", "doc_id": "d1", "relevance": 1}\n', encoding="utf-8"
    )
    config = {
        "dataset": {
            "documents_path": str(data_dir / "docs.jsonl"),
            "queries_path": str(data_dir / "queries.jsonl"),
            "labels_path": str(data_dir / "labels.jsonl"),
        },
        "retrieval": {"top_k": 1},
        "embedding": {"model_name": "dummy"},
        "output": {"run_dir": str(tmp_path / "run")},
    }
    config.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(path)


def test_run_writes_every_output_file(tmp_path, no_model):
    assert cli.main(["run", "--config", _write_config(tmp_path)]) == 0
    for name in OUTPUT_FILES:
        assert (tmp_path / "run" / name).is_file(), name


def test_metrics_json_holds_the_aggregate_metrics(tmp_path, no_model):
    cli.main(["run", "--config", _write_config(tmp_path)])
    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert metrics["num_queries"] == 1
    assert metrics["recall_at_k"] == 1.0


def test_run_prints_a_summary(tmp_path, no_model, capsys):
    cli.main(["run", "--config", _write_config(tmp_path)])
    out = capsys.readouterr().out
    assert "Evaluated 1 queries at k=1" in out
    assert "recall_at_k" in out


def test_invalid_config_exits_with_code_2_and_no_traceback(tmp_path, capsys):
    path = _write_config(tmp_path, embedding="not-a-mapping")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["run", "--config", path])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "invalid config" in err
    assert "Traceback" not in err


def test_missing_config_file_exits_with_code_2(tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["run", "--config", str(tmp_path / "absent.yaml")])
    assert excinfo.value.code == 2
    assert "absent.yaml" in capsys.readouterr().err


def test_subcommand_is_required():
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code == 2
