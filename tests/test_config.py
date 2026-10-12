"""Tests for configuration loading.

Covers a valid config, defaults for optional fields, and each way a config can
be malformed: a missing section, an unknown field, a missing required field, a
section that is not a mapping, and a file that is not a mapping at all. Every
problem should be reported together, in a single error.
"""

import copy
from pathlib import Path

import pytest
import yaml

from retrieval_eval.config import (
    Config,
    DatasetConfig,
    RetrievalConfig,
    _collect_section,
    load_config,
)

EXAMPLE_CONFIG = Path(__file__).parent.parent / "configs" / "example.yaml"

VALID = {
    "dataset": {
        "documents_path": "docs.jsonl",
        "queries_path": "queries.jsonl",
        "labels_path": "labels.jsonl",
    },
    "retrieval": {"top_k": 3, "similarity": "cosine", "candidate_depth": 20},
    "embedding": {"model_name": "some-model"},
    "output": {"run_dir": "runs/test"},
}


def _write(tmp_path, content) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(
        content if isinstance(content, str) else yaml.safe_dump(content),
        encoding="utf-8",
    )
    return str(path)


def _error_for(tmp_path, content) -> str:
    with pytest.raises(ValueError) as excinfo:
        load_config(_write(tmp_path, content))
    return str(excinfo.value)


# load_config: valid input


def test_load_config_builds_typed_sections(tmp_path):
    config = load_config(_write(tmp_path, VALID))
    assert isinstance(config, Config)
    assert config.dataset.documents_path == "docs.jsonl"
    assert config.retrieval.top_k == 3
    assert config.retrieval.candidate_depth == 20
    assert config.embedding.model_name == "some-model"
    assert config.output.run_dir == "runs/test"


def test_load_config_applies_defaults_for_optional_fields(tmp_path):
    raw = copy.deepcopy(VALID)
    raw["retrieval"] = {}
    config = load_config(_write(tmp_path, raw))
    assert config.retrieval == RetrievalConfig()


def test_revision_is_optional_and_unpinned_by_default(tmp_path):
    config = load_config(_write(tmp_path, VALID))
    assert config.embedding.revision is None


def test_revision_is_loaded_when_given(tmp_path):
    raw = copy.deepcopy(VALID)
    raw["embedding"]["revision"] = "abc123"
    assert load_config(_write(tmp_path, raw)).embedding.revision == "abc123"


def test_example_config_in_repo_loads():
    config = load_config(str(EXAMPLE_CONFIG))
    assert config.retrieval.top_k > 0


@pytest.mark.parametrize("name", ["example.yaml", "scifact.yaml"])
def test_shipped_configs_pin_a_full_commit_hash(name):
    revision = load_config(str(EXAMPLE_CONFIG.parent / name)).embedding.revision
    assert revision is not None
    assert len(revision) == 40
    assert all(c in "0123456789abcdef" for c in revision)


# load_config: malformed input


def test_missing_section_is_reported(tmp_path):
    raw = copy.deepcopy(VALID)
    del raw["output"]
    assert "missing section: 'output'" in _error_for(tmp_path, raw)


def test_unknown_field_is_reported(tmp_path):
    raw = copy.deepcopy(VALID)
    raw["retrieval"]["topk"] = 5  # typo for top_k
    assert "retrieval: unknown field(s): topk" in _error_for(tmp_path, raw)


def test_missing_required_field_is_reported(tmp_path):
    raw = copy.deepcopy(VALID)
    del raw["dataset"]["labels_path"]
    message = _error_for(tmp_path, raw)
    assert "dataset: missing required field(s): labels_path" in message


def test_section_that_is_not_a_mapping_is_reported(tmp_path):
    raw = copy.deepcopy(VALID)
    raw["embedding"] = "some-model"
    message = _error_for(tmp_path, raw)
    assert "embedding: expected a mapping, got str" in message


def test_every_problem_is_reported_in_one_error(tmp_path):
    raw = copy.deepcopy(VALID)
    del raw["output"]
    raw["retrieval"]["topk"] = 5
    del raw["dataset"]["labels_path"]
    message = _error_for(tmp_path, raw)
    assert "3 problem(s)" in message
    assert "output" in message
    assert "topk" in message
    assert "labels_path" in message


def test_top_level_list_is_rejected(tmp_path):
    message = _error_for(tmp_path, "- not\n- a mapping\n")
    assert "expected a YAML mapping at the top level, got list" in message


def test_empty_file_is_rejected(tmp_path):
    message = _error_for(tmp_path, "")
    assert "expected a YAML mapping at the top level, got NoneType" in message


# _collect_section: validates one section and returns (kwargs, problems)


def test_collect_section_returns_kwargs_and_no_problems_when_valid():
    kwargs, problems = _collect_section(DatasetConfig, VALID["dataset"], "dataset")
    assert kwargs == VALID["dataset"]
    assert problems == []


def test_collect_section_reports_missing_section():
    kwargs, problems = _collect_section(DatasetConfig, None, "dataset")
    assert kwargs == {}
    assert problems == ["missing section: 'dataset'"]


def test_collect_section_omits_defaulted_fields_left_out():
    kwargs, problems = _collect_section(RetrievalConfig, {"top_k": 3}, "retrieval")
    assert kwargs == {"top_k": 3}
    assert problems == []
