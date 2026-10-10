"""Configuration loading.

Parses a YAML config file into typed dataclasses so the rest of the code can
rely on attribute access instead of dictionary lookups, and fails loudly when
a required section or field is missing.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    documents_path: str
    queries_path: str
    labels_path: str


@dataclass
class RetrievalConfig:
    top_k: int = 5
    similarity: str = "cosine"
    candidate_depth: int = 50
    """How deep to rank before cutting to `top_k`.

    Metrics are computed at `top_k`, but failure analysis needs the deeper list
    to tell a relevant document ranked just below the cut from one that was
    never retrieved at all. Values below `top_k` are raised to it.
    """


@dataclass
class EmbeddingConfig:
    model_name: str


@dataclass
class OutputConfig:
    run_dir: str


@dataclass
class Config:
    dataset: DatasetConfig
    retrieval: RetrievalConfig
    embedding: EmbeddingConfig
    output: OutputConfig


def _collect_section(
    cls: type, raw: Any, section: str
) -> tuple[dict[str, Any], list[str]]:
    """Validate one config section against a dataclass.

    Returns the section's kwargs and a list of problems: a missing section,
    a section that is not a mapping, unknown keys, or absent required fields.
    """
    if raw is None:
        return {}, [f"missing section: {section!r}"]
    if not isinstance(raw, dict):
        return {}, [f"{section}: expected a mapping, got {type(raw).__name__}"]

    problems: list[str] = []
    spec = {f.name: f for f in fields(cls)}

    unknown = sorted(set(raw) - set(spec))
    if unknown:
        problems.append(f"{section}: unknown field(s): {', '.join(unknown)}")

    missing = [
        name
        for name, f in spec.items()
        if name not in raw and f.default is MISSING and f.default_factory is MISSING
    ]
    if missing:
        problems.append(f"{section}: missing required field(s): {', '.join(missing)}")

    return {name: raw[name] for name in spec if name in raw}, problems


def load_config(path: str) -> Config:
    """Load a YAML file (see `configs/example.yaml`) into a `Config`.

    Every problem across every section is collected and reported in one
    `ValueError`, so a malformed config is fixed in a single pass rather than
    one error per run.

    Raises `ValueError` if the file is not a YAML mapping, a section is absent
    or malformed, a required field is missing, or an unknown field is present.
    """
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: expected a YAML mapping at the top level, "
            f"got {type(raw).__name__}"
        )

    sections = {
        "dataset": (DatasetConfig, raw.get("dataset")),
        "retrieval": (RetrievalConfig, raw.get("retrieval")),
        "embedding": (EmbeddingConfig, raw.get("embedding")),
        "output": (OutputConfig, raw.get("output")),
    }

    problems: list[str] = []
    kwargs: dict[str, dict[str, Any]] = {}
    for name, (cls, value) in sections.items():
        kwargs[name], section_problems = _collect_section(cls, value, name)
        problems.extend(section_problems)

    if problems:
        raise ValueError(
            f"{path}: invalid config ({len(problems)} problem(s)):\n  "
            + "\n  ".join(problems)
        )

    return Config(
        dataset=DatasetConfig(**kwargs["dataset"]),
        retrieval=RetrievalConfig(**kwargs["retrieval"]),
        embedding=EmbeddingConfig(**kwargs["embedding"]),
        output=OutputConfig(**kwargs["output"]),
    )
