"""Dataset loading and validation.

The harness consumes three JSONL files: documents, queries, and relevance
labels. This module loads them into typed records and validates referential
integrity *before* any expensive embedding work happens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


@dataclass(frozen=True)
class RelevanceLabel:
    query_id: str
    doc_id: str
    relevance: int


@dataclass
class Dataset:
    documents: list[Document] = field(default_factory=list)
    queries: list[Query] = field(default_factory=list)
    labels: list[RelevanceLabel] = field(default_factory=list)

    def relevant_doc_ids(self, query_id: str) -> set[str]:
        """Return the set of doc_ids judged relevant (relevance > 0) for a query."""
        return {
            label.doc_id
            for label in self.labels
            if label.query_id == query_id and label.relevance > 0
        }

    def gains(self, query_id: str) -> dict[str, float]:
        """Return {doc_id: relevance} for a query, for graded metrics like nDCG."""
        return {
            label.doc_id: label.relevance
            for label in self.labels
            if label.query_id == query_id
        }


def load_jsonl(path: str) -> list[dict]:
    """Read a JSONL file and return one dict per non-empty line.

    Raises `ValueError` with the offending line number if a line is not valid
    JSON or does not decode to an object.
    """
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{lineno}: expected a JSON object, "
                    f"got {type(record).__name__}"
                )
            records.append(record)
    return records


def _require(record: dict, fields: tuple[str, ...], path: str, index: int) -> None:
    """Raise `ValueError` naming every required field missing from `record`."""
    missing = [name for name in fields if name not in record]
    if missing:
        raise ValueError(
            f"{path}: record {index} is missing required field(s): {', '.join(missing)}"
        )


def load_documents(path: str) -> list[Document]:
    """Load documents; require fields `doc_id`, `text`.

    Raises `ValueError` if a required field is missing or if a `doc_id`
    repeats, since duplicate ids make retrieval results ambiguous.
    """
    documents: list[Document] = []
    seen: set[str] = set()
    for index, record in enumerate(load_jsonl(path), start=1):
        _require(record, ("doc_id", "text"), path, index)
        doc_id = record["doc_id"]
        if doc_id in seen:
            raise ValueError(f"{path}: record {index} repeats doc_id {doc_id!r}")
        seen.add(doc_id)
        documents.append(Document(doc_id=doc_id, text=record["text"]))
    return documents


def load_queries(path: str) -> list[Query]:
    """Load queries; require fields `query_id`, `text`, and reject duplicate ids."""
    queries: list[Query] = []
    seen: set[str] = set()
    for index, record in enumerate(load_jsonl(path), start=1):
        _require(record, ("query_id", "text"), path, index)
        query_id = record["query_id"]
        if query_id in seen:
            raise ValueError(f"{path}: record {index} repeats query_id {query_id!r}")
        seen.add(query_id)
        queries.append(Query(query_id=query_id, text=record["text"]))
    return queries


def load_labels(path: str) -> list[RelevanceLabel]:
    """Load relevance labels; require `query_id`, `doc_id`, `relevance` (int).

    Repeated (query_id, doc_id) pairs are not rejected here; referential
    integrity against the loaded documents and queries is checked separately
    by `validate_dataset`.
    """
    labels: list[RelevanceLabel] = []
    for index, record in enumerate(load_jsonl(path), start=1):
        _require(record, ("query_id", "doc_id", "relevance"), path, index)
        relevance = record["relevance"]
        if isinstance(relevance, bool) or not isinstance(relevance, int):
            raise ValueError(
                f"{path}: record {index} has non-integer relevance {relevance!r}"
            )
        labels.append(
            RelevanceLabel(
                query_id=record["query_id"],
                doc_id=record["doc_id"],
                relevance=relevance,
            )
        )
    return labels


def load_dataset(documents_path: str, queries_path: str, labels_path: str) -> Dataset:
    """Load all three files and return a validated `Dataset`.

    Raises `ValueError` if any file is malformed or if the loaded records fail
    referential integrity checks
    """
    documents = load_documents(documents_path)
    queries = load_queries(queries_path)
    labels = load_labels(labels_path)
    dataset = Dataset(documents=documents, queries=queries, labels=labels)
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: Dataset) -> None:
    """Check referential integrity across the loaded records.

    Collects every violation and raises a single `ValueError` listing all of
    them, so one pass over a bad file surfaces all its problems instead of
    only the first.

    Raises `ValueError` if a label references an unknown query_id or doc_id,
    carries a negative relevance, or repeats a (query_id, doc_id) pair.
    """
    query_ids = {query.query_id for query in dataset.queries}
    doc_ids = {doc.doc_id for doc in dataset.documents}

    problems: list[str] = []
    seen: set[tuple[str, str]] = set()

    for index, label in enumerate(dataset.labels, start=1):
        if label.query_id not in query_ids:
            problems.append(f"label {index}: unknown query_id {label.query_id!r}")
        if label.doc_id not in doc_ids:
            problems.append(f"label {index}: unknown doc_id {label.doc_id!r}")
        if label.relevance < 0:
            problems.append(f"label {index}: negative relevance {label.relevance!r}")

        pair = (label.query_id, label.doc_id)
        if pair in seen:
            problems.append(
                f"label {index}: duplicate pair "
                f"query_id={label.query_id!r}, doc_id={label.doc_id!r}"
            )
        seen.add(pair)

    if problems:
        raise ValueError(
            f"dataset failed validation ({len(problems)} problem(s)):\n "
            + "\n ".join(problems)
        )
