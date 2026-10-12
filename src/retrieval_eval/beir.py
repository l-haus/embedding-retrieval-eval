"""Conversion from BEIR benchmark datasets to the harness's JSONL format.

BEIR ships each dataset as a zip holding `corpus.jsonl` (`_id`, `title`,
`text`), `queries.jsonl` (`_id`, `text`), and one `qrels/<split>.tsv` per split.
This module downloads that archive, verifies it against BEIR's published
checksum, and writes `documents.jsonl`, `queries.jsonl`, and `labels.jsonl`
for a single split, plus a manifest recording exactly what was produced.

Only queries that appear in the chosen split's qrels are kept. A dataset's
queries file covers every split, so keeping all of them would flood a test
run with queries that have no labels.

Usage:

    python -m retrieval_eval.beir scifact --split test --out data/scifact
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable
from pathlib import Path

from . import data

BEIR_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
)

# Published in the BEIR repository's dataset table.
KNOWN_MD5 = {"scifact": "5f7d1de60b170fc8027bb7898e2efca1"}

# BEIR's sentence-transformers wrapper joins title and text with a single
# space; matching it keeps results comparable with published figures.
TITLE_SEPARATOR = " "


def join_title(title: str, text: str) -> str:
    """Join a document's title and body the way BEIR does."""
    return f"{title}{TITLE_SEPARATOR}{text}".strip() if title else text.strip()


def _member(archive: zipfile.ZipFile, suffix: str) -> str:
    """Return the one archive member whose path ends with `/suffix`."""
    matches = [
        name
        for name in archive.namelist()
        if name == suffix or name.endswith(f"/{suffix}")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {suffix!r} in the archive, found {len(matches)}"
        )
    return matches[0]


def _read_jsonl(archive: zipfile.ZipFile, suffix: str) -> list[dict]:
    with archive.open(_member(archive, suffix)) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8")
        return [json.loads(line) for line in text if line.strip()]


def read_qrels(lines: Iterable[str]) -> list[dict]:
    """Parse a BEIR qrels TSV into label records, skipping the header row."""
    labels: list[dict] = []
    for row in csv.reader(lines, delimiter="\t"):
        if not row or row[0] == "query-id":
            continue
        query_id, doc_id, score = row
        labels.append({"query_id": query_id, "doc_id": doc_id, "relevance": int(score)})
    return labels


def convert_archive(
    archive_path: Path, split: str
) -> tuple[list[dict], list[dict], list[dict]]:
    """Read a BEIR archive and return documents, queries, and labels for a split.

    Documents keep corpus order and queries keep queries-file order, so the
    output is the same on every run.
    """
    with zipfile.ZipFile(archive_path) as archive:
        corpus = _read_jsonl(archive, "corpus.jsonl")
        all_queries = _read_jsonl(archive, "queries.jsonl")
        with archive.open(_member(archive, f"qrels/{split}.tsv")) as raw:
            labels = read_qrels(io.TextIOWrapper(raw, encoding="utf-8"))

    documents = [
        {"doc_id": row["_id"], "text": join_title(row.get("title", ""), row["text"])}
        for row in corpus
    ]
    judged = {label["query_id"] for label in labels}
    queries = [
        {"query_id": row["_id"], "text": row["text"]}
        for row in all_queries
        if row["_id"] in judged
    ]
    return documents, queries, labels


def file_digest(path: Path, algorithm: str) -> str:
    """Return the hex digest of a file, read in chunks."""
    digest = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, name: str) -> None:
    """Raise `ValueError` if the archive does not match BEIR's published md5.

    Datasets without a published checksum are accepted unchecked.
    """
    expected = KNOWN_MD5.get(name)
    if expected is None:
        return
    actual = file_digest(path, "md5")
    if actual != expected:
        raise ValueError(
            f"{path}: md5 {actual} does not match BEIR's published {expected}; "
            "delete the file and download again"
        )


def download(name: str, cache_dir: Path) -> Path:
    """Download a BEIR archive into `cache_dir` unless it is already there.

    Downloads go to a temporary file renamed on completion, so an interrupted
    download is never mistaken for a cached archive on the next run.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{name}.zip"
    if not destination.exists():
        partial = destination.with_suffix(".zip.part")
        url = BEIR_URL.format(name=name)
        print(f"Downloading {url}")
        with urllib.request.urlopen(url) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
        partial.rename(destination)
    verify_archive(destination, name)
    return destination


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_dataset(
    out_dir: Path,
    documents: list[dict],
    queries: list[dict],
    labels: list[dict],
    manifest: dict,
) -> None:
    """Write the three JSONL files and the manifest, then validate the result.

    Validation goes through the same loader a run uses, so a conversion that
    produces dangling or duplicate labels fails here rather than at run time.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "documents.jsonl", documents)
    _write_jsonl(out_dir / "queries.jsonl", queries)
    _write_jsonl(out_dir / "labels.jsonl", labels)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    data.load_dataset(
        str(out_dir / "documents.jsonl"),
        str(out_dir / "queries.jsonl"),
        str(out_dir / "labels.jsonl"),
    )


def prepare(name: str, split: str, out_dir: Path, cache_dir: Path) -> dict:
    """Download, convert, write, and validate one split. Returns the manifest.

    The manifest holds no timestamps, so preparing the same archive twice
    produces byte-identical files.
    """
    archive = download(name, cache_dir)
    documents, queries, labels = convert_archive(archive, split)
    manifest = {
        "dataset": name,
        "split": split,
        "source_url": BEIR_URL.format(name=name),
        "archive_md5": file_digest(archive, "md5"),
        "archive_sha256": file_digest(archive, "sha256"),
        "title_join": f"title + {TITLE_SEPARATOR!r} + text",
        "num_documents": len(documents),
        "num_queries": len(queries),
        "num_labels": len(labels),
    }
    write_dataset(out_dir, documents, queries, labels, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Prepare a BEIR dataset split; return a process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m retrieval_eval.beir",
        description="Convert a BEIR dataset split to the harness's JSONL format.",
    )
    parser.add_argument("name", help="BEIR dataset name, e.g. scifact")
    parser.add_argument("--split", default="test", help="qrels split (default: test)")
    parser.add_argument(
        "--out", type=Path, help="output directory (default: data/<name>)"
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/raw"),
        help="where downloaded archives are kept (default: data/raw)",
    )
    args = parser.parse_args(argv)

    out_dir = args.out or Path("data") / args.name
    try:
        manifest = prepare(args.name, args.split, out_dir, args.cache)
    except (ValueError, OSError, urllib.error.URLError) as exc:
        parser.exit(status=2, message=f"{parser.prog}: error: {exc}\n")

    print(
        f"Wrote {manifest['num_documents']} documents, "
        f"{manifest['num_queries']} queries, and {manifest['num_labels']} labels "
        f"to {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
