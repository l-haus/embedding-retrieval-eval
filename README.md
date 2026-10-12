# Embedding Retrieval Evaluation Harness

Embedding-based retrieval is usually demonstrated with a handful of queries that
work. This harness measures whether it works across a labelled set, and when it
doesn't, why: given documents, queries, and relevance labels, it ranks the
corpus for each query, computes standard retrieval metrics, sorts every failure
into an explainable category, and writes a report.

Every number is traceable to the exact data, model weights, and settings that
produced it. Inputs are validated before any embedding work, model revisions
are pinned, and two runs over the same inputs write byte-identical output.

## Results on BEIR SciFact

[SciFact](https://github.com/allenai/scifact) pairs 300 scientific claims with
5,183 abstracts. With `all-MiniLM-L6-v2`, pinned to a fixed commit:

| metric | value |
|---|---|
| nDCG@10 | 0.6451 |
| recall@10 | 0.7833 |
| MRR@10 | 0.6047 |
| precision@10 | 0.0883 |

**The result matches published runs.** Independent evaluations of the same model
on SciFact report nDCG@10 of
[0.6451](https://github.com/NikosMav/scifact-hybrid-retrieval), which this
harness reproduces to four decimal places. That is an outside check on the
metrics, the data conversion, and the way titles and abstracts are joined. For
context, BEIR's BM25 baseline scores 0.665: a keyword search still edges out
this model here.

**Precision is near its ceiling.** 277 of the 300 claims have exactly one
relevant abstract, so the best achievable precision@10 is 0.113. The model
reaches 78% of it, in line with its recall.

**The failures split into near misses and real misses.** 62 claims retrieved no
relevant abstract in the top 10. In 40 of them the abstract ranked between 11
and 100, close enough for a reranker to recover. In the other 22 it was not in
the top 100 at all.

How far a miss falls tracks how much vocabulary the claim shares with its
evidence. The median share of a claim's content words that appear in its
relevant abstract is 59% for claims retrieved correctly, 33% for near misses,
and 19% for real misses. Connecting the real misses takes a step of domain
knowledge:

| claim | relevant abstract |
|---|---|
| A total of 1,000 people in the UK are asymptomatic carriers of vCJD infection. | Prevalent abnormal prion protein in human appendixes after bovine spongiform encephalopathy epizootic |
| Aspirin inhibits the production of PGE2. | Cyclooxygenase-Dependent Tumor Growth through Evasion of Immunity |
| DMRT1 is a sex-determining gene that is epigenetically regulated by the MHM region. | Somatic sex identity is cell-autonomous in the chicken |

vCJD is a prion disease, and aspirin blocks the cyclooxygenase that produces
PGE2. Because these pairs share almost no words, a lexical baseline is unlikely
to recover them either; a model trained on biomedical text is the more promising
direction.

**Most of the corpus is truncated.** 3,681 of the 5,183 abstracts (71%) exceed
the model's 256 word-piece input limit, so they are embedded from their opening
only. Whether that costs accuracy is the next experiment
([roadmap](docs/roadmap.md)).

The full report is in [`reports/example_report.md`](reports/example_report.md).

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/l-haus/embedding-retrieval-eval
cd embedding-retrieval-eval
make sync          # install the project and dev dependencies
make test          # run the test suite
make run           # evaluate the bundled sample (6 documents, 3 queries)
make run-scifact   # download SciFact, verify it, and evaluate it
```

Each run writes four files to the configured run directory:

| file | contents |
|---|---|
| `metrics.json` | aggregate metrics |
| `retrieval_results.jsonl` | per-query top-k results and metrics |
| `failure_cases.jsonl` | every query that missed, with its category and missed documents |
| `report.md` | a readable summary of all of the above |

Without `make`: `pip install -e ".[dev]"`, then
`retrieval-eval run --config configs/example.yaml`.

## Inputs

Three JSONL files:

```json
{"doc_id": "d1", "text": "The system validates manifests before batch processing."}
{"query_id": "q1", "text": "How does the pipeline catch bad inputs?"}
{"query_id": "q1", "doc_id": "d1", "relevance": 1}
```

Before anything is embedded, the dataset is checked. A malformed record (invalid
JSON, a missing field, a duplicate id) stops loading at that record, with its
position in the file. Problems across files (labels pointing at unknown
documents or queries, negative relevance, repeated `(query_id, doc_id)` pairs)
are collected and reported together, each with the label it came from.

Text is used exactly as given; normalization is treated as part of the
configuration under test, not something the harness does silently.

Any [BEIR](https://github.com/beir-cellar/beir) dataset can be converted:

```bash
python -m retrieval_eval.beir scifact --split test --out data/scifact
```

This keeps only queries judged in the chosen split, joins titles and text the
way BEIR does, checks the archive against BEIR's published checksum where one
exists, and writes a manifest recording what was produced.

## Configuration

```yaml
dataset:
  documents_path: data/scifact/documents.jsonl
  queries_path: data/scifact/queries.jsonl
  labels_path: data/scifact/labels.jsonl

retrieval:
  top_k: 10              # metrics are computed at this depth
  candidate_depth: 100   # ranking depth used by failure analysis
  similarity: cosine     # the only one implemented; others are rejected

embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  revision: 1110a243fdf4706b3f48f1d95db1a4f5529b4d41   # optional, recommended

output:
  run_dir: runs/scifact
```

A config with a missing section, a missing required field, or an unknown field
fails with every problem listed. Runs without a pinned `revision` are marked as
unpinned in the report.

## How the numbers are defined

Retrieval metrics have common variants that give different numbers. These are
the ones used here:

| metric | convention |
|---|---|
| precision@k | relevant documents in the top k, divided by k |
| recall@k | relevant documents in the top k, divided by all relevant documents |
| MRR@k | 1 / rank of the first relevant document, or 0 if none is in the top k |
| nDCG@k | the ideal ranking is built from every judged document, including ones never retrieved, then scored to depth k |

Averages cover only queries with at least one relevant document. Unlabelled
queries are counted separately and never reported as retrieval failures.

Each query that misses is assigned one category:

| category | meaning |
|---|---|
| `relevant_ranked_low` | a relevant document was ranked, but below `top_k` |
| `no_relevant_retrieved` | no relevant document within `candidate_depth` |
| `no_labels` | the query has no relevance labels and cannot be judged |

Documents are ranked to `candidate_depth` but scored at `top_k`. The deeper list
is what lets the first two categories be told apart.

## Design

The pipeline and the reasoning behind it are described in
[`docs/architecture.md`](docs/architecture.md). In short:

- **The embedding model sits behind an interface**, so the full pipeline runs in
  tests against a deterministic stand-in, with no download and no network.
- **Validation runs first**, so a malformed dataset fails in seconds, not after
  an embedding pass.
- **Metrics are pure functions**, unit-tested against hand-computed values.
- **Output is deterministic.** Ties are broken stably, records are written in
  dataset order, and sets are sorted before they are written.

## Limitations

[`docs/limitations.md`](docs/limitations.md) lists what this harness does not do
and where its numbers can mislead. The most important:

- A retrieved document with no label counts as non-relevant, which understates
  precision on sparsely judged data.
- No significance testing, so a small gap between two runs may be noise.
- Long texts are truncated by the model; the report counts how many, but not
  whether it mattered.

Planned work, and work deliberately left out, is in
[`docs/roadmap.md`](docs/roadmap.md).

## Development

```bash
make hooks   # install pre-commit hooks (ruff lint and format)
make check   # lint, then run the tests
make cov     # tests with a coverage report
```

## Data and license

SciFact is released under CC BY-NC 2.0 (see the
[SciFact repository](https://github.com/allenai/scifact)). It is
downloaded on demand from BEIR and not redistributed here; the example report
quotes a small number of claims and titles from it.

If you use these results, please cite the datasets:

- Wadden et al., *Fact or Fiction: Verifying Scientific Claims*, EMNLP 2020.
- Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of
  Information Retrieval Models*, NeurIPS 2021 Datasets and Benchmarks Track.

The code is released under the license in [`LICENSE`](LICENSE).
