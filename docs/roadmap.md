# Roadmap

Deferred work, with the reason for deferring it. The core harness is
deliberately small; anything here was considered and left out on purpose.

Status: `planned` (intended next), `considered` (would be useful, not
committed), `rejected` (deliberately out of scope), `done` (shipped, kept here
so the reasoning stays next to the rest).

## Input quality

**Truncation counts in the report** — `done`

The report's Inputs section counts documents and queries longer than the
model's input limit. On SciFact this is the number that says how much of the
corpus the model never sees.

**Does truncation cost accuracy?** — `planned`

On SciFact, 71% of abstracts exceed `all-MiniLM-L6-v2`'s 256 word-piece limit,
so most of the corpus is embedded from its opening only. Whether that matters
is untested. The direct check is a comparison run against a model with a longer
input limit, which needs the `compare` command below. Worth reading alongside
the 22 claims whose relevant abstract was not retrieved at all: if their
evidence sits late in the abstract, truncation is a suspect.

**Corpus statistics in the report** — `planned`

Extends the Inputs section: min/median/max text length, empty texts, exact
duplicate text groups, and a count of documents containing markup-like
substrings. Characterizes the inputs without modifying them, so a reader who
sees an unexpected score has somewhere to look first.

**Near-duplicate detection** — `considered`

Exact matching misses documents differing only by whitespace, punctuation, or
markup. Shingling or minhash would catch them. Needs a normalization scheme,
which conflicts with treating preprocessing as part of the configuration under
test — so it would have to report, never rewrite.

**Input normalization** — `rejected`

Tag stripping, punctuation standardization, case folding. Rejected because
preprocessing belongs in the configuration under test: a harness that
normalizes silently measures its own assumptions alongside the model, and
makes "does cleaning improve recall@10?" unanswerable. See `limitations.md`.

## Reproducibility

**Pin model revisions** — `done`

`embedding.revision` is passed to the backend and shown in the report, and the
shipped configs pin a commit hash. What the report records is the configured
revision, not one resolved at run time: with a commit hash the two are the
same, which is why the configs use one.

**Record library versions in the run** — `considered`

Pinned weights still run through `sentence-transformers`, `transformers`, and
`torch`. Writing their versions into the report would make a run
self-describing without needing `uv.lock` alongside it.

## Retrieval

**FAISS backend** — `considered`

Cosine over normalized vectors is correct and trivially testable at sample
scale. FAISS would slot in behind the same interface and only matters once the
corpus outgrows a dense matmul.

**BM25 / lexical baseline and hybrid retrieval** — `considered`

A lexical baseline is the honest control for "is the embedding model actually
helping?". The most valuable single addition here, and the one most likely to
change a conclusion.

## Evaluation

**BEIR SciFact as the reference dataset** — `planned`

A conversion script from BEIR's format (`_id`, title and text, qrels TSV) to
the harness's JSONL, filtered to the 300-claim test split, with the data
downloaded rather than committed (SciFact is CC BY-NC 2.0). Replaces the
illustrative sample with a real benchmark, and its published nDCG@10 figures
give an external check that the metrics are implemented correctly. Scored at
`top_k: 10` to match how those figures are reported.

**Config comparison run** — `planned`

`configs/compare_configs.yaml` exists but nothing consumes it. A `compare`
subcommand running two configs and emitting a side-by-side metric table is the
last piece of the original scope, and the feature the harness is really for.

**Slice-based metrics** — `considered`

Aggregate metrics hide subgroup failures. Slicing by query type or length
would surface them. Needs slice metadata in the query file.

**Confidence intervals / significance tests** — `considered`

Currently two configs are compared by raw metric difference with no sense of
whether the gap is noise. Bootstrap CIs over per-query scores would fix this.
Meaningless on the current sample size.

## Engineering

**Type-check config values** — `considered`

Check each field against its dataclass annotation at load time, so
`top_k: "five"` is reported as a config error alongside the other problems
instead of surfacing later inside retrieval. A small loop over `fields()`.
Validation libraries such as pydantic would do this, but the cross-section
collect-all reporting would still need writing by hand.

**Embedding cache** — `considered`

Re-embedding an unchanged corpus on every run is the main cost in repeated
comparison runs. Keyed by model name plus text hash.

**GitHub Actions running the test suite** — `planned`

`make check` on push. Cheap, and makes the green suite visible rather than
claimed.

## Extensions

**RAG evaluation layer** — `considered`

Generation over the retrieved context, plus citation correctness
(deterministic), faithfulness, and answer relevance (LLM-as-judge). Extends
retrieval evaluation into RAG evaluation while reusing the config, reporting,
and failure-case machinery. Judge calls would need caching and a pinned model
to stay reproducible, and the limitations of LLM-as-judge would need stating
as plainly as everything else here. Roughly 10–15 hours on top of the core.

**Chatbot, agent, UI, LangChain/LlamaIndex, fine-tuning, cloud deployment** —
`rejected`

Out of scope. This is an evaluation harness, not an application.
