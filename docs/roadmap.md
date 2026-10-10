# Roadmap

Deferred work, with the reason for deferring it. The core harness is
deliberately small; anything here was considered and left out on purpose.

Status: `planned` (intended next), `considered` (would be useful, not
committed), `rejected` (deliberately out of scope).

## Input quality

**Corpus statistics in the report** — `planned`

Document and query counts, min/median/max text length, empty texts, exact
duplicate text groups, and a count of documents containing markup-like
substrings. Characterizes the inputs without modifying them, so a reader who
sees an unexpected score has somewhere to look first. Roughly 20 lines plus a
report section. Deferred until the core run is green end to end.

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

**Pin model revisions** — `planned`

Accept an optional revision hash beside `model_name` in the embedding config,
pass it to the backend, and record the resolved revision in the run metadata so
a report states exactly which weights produced it. Small change, and it closes
the largest hole in the harness's reproducibility claim.

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
