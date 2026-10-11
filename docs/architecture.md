# Architecture

```text
documents.jsonl ─┐
queries.jsonl  ──┤
labels.jsonl   ──┘
        │
        ▼
   data.load_dataset  ──►  validate_dataset (referential integrity)
        │
        ▼
   embeddings.Embedder.encode   (documents + queries → vectors)
        │
        ▼
   retrieval.cosine_top_k        (ranked to candidate_depth)
        │
        ├──►  metrics           (precision@k, recall@k, MRR, nDCG@k, at top_k)
        └──►  failure_analysis  (per-query outcome category, full candidate list)
        │
        ▼
   evaluation.run_evaluation  ──►  RunResult
        │
        ▼
   cli  ──►  metrics.json, retrieval_results.jsonl, failure_cases.jsonl, report.md
```

## Design choices

- **The `Embedder` interface** decouples retrieval from any model library, and
  lets the full pipeline run against a deterministic dummy embedder in tests.
- **Validation runs before embedding**, so a malformed dataset fails cheaply
  and every problem is reported in one pass.
- **Metrics are pure functions** with documented conventions, unit-tested
  against hand-computed values.
- **Two depths.** Documents are ranked to `candidate_depth` but scored at
  `top_k`. Metrics see only the top-k; failure analysis sees the deeper list,
  which is what lets it separate a relevant document ranked just below the cut
  from one never retrieved.
- **Deterministic output.** Stable sorting, explicit ordering of records, and
  sorted sets mean two runs over identical inputs write identical files.
