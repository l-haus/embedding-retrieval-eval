# Limitations

What this harness does not do, and where its numbers can mislead. Read this
before drawing conclusions from a report.

## Data and labels

- **Small, synthetic labels.** The sample set is tiny and hand-labelled, so
  metrics are illustrative rather than statistically meaningful.
- **Binary-ish relevance.** Labels are mostly 1/0. nDCG supports graded
  relevance, but the sample data barely exercises it.
- **Unlabelled positives.** A retrieved document with no label counts as
  non-relevant, which understates precision on sparsely judged collections.
- **No statistical significance.** No confidence intervals or significance
  tests, so a small gap between two configurations may be noise.

## Text handling

- **Text is used as given.** No normalization of any kind: no tag stripping,
  no punctuation or whitespace standardization, no case folding, no Unicode
  normalization. Preprocessing is treated as part of the configuration under
  test, so a comparison between two configs measures the models rather than
  the harness's hidden assumptions. The consequence is that inconsistent input
  formatting shows up as lower scores with no indication of the cause. Clean
  and normalize upstream, consistently across documents and queries.
- **Long texts are truncated silently.** Embedding models have a maximum input
  length; `all-MiniLM-L6-v2` truncates anything beyond 256 word pieces. Text
  past that limit never reaches the model, and nothing in the report says how
  much of the corpus was cut. On collections of full abstracts this can be a
  large share of the text.
- **Duplicate detection is exact-match only.** Documents differing by
  whitespace, punctuation, or markup are distinct to this tool even when
  identical to a reader.
- **Duplicate text distorts scoring.** Documents are identified by `doc_id`,
  never by content. Two documents with identical text embed identically and tie
  on score; the stable sort breaks the tie by corpus order. If only one of them
  is labelled relevant, metrics depend on which appears first in the file — the
  same retrieval quality can score 1.0 or 0.0. Failure records compound this by
  listing the unretrieved twin under `missed_doc_ids`, pointing at a document
  whose text *was* returned. Near-duplicate corpora with sparse judgements will
  understate retrieval quality.

## Method

- **Single similarity metric.** Cosine only; any other `similarity` value in
  the config is rejected rather than silently ignored. No lexical or hybrid
  baseline in the core; see `roadmap.md`.
- **precision@k is capped by the number of relevant documents.** A query with
  two relevant documents scores at most 2/k, however good the ranking. On
  sparsely labelled data a low precision can therefore be a perfect result;
  read it alongside recall and nDCG, which are normalized to what was
  achievable.
- **MRR is MRR@k.** Like every other metric here, reciprocal rank is cut at
  `top_k`: a relevant document first appearing below the cut contributes 0,
  not 1/rank. Uncut MRR would credit retrieval the user never sees.
- **nDCG ideal ranking.** The ideal ranking is built from the full relevance
  set, including documents that were never retrieved, then scored to depth k.
  Missing a relevant document lowers nDCG, while a perfect top-k still scores
  1.0. This is the stricter of the two common conventions, chosen because
  failing to retrieve a relevant document is a retrieval failure.
- **Ties are deterministic, not meaningful.** Stable sorting makes repeated
  runs agree on tied scores, but the order it picks is corpus order, not a
  judgement about which document is better.

## Configuration

- **Config values are not type-checked.** Loading checks that sections and
  required fields are present and that no unknown fields appear, but not the
  types of values. `top_k: "five"` loads without complaint and fails later,
  inside retrieval, with an error that points away from the config file.

## Reproducibility

- **Model names are not pinned to a revision.** A config naming
  `sentence-transformers/all-MiniLM-L6-v2` resolves to whatever the model hub
  serves at run time. The same config can therefore load different weights
  months apart and produce different numbers, with nothing in the report
  indicating that anything changed. This is the largest remaining gap in the
  harness's reproducibility claim, and unlike the others it is inherited from
  the ecosystem rather than chosen. Recording a revision hash alongside the
  model name would close it; see `roadmap.md`.
