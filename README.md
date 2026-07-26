# Embedding Retrieval Evaluation Harness

Embedding-based retrieval systems are often demonstrated with a few successful
examples, but production use requires systematic evaluation.

This project provides a small, reproducible harness for evaluating semantic
retrieval systems. Given a document corpus, queries, and relevance labels, it
retrieves top-k results, computes retrieval metrics, compares configurations,
and surfaces failure cases for review.

The goal is not to build another chatbot. The goal is to make retrieval quality
**measurable, debuggable, and comparable** across model or configuration changes.
