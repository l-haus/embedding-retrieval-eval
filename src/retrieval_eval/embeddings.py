"""Embedding backends.

The embedding model sits behind a small interface so retrieval and evaluation
depend on no specific library. That is what makes models swappable for
comparison runs, and what lets tests inject a fast dummy embedder instead of
downloading weights.

The `sentence-transformers` backend is tested against a stand-in for the
library, which checks how it is called without downloading a model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Encodes a list of texts into a 2D float array of shape (len(texts), dim)."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray: ...

    def count_truncated(self, texts: list[str]) -> int | None:
        """Return how many texts exceed the model's input limit.

        Returns None when the backend cannot tell, which the report states
        rather than implying nothing was truncated.
        """
        return None


class SentenceTransformerEmbedder(Embedder):
    """Embedder backed by `sentence-transformers`.

    `sentence_transformers` is imported inside `__init__` rather than at module
    scope: `evaluation` imports this module only for the `Embedder` type, and a
    top-level import would pull torch and transformers into every run that never
    touches a real model, including the test suite.
    """

    def __init__(self, model_name: str, revision: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.revision = revision
        self.model = SentenceTransformer(model_name, revision=revision)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return embeddings as a float32 array of shape (len(texts), dim).

        The 2D shape is guaranteed even for an empty input: the backend returns
        a 1D empty array in that case, which would fail downstream where
        `retrieval.normalize` reduces over axis 1.
        """
        vectors = self.model.encode(texts, convert_to_numpy=True)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            dim = self.model.get_sentence_embedding_dimension()
            vectors = vectors.reshape(len(texts), dim)
        return vectors

    def count_truncated(self, texts: list[str]) -> int:
        """Count texts whose tokenized length exceeds the model's maximum.

        Lengths include the special tokens the model adds, since the maximum
        does too. Anything past the limit is dropped silently by the backend.
        """
        if not texts:
            return 0
        limit = self.model.max_seq_length
        encoded = self.model.tokenizer(
            texts, add_special_tokens=True, truncation=False, verbose=False
        )
        return sum(len(ids) > limit for ids in encoded["input_ids"])


def get_embedder(model_name: str, revision: str | None = None) -> Embedder:
    """Return an `Embedder` for the given model name and revision."""
    return SentenceTransformerEmbedder(model_name, revision=revision)
