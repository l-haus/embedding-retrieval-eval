"""Embedding backends.

The embedding model sits behind a small interface so retrieval and evaluation
depend on no specific library. That is what makes models swappable for
comparison runs, and what lets tests inject a fast dummy embedder instead of
downloading weights.

The concrete `sentence-transformers` backend has no unit test of its own, since
instantiating it downloads a model. What is tested is the interface, through the
dummy embedder in the end-to-end test.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Encodes a list of texts into a 2D float array of shape (len(texts), dim)."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder(Embedder):
    """Embedder backed by `sentence-transformers`.

    `sentence_transformers` is imported inside `__init__` rather than at module
    scope: `evaluation` imports this module only for the `Embedder` type, and a
    top-level import would pull torch and transformers into every run that never
    touches a real model, including the test suite.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

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


def get_embedder(model_name: str) -> Embedder:
    """Return an `Embedder` for the given model name.

    The indirection is the seam for alternative backends — a cached or lexical
    embedder would be dispatched here without touching `evaluation`.
    """
    return SentenceTransformerEmbedder(model_name)
