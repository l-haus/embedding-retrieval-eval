"""Tests for the sentence-transformers backend.

A stand-in `sentence_transformers` module replaces the real library, so these
check how the backend calls it — revision pinning, output shape and dtype, and
truncation counting — without downloading a model or importing torch.
"""

import sys
import types
from typing import ClassVar

import numpy as np
import pytest

from retrieval_eval import embeddings


class FakeTokenizer:
    """Tokenizes on whitespace and adds two special tokens, like [CLS]/[SEP]."""

    def __call__(self, texts, add_special_tokens, truncation, verbose):
        assert add_special_tokens and not truncation
        return {"input_ids": [[0] * (len(t.split()) + 2) for t in texts]}


class FakeSentenceTransformer:
    instances: ClassVar[list["FakeSentenceTransformer"]] = []

    def __init__(self, model_name, revision=None):
        self.model_name = model_name
        self.revision = revision
        self.max_seq_length = 6
        self.tokenizer = FakeTokenizer()
        FakeSentenceTransformer.instances.append(self)

    def encode(self, texts, convert_to_numpy):
        if not texts:
            return np.zeros((0,))  # the real backend returns 1D when empty
        return np.ones((len(texts), 4), dtype=np.float64)

    def get_sentence_embedding_dimension(self):
        return 4


@pytest.fixture
def fake_backend(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    FakeSentenceTransformer.instances = []
    return FakeSentenceTransformer


def test_revision_is_passed_to_the_backend(fake_backend):
    embeddings.get_embedder("some/model", revision="abc123")
    (model,) = fake_backend.instances
    assert (model.model_name, model.revision) == ("some/model", "abc123")


def test_revision_defaults_to_unpinned(fake_backend):
    embeddings.get_embedder("some/model")
    assert fake_backend.instances[0].revision is None


def test_encode_returns_float32_2d(fake_backend):
    out = embeddings.get_embedder("m").encode(["a", "b"])
    assert out.shape == (2, 4)
    assert out.dtype == np.float32


def test_encode_keeps_2d_shape_for_empty_input(fake_backend):
    assert embeddings.get_embedder("m").encode([]).shape == (0, 4)


def test_count_truncated_includes_special_tokens(fake_backend):
    embedder = embeddings.get_embedder("m")  # max_seq_length 6, +2 special tokens
    texts = ["one two three four", "one two three four five", "a"]
    # 4+2=6 fits exactly; 5+2=7 is over; 1+2=3 fits.
    assert embedder.count_truncated(texts) == 1


def test_count_truncated_of_nothing_is_zero(fake_backend):
    assert embeddings.get_embedder("m").count_truncated([]) == 0


def test_base_embedder_reports_truncation_as_unknown():
    class Minimal(embeddings.Embedder):
        def encode(self, texts):
            return np.zeros((len(texts), 2))

    assert Minimal().count_truncated(["anything"]) is None
