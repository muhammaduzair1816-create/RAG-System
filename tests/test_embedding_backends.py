"""Cross-backend equivalence for the embedding layer.

The ONNX backend exists to cut ~350 MB of resident memory. That is only a safe
default if it returns the *same* vectors as the torch backend — otherwise
switching would silently invalidate every index built with the other one.
"""

from __future__ import annotations

import dataclasses
from importlib.util import find_spec

import numpy as np
import pytest

from src.config import Settings
from src.embeddings import get_embedder, reset_embedder_cache

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

TEXTS = [
    "Pinecone supports metadata filtering on vector queries.",
    "The cat sat on a warm windowsill during the afternoon.",
    "Retrieval augmented generation keeps answers anchored to a trusted corpus.",
]
QUERY = "How does Pinecone handle metadata filtering?"

requires_onnx = pytest.mark.skipif(
    find_spec("onnxruntime") is None or find_spec("tokenizers") is None,
    reason="onnxruntime/tokenizers not installed",
)
requires_torch = pytest.mark.skipif(
    find_spec("sentence_transformers") is None,
    reason="sentence-transformers not installed (optional backend)",
)


def _embedder(backend: str):
    reset_embedder_cache()
    return get_embedder(dataclasses.replace(Settings(), embedding_backend=backend))


@pytest.fixture(scope="module")
def onnx_embedder():
    return _embedder("onnx")


@requires_onnx
def test_onnx_backend_reports_the_expected_shape(onnx_embedder):
    assert onnx_embedder.dimension == 384
    vectors = onnx_embedder.embed_documents(TEXTS)
    assert vectors.shape == (3, 384)
    assert vectors.dtype == np.float32


@requires_onnx
def test_onnx_vectors_are_l2_normalised(onnx_embedder):
    norms = np.linalg.norm(onnx_embedder.embed_documents(TEXTS), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


@requires_onnx
def test_onnx_batching_does_not_change_results(onnx_embedder):
    """Peak memory is controlled by batch size, so it must not affect output."""
    one = onnx_embedder.embed_documents(TEXTS, batch_size=1)
    many = onnx_embedder.embed_documents(TEXTS, batch_size=8)
    assert np.allclose(one, many, atol=1e-6)


@requires_onnx
def test_onnx_separates_relevant_from_irrelevant(onnx_embedder):
    query = onnx_embedder.embed_query(QUERY)
    docs = onnx_embedder.embed_documents(TEXTS)
    assert float(query @ docs[0]) > 0.6   # on topic
    assert float(query @ docs[1]) < 0.2   # unrelated


@requires_onnx
@requires_torch
def test_onnx_and_torch_backends_agree(onnx_embedder):
    """The claim that makes onnx a safe default: identical vectors."""
    onnx_docs = onnx_embedder.embed_documents(TEXTS)
    onnx_query = onnx_embedder.embed_query(QUERY)

    torch_embedder = _embedder("sentence-transformers")
    torch_docs = torch_embedder.embed_documents(TEXTS)
    torch_query = torch_embedder.embed_query(QUERY)

    assert onnx_embedder.dimension == torch_embedder.dimension

    per_doc_cosine = (onnx_docs * torch_docs).sum(axis=1)
    assert per_doc_cosine.min() > 0.99999, f"vectors diverge: {per_doc_cosine}"
    assert np.abs(onnx_docs - torch_docs).max() < 1e-5
    assert float(onnx_query @ torch_query) > 0.99999

    # Retrieval scores are what Pinecone ranks on; they must match too.
    assert np.allclose(onnx_query @ onnx_docs.T, torch_query @ torch_docs.T, atol=1e-5)
