"""Index and search with Qdrant's in-memory mode and a fake embedder (no Docker, no model)."""

import numpy as np
import pytest
from qdrant_client import QdrantClient

from ragbench.chunking import paragraph_chunks
from ragbench.data import parse_paper
from ragbench.index import build_index, collection_name, point_id
from ragbench.retrieve import dense_search


class FakeEmbedder:
    """Bag-of-words hashed into a small vector: identical wording gives identical vectors."""

    model_name = "fake"
    dim = 64

    def __init__(self):
        self.document_calls = 0

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim)
        for word in text.lower().split():
            vector[sum(map(ord, word)) % self.dim] += 1
        return vector / max(np.linalg.norm(vector), 1e-9)

    def encode_documents(self, texts):
        self.document_calls += 1
        return np.stack([self._vector(t) for t in texts])

    def encode_query(self, text):
        return self._vector(text)


@pytest.fixture
def client():
    return QdrantClient(location=":memory:")


@pytest.fixture
def paper(raw_paper):
    return parse_paper(raw_paper)[0]


def other_paper(raw_paper):
    raw = {**raw_paper, "id": "9999.0000"}
    return parse_paper(raw)[0]


def test_collection_name_separates_setups():
    assert collection_name("validation", "paragraph", "BAAI/bge-small-en-v1.5") == (
        "validation__paragraph__bge-small-en-v1.5"
    )


def test_point_id_is_stable_uuid():
    assert point_id("a:c1") == point_id("a:c1")
    assert point_id("a:c1") != point_id("a:c2")


def test_search_ranks_the_matching_paragraph_first(client, paper):
    embedder = FakeEmbedder()
    build_index(client, "c", paragraph_chunks(paper, with_section=False), embedder)
    hits = dense_search(client, "c", embedder.encode_query("TinyNet has two layers."), paper.id, 3)
    assert hits[0].paragraph_ids == ("1234.5678:p2",)
    assert hits[0].text == "TinyNet has two layers."
    assert len(hits) == 3


def test_search_is_scoped_to_the_paper(client, paper, raw_paper):
    embedder = FakeEmbedder()
    chunks = paragraph_chunks(paper) + paragraph_chunks(other_paper(raw_paper))
    build_index(client, "c", chunks, embedder)
    hits = dense_search(client, "c", embedder.encode_query("TinyNet"), "9999.0000", 10)
    assert hits
    assert all(pid.startswith("9999.0000:") for h in hits for pid in h.paragraph_ids)


def test_build_index_reuses_a_matching_collection_and_rebuilds_otherwise(client, paper):
    embedder = FakeEmbedder()
    chunks = paragraph_chunks(paper)
    assert build_index(client, "c", chunks, embedder).built
    assert not build_index(client, "c", chunks, embedder).built
    assert embedder.document_calls == 1

    assert build_index(client, "c", chunks[:2], embedder).built
    assert client.count("c", exact=True).count == 2
