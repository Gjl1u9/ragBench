"""Embed chunks and store them in Qdrant.

All papers of a split share one collection, and queries filter on `paper_id`. The collection
name includes the split, chunker and model, so indexes for different setups live side by side.
"""

import re
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from ragbench.chunking import Chunk

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# BGE v1.5 models expect this instruction on queries (not on documents).
QUERY_PREFIXES = {DEFAULT_MODEL: "Represent this sentence for searching relevant passages: "}
UPSERT_BATCH = 256


class Embedder(Protocol):
    model_name: str
    dim: int

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        self._query_prefix = QUERY_PREFIXES.get(model_name, "")

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
        )

    def encode_query(self, text: str) -> np.ndarray:
        return self._model.encode(self._query_prefix + text, normalize_embeddings=True)

    def count_truncated(self, texts: list[str]) -> int:
        """How many texts exceed the model's input limit and lose their tail when embedded."""
        limit = self._model.max_seq_length
        tokenized = self._model.tokenizer(texts, truncation=False, verbose=False)
        return sum(len(ids) > limit for ids in tokenized["input_ids"])


@dataclass(frozen=True)
class IndexResult:
    collection: str
    points: int
    built: bool  # False when an up-to-date collection was reused
    seconds: float


def collection_name(split: str, chunker: str, model_name: str) -> str:
    model = re.sub(r"[^a-z0-9.-]+", "-", model_name.split("/")[-1].lower())
    return f"{split}__{chunker}__{model}"


def point_id(chunk_id: str) -> str:
    # Qdrant ids must be integers or UUIDs; derive a stable UUID from the chunk id.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def build_index(
    client: QdrantClient, collection: str, chunks: list[Chunk], embedder: Embedder
) -> IndexResult:
    """Create the collection, or reuse it if it already holds exactly these chunks."""
    start = time.perf_counter()
    if client.collection_exists(collection) and client.count(collection, exact=True).count == len(
        chunks
    ):
        return IndexResult(collection, len(chunks), False, time.perf_counter() - start)

    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection, vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE)
    )
    client.create_payload_index(collection, "paper_id", PayloadSchemaType.KEYWORD)

    vectors = embedder.encode_documents([c.text for c in chunks])
    for batch in _batched(zip(chunks, vectors, strict=True), UPSERT_BATCH):
        client.upsert(
            collection,
            points=[
                PointStruct(
                    id=point_id(chunk.id),
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": chunk.id,
                        "paper_id": chunk.paper_id,
                        "section": chunk.section,
                        "text": chunk.text,
                        "paragraph_ids": list(chunk.paragraph_ids),
                    },
                )
                for chunk, vector in batch
            ],
        )
    return IndexResult(collection, len(chunks), True, time.perf_counter() - start)


def _batched(items: Iterable, size: int) -> Iterable[list]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
