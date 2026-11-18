"""Dense retrieval from Qdrant, restricted to one paper, and rank fusion of result lists."""

from dataclasses import dataclass, replace

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float
    text: str
    paragraph_ids: tuple[str, ...]


def dense_search(
    client: QdrantClient, collection: str, query_vector: np.ndarray, paper_id: str, k: int
) -> list[Hit]:
    """Return the top-k chunks of `paper_id`, best first.

    QASPER questions are about one specific paper, so retrieval is scoped to that paper.
    """
    response = client.query_points(
        collection,
        query=query_vector.tolist(),
        query_filter=Filter(
            must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
        ),
        limit=k,
        with_payload=True,
    )
    return [
        Hit(
            chunk_id=point.payload["chunk_id"],
            score=point.score,
            text=point.payload["text"],
            paragraph_ids=tuple(point.payload["paragraph_ids"]),
        )
        for point in response.points
    ]


def reciprocal_rank_fusion(rankings: list[list[Hit]], k: int, rrf_k: int = 60) -> list[Hit]:
    """Merge ranked lists by Reciprocal Rank Fusion; return the top k, best first.

    Each list contributes 1 / (rrf_k + rank) to a chunk's score. Only ranks are used, so
    BM25 scores and cosine similarities never need to share a scale. rrf_k = 60 is the
    customary default from the original RRF paper.
    """
    scores: dict[str, float] = {}
    hits: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1 / (rrf_k + rank)
            hits.setdefault(hit.chunk_id, hit)
    best = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
    return [replace(hits[chunk_id], score=scores[chunk_id]) for chunk_id in best]
