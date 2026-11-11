"""Dense retrieval from Qdrant, restricted to one paper."""

from dataclasses import dataclass

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
