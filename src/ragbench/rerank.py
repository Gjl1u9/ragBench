"""Rerank retrieved chunks with a cross-encoder.

A cross-encoder reads the question and a chunk together and scores how well the chunk answers
it. That is more accurate than comparing separate embeddings, but every (question, chunk) pair
needs its own forward pass, so it only reorders a short candidate list. It can never recover a
chunk that is missing from the candidates.
"""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from sentence_transformers import CrossEncoder

from ragbench.retrieve import Hit

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def order_by_scores(hits: list[Hit], scores: Sequence[float], k: int) -> list[Hit]:
    """Sort hits by the given scores, best first; ties keep the original order."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")[:k]
    return [replace(hits[i], score=float(scores[i])) for i in order]


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER):
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self._model.predict(
            [(query, hit.text) for hit in hits], batch_size=64, show_progress_bar=False
        )
        return order_by_scores(hits, scores, k)
