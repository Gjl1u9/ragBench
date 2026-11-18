"""Keyword retrieval with BM25, restricted to one paper.

Each paper gets its own BM25 index, so term statistics (IDF) come from that paper's chunks,
matching the per-paper scope of dense retrieval.
"""

import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from ragbench.chunking import Chunk
from ragbench.retrieve import Hit

# Keeps names like "bert-base", "gpt-2" and "1.5" together as single tokens.
TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        by_paper: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            by_paper[chunk.paper_id].append(chunk)
        self._papers = {
            paper_id: (paper_chunks, BM25Okapi([tokenize(c.text) for c in paper_chunks]))
            for paper_id, paper_chunks in by_paper.items()
        }

    def search(self, query: str, paper_id: str, k: int) -> list[Hit]:
        """Return the top-k chunks of `paper_id` by BM25 score, best first."""
        chunks, bm25 = self._papers[paper_id]
        scores = bm25.get_scores(tokenize(query))
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            Hit(chunks[i].id, float(scores[i]), chunks[i].text, chunks[i].paragraph_ids)
            for i in order
        ]
