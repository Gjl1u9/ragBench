"""Assemble a retrieval pipeline (dense, BM25 or hybrid, optionally reranked) from options.

Shared by the retrieval evaluation and by generation, so both run exactly the same retrieval.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from qdrant_client import QdrantClient

from ragbench.bm25 import BM25Retriever
from ragbench.chunking import Chunk
from ragbench.config import settings
from ragbench.data import Question
from ragbench.index import DEFAULT_MODEL, SentenceTransformerEmbedder, build_index, collection_name
from ragbench.rerank import CrossEncoderReranker
from ragbench.retrieve import Hit, dense_search, reciprocal_rank_fusion

Retriever = Callable[[Question, int], list[Hit]]

RETRIEVERS = ("dense", "bm25", "hybrid")
# How many candidates each retriever contributes before fusion. Papers average ~54 chunks,
# so 50 leaves fusion almost the whole paper to rank.
HYBRID_CANDIDATES = 50
RERANK_CANDIDATES = 50


@dataclass
class Pipeline:
    retrieve: Retriever
    first_stage: Retriever  # what the reranker reorders; same as `retrieve` without reranking
    config: dict
    extra: dict = field(default_factory=dict)  # index and truncation facts for the report


def build_pipeline(
    split: str,
    chunker: str,
    chunks: list[Chunk],
    retriever: str = "dense",
    model: str = DEFAULT_MODEL,
    rerank: str | None = None,
    candidates: int = RERANK_CANDIDATES,
) -> Pipeline:
    if retriever not in RETRIEVERS:
        raise ValueError(f"unknown retriever {retriever!r}")
    uses_dense = retriever in ("dense", "hybrid")
    config: dict = {
        "retriever": retriever,
        "split": split,
        "chunker": chunker,
        "model": model if uses_dense else None,
        "retrieval_scope": "paper",
    }
    extra: dict = {}

    if uses_dense:
        embedder = SentenceTransformerEmbedder(model)
        client = QdrantClient(url=settings.qdrant_url)
        collection = collection_name(split, chunker, model)
        extra["index"] = asdict(build_index(client, collection, chunks, embedder))
        extra["truncated_chunks"] = embedder.count_truncated([c.text for c in chunks])

        def dense(question: Question, k: int) -> list[Hit]:
            vector = embedder.encode_query(question.text)
            return dense_search(client, collection, vector, question.paper_id, k)

    if retriever in ("bm25", "hybrid"):
        bm25 = BM25Retriever(chunks)

        def keyword(question: Question, k: int) -> list[Hit]:
            return bm25.search(question.text, question.paper_id, k)

    if retriever == "dense":
        first_stage = dense
    elif retriever == "bm25":
        first_stage = keyword
    else:
        config.update(fusion="rrf", rrf_k=60, candidates=HYBRID_CANDIDATES)

        def first_stage(question: Question, k: int) -> list[Hit]:
            rankings = [dense(question, HYBRID_CANDIDATES), keyword(question, HYBRID_CANDIDATES)]
            return reciprocal_rank_fusion(rankings, k)

    if not rerank:
        return Pipeline(first_stage, first_stage, config, extra)

    config.update(reranker=rerank, rerank_candidates=candidates)
    reranker = CrossEncoderReranker(rerank)

    def reranked(question: Question, k: int) -> list[Hit]:
        return reranker.rerank(question.text, first_stage(question, candidates), k)

    return Pipeline(reranked, first_stage, config, extra)
