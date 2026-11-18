"""Score retrieval against QASPER's gold evidence. Makes no LLM calls.

Run with: uv run python -m ragbench.evaluate --retriever hybrid --split validation

Scoring rules:
- Annotators who marked a question unanswerable, or whose evidence resolved to no paragraph,
  have no gold set and are skipped. Questions left with no gold set are not scored.
- A question's score is the best score across its annotators, as is conventional for QASPER.
- A retrieved chunk counts for every paragraph it covers, so chunkers can be compared.
"""

import argparse
import json
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime

from qdrant_client import QdrantClient

from ragbench.bm25 import BM25Retriever
from ragbench.chunking import CHUNKERS
from ragbench.config import settings
from ragbench.data import Paper, Question, load_split
from ragbench.index import (
    DEFAULT_MODEL,
    SentenceTransformerEmbedder,
    build_index,
    collection_name,
)
from ragbench.retrieve import Hit, dense_search, reciprocal_rank_fusion

KS = (1, 3, 5, 10)
RETRIEVERS = ("dense", "bm25", "hybrid")
# How many candidates each retriever contributes before fusion. Papers average ~54 chunks,
# so 50 leaves fusion almost the whole paper to rank.
HYBRID_CANDIDATES = 50

Ranked = Sequence[Sequence[str]]  # paragraph ids of each retrieved chunk, best first
Retriever = Callable[[Question, int], list[Hit]]


def recall_at_k(gold: set[str], ranked: Ranked, k: int) -> float:
    found = {pid for ids in ranked[:k] for pid in ids}
    return len(gold & found) / len(gold)


def reciprocal_rank(gold: set[str], ranked: Ranked) -> float:
    for rank, ids in enumerate(ranked, start=1):
        if gold.intersection(ids):
            return 1 / rank
    return 0.0


def gold_sets(question: Question) -> list[set[str]]:
    return [
        set(a.evidence_ids) for a in question.answers if a.type != "unanswerable" and a.evidence_ids
    ]


def score_question(question: Question, ranked: Ranked, ks: Sequence[int] = KS) -> dict | None:
    golds = gold_sets(question)
    if not golds:
        return None
    scores = {f"recall@{k}": max(recall_at_k(g, ranked, k) for g in golds) for k in ks}
    scores["mrr"] = max(reciprocal_rank(g, ranked) for g in golds)
    return scores


def evaluate(questions: list[Question], retrieve: Retriever, ks: Sequence[int] = KS) -> dict:
    """Run `retrieve` on every scorable question and average the metrics."""
    rows = []
    latencies_ms = []
    scorable = [q for q in questions if gold_sets(q)]
    for question in scorable:
        start = time.perf_counter()
        hits = retrieve(question, max(ks))
        latencies_ms.append((time.perf_counter() - start) * 1000)
        rows.append(score_question(question, [h.paragraph_ids for h in hits], ks))
    return {
        "questions_total": len(questions),
        "questions_scored": len(rows),
        "metrics": {key: round(statistics.mean(r[key] for r in rows), 4) for key in rows[0]},
        "latency_ms": {
            "median": round(statistics.median(latencies_ms), 2),
            "p95": round(statistics.quantiles(latencies_ms, n=20)[-1], 2),
        },
    }


def chance_fraction(papers: list[Paper], questions: list[Question], chunker, k: int) -> float:
    """Expected share of a paper's chunks that k random picks would return: a floor to beat."""
    sizes = {p.id: len(chunker(p)) for p in papers}
    fractions = [min(k, sizes[q.paper_id]) / sizes[q.paper_id] for q in questions if gold_sets(q)]
    return round(statistics.mean(fractions), 4)


def _git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retriever", default="dense", choices=RETRIEVERS)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--chunker", default="paragraph", choices=sorted(CHUNKERS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--final", action="store_true", help="required for the test split: final numbers only"
    )
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        parser.error("the test split is for final reported numbers; pass --final to confirm")

    papers, questions = load_split(args.split)
    chunker = CHUNKERS[args.chunker]
    chunks = [c for p in papers for c in chunker(p)]

    uses_dense = args.retriever in ("dense", "hybrid")
    config = {
        "retriever": args.retriever,
        "split": args.split,
        "chunker": args.chunker,
        "model": args.model if uses_dense else None,
        "retrieval_scope": "paper",
    }
    extra: dict = {}

    if uses_dense:
        embedder = SentenceTransformerEmbedder(args.model)
        client = QdrantClient(url=settings.qdrant_url)
        collection = collection_name(args.split, args.chunker, args.model)
        extra["index"] = asdict(build_index(client, collection, chunks, embedder))
        extra["truncated_chunks"] = embedder.count_truncated([c.text for c in chunks])

        def dense(question: Question, k: int) -> list[Hit]:
            vector = embedder.encode_query(question.text)
            return dense_search(client, collection, vector, question.paper_id, k)

    if args.retriever in ("bm25", "hybrid"):
        bm25 = BM25Retriever(chunks)

        def keyword(question: Question, k: int) -> list[Hit]:
            return bm25.search(question.text, question.paper_id, k)

    if args.retriever == "dense":
        retrieve = dense
    elif args.retriever == "bm25":
        retrieve = keyword
    else:
        config.update(fusion="rrf", rrf_k=60, candidates=HYBRID_CANDIDATES)

        def retrieve(question: Question, k: int) -> list[Hit]:
            rankings = [dense(question, HYBRID_CANDIDATES), keyword(question, HYBRID_CANDIDATES)]
            return reciprocal_rank_fusion(rankings, k)

    report = evaluate(questions, retrieve)
    report = {
        "experiment": args.retriever,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config": config,
        "chunks": len(chunks),
        **extra,
        **report,
        "chance_fraction": {f"k={k}": chance_fraction(papers, questions, chunker, k) for k in KS},
        "answer_score": None,  # retrieval only: no generation, so no answer score or LLM cost
        "cost": None,
    }

    settings.results_dir.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = settings.results_dir / f"{stamp}-{args.retriever}-{args.split}-{args.chunker}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
