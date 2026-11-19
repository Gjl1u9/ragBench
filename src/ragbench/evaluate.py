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
from collections.abc import Sequence
from datetime import UTC, datetime

from ragbench.chunking import CHUNKERS
from ragbench.config import settings
from ragbench.data import Paper, Question, load_split
from ragbench.index import DEFAULT_MODEL
from ragbench.pipeline import RERANK_CANDIDATES, RETRIEVERS, Retriever, build_pipeline
from ragbench.rerank import DEFAULT_RERANKER

KS = (1, 3, 5, 10)

Ranked = Sequence[Sequence[str]]  # paragraph ids of each retrieved chunk, best first


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
    rows: dict[str, dict] = {}
    latencies_ms = []
    scorable = [q for q in questions if gold_sets(q)]
    for question in scorable:
        start = time.perf_counter()
        hits = retrieve(question, max(ks))
        latencies_ms.append((time.perf_counter() - start) * 1000)
        rows[question.id] = score_question(question, [h.paragraph_ids for h in hits], ks)
    keys = next(iter(rows.values()))
    return {
        "questions_total": len(questions),
        "questions_scored": len(rows),
        "metrics": {key: round(statistics.mean(r[key] for r in rows.values()), 4) for key in keys},
        "latency_ms": {
            "median": round(statistics.median(latencies_ms), 2),
            "p95": round(statistics.quantiles(latencies_ms, n=20)[-1], 2),
        },
        # Kept so two runs can be compared question by question (paired tests) without rerunning.
        "per_question": rows,
    }


def chance_fraction(papers: list[Paper], questions: list[Question], chunker, k: int) -> float:
    """Expected share of a paper's chunks that k random picks would return: a floor to beat."""
    sizes = {p.id: len(chunker(p)) for p in papers}
    fractions = [min(k, sizes[q.paper_id]) / sizes[q.paper_id] for q in questions if gold_sets(q)]
    return round(statistics.mean(fractions), 4)


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retriever", default="dense", choices=RETRIEVERS)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--chunker", default="paragraph", choices=sorted(CHUNKERS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--rerank",
        nargs="?",
        const=DEFAULT_RERANKER,
        metavar="MODEL",
        help=f"rerank the retriever's candidates with a cross-encoder (default {DEFAULT_RERANKER})",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=RERANK_CANDIDATES,
        help="how many candidates to rerank (with --rerank)",
    )
    parser.add_argument(
        "--final", action="store_true", help="required for the test split: final numbers only"
    )
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        parser.error("the test split is for final reported numbers; pass --final to confirm")

    papers, questions = load_split(args.split)
    chunker = CHUNKERS[args.chunker]
    chunks = [c for p in papers for c in chunker(p)]

    if args.rerank and args.candidates < max(KS):
        parser.error(f"--candidates must be at least {max(KS)}")
    pipeline = build_pipeline(
        args.split, args.chunker, chunks, args.retriever, args.model, args.rerank, args.candidates
    )

    experiment = args.retriever
    if args.rerank:
        experiment += "+rerank"
        # The reranker can only reorder what it is given, so this recall is its ceiling.
        ceiling = evaluate(questions, pipeline.first_stage, ks=(args.candidates,))["metrics"]
        pipeline.extra["candidate_recall"] = ceiling[f"recall@{args.candidates}"]

    report = evaluate(questions, pipeline.retrieve)
    report = {
        "experiment": experiment,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config": pipeline.config,
        "chunks": len(chunks),
        **pipeline.extra,
        **report,
        "chance_fraction": {f"k={k}": chance_fraction(papers, questions, chunker, k) for k in KS},
        "answer_score": None,  # retrieval only: no generation, so no answer score or LLM cost
        "cost": None,
    }

    settings.results_dir.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    name = args.retriever
    if args.rerank:
        name += "-rerank-" + args.rerank.split("/")[-1]
    path = settings.results_dir / f"{stamp}-{name}-{args.split}-{args.chunker}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
