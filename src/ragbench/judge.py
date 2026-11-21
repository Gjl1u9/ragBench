"""Check Answer F1 against an LLM judge on a sample of answers.

Run with: uv run python -m ragbench.judge results/<generate run>.json --sample 50

Token F1 rewards shared words, so a correct answer in other words scores low and a wrong
answer that reuses the question's words can score high. The judge reads the question, every
annotator's reference answer and the candidate, and decides whether the meaning matches. It
does not see the paper, so it checks agreement with the references, not truth. Judging is
kept to a sample, as CLAUDE.md asks, and every verdict is cached like any other LLM call.
"""

import argparse
import json
import random
import re
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import anyio

from ragbench.answers import gold_answer
from ragbench.config import settings
from ragbench.data import Question, load_split
from ragbench.evaluate import git_commit
from ragbench.generate import ClaudeAnswerer

DEFAULT_JUDGE = "sonnet"
SAMPLE_SEED = 0
SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

JUDGE_PROMPT = """\
You grade answers to questions about scientific papers. You get a question, reference answers \
written by different annotators, and a candidate answer. Decide whether the candidate means \
the same as at least one reference. Judge meaning only: ignore wording, length and format.
A reference of "Unanswerable" means the paper does not answer the question; a candidate \
matches it only if it also says the question cannot be answered.
Verdicts:
- "correct": same meaning as a reference.
- "partial": right direction, but misses or misstates a key part of every reference.
- "incorrect": anything else.
Reply with one line of JSON and nothing else: \
{"verdict": "correct" | "partial" | "incorrect", "reason": "<at most 15 words>"}"""


def judge_prompt(question: Question, candidate: str) -> str:
    references = "\n".join(f"- {gold_answer(a)}" for a in question.answers)
    return (
        f"Question: {question.text}\n\nReference answers:\n{references}\n\n"
        f"Candidate answer: {candidate}"
    )


def parse_verdict(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"judge reply has no JSON: {text!r}")
    verdict = json.loads(match.group(0))
    if verdict.get("verdict") not in SCORES:
        raise ValueError(f"unknown verdict in {text!r}")
    return verdict


def bucket(f1: float) -> str:
    """F1 range, used to show where F1 and the judge disagree."""
    if f1 == 0:
        return "f1 = 0"
    if f1 < 0.5:
        return "0 < f1 < 0.5"
    if f1 < 1:
        return "0.5 <= f1 < 1"
    return "f1 = 1"


def summarize(rows: dict[str, dict]) -> dict:
    by_bucket: dict[str, Counter] = {}
    for row in rows.values():
        by_bucket.setdefault(bucket(row["answer_f1"]), Counter())[row["verdict"]] += 1
    low_f1 = [r for r in rows.values() if r["answer_f1"] < 0.5]
    high_f1 = [r for r in rows.values() if r["answer_f1"] >= 0.5]
    return {
        "questions": len(rows),
        "answer_f1": round(statistics.mean(r["answer_f1"] for r in rows.values()), 4),
        # correct = 1, partial = 0.5, incorrect = 0
        "judge_score": round(statistics.mean(r["judge_score"] for r in rows.values()), 4),
        "verdicts": dict(Counter(r["verdict"] for r in rows.values())),
        "verdicts_by_f1": {b: dict(c) for b, c in sorted(by_bucket.items())},
        # F1 < 0.5 that the judge calls correct: answers F1 undervalues.
        "low_f1_judged_correct": sum(r["verdict"] == "correct" for r in low_f1),
        "low_f1_total": len(low_f1),
        # F1 >= 0.5 that the judge calls incorrect: answers F1 overvalues.
        "high_f1_judged_incorrect": sum(r["verdict"] == "incorrect" for r in high_f1),
        "high_f1_total": len(high_f1),
    }


def sample_ids(ids, n: int, seed: int = SAMPLE_SEED) -> list[str]:
    return sorted(random.Random(seed).sample(sorted(ids), min(n, len(ids))))


async def run(args) -> dict:
    source = json.loads(args.run.read_text())
    split = source["config"]["subset"]["split"]
    _, questions = load_split(split)
    by_id = {q.id: q for q in questions}
    chosen = sample_ids(source["per_question"], args.sample)

    judge = ClaudeAnswerer(args.model, system_prompt=JUDGE_PROMPT)
    rows = {}
    for qid in chosen:
        answered = source["per_question"][qid]
        response = await judge.answer(judge_prompt(by_id[qid], answered["prediction"]))
        verdict = parse_verdict(response["text"])
        rows[qid] = {
            "prediction": answered["prediction"],
            "answer_f1": answered["answer_f1"],
            "verdict": verdict["verdict"],
            "judge_score": SCORES[verdict["verdict"]],
            "reason": verdict.get("reason", ""),
            "cost_estimate_usd": response["cost_estimate_usd"],
        }

    return {
        "experiment": "judge",
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config": {
            "source_run": args.run.name,
            "source_config": source["config"],
            "judge_model": args.model,
            "sample": len(chosen),
            "sample_seed": SAMPLE_SEED,
        },
        "new_llm_calls": judge.calls_made,
        "cost_estimate_usd": round(sum(r["cost_estimate_usd"] or 0 for r in rows.values()), 4),
        **summarize(rows),
        "per_question": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="a results/*-generate-*.json file")
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--model", default=DEFAULT_JUDGE)
    args = parser.parse_args()

    report = anyio.run(run, args)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = settings.results_dir / f"{stamp}-judge-{args.run.stem}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_question"}, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
