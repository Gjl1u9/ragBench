"""Answer QASPER questions with Claude from retrieved passages, and score the answers.

Run a pilot first:  uv run python -m ragbench.generate --context retrieved --k 10 --limit 5
Then the full run:  uv run python -m ragbench.generate --context retrieved --k 10

Claude is called through the Claude Agent SDK on the user's subscription. Calls are locked
down (no tools, no MCP servers, no user or project settings) so each one carries only our
prompt. Every response is cached under results/llm_cache/ keyed by model and prompt, so a
rerun of the same configuration makes no new calls.
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)

from ragbench.answers import score_answer
from ragbench.chunking import CHUNKERS, Chunk
from ragbench.config import settings
from ragbench.data import Paper, Question, load_split
from ragbench.evaluate import git_commit, score_question
from ragbench.index import DEFAULT_MODEL
from ragbench.pipeline import build_pipeline
from ragbench.rerank import DEFAULT_RERANKER

DEFAULT_LLM = "haiku"
SUBSET_SIZE = 100
SUBSET_SEED = 0

SYSTEM_PROMPT = """\
You answer a question about a scientific paper using only the numbered passages given.
Answer as briefly as possible:
- If the answer can be copied from the passages, copy the shortest exact phrase or phrases, \
separated by ", ".
- For a yes/no question, answer only "Yes" or "No".
- If the passages do not contain the answer, answer only "Unanswerable".
- Otherwise answer in one short sentence.
Output only the answer, with no explanation and no passage numbers."""
PROMPT_VERSION = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:8]


def dev_subset(questions: list[Question], n: int = SUBSET_SIZE, seed: int = SUBSET_SEED):
    """A fixed random sample of questions, so every generation run uses the same ones."""
    by_id = sorted(questions, key=lambda q: q.id)
    return sorted(random.Random(seed).sample(by_id, min(n, len(by_id))), key=lambda q: q.id)


def build_prompt(paper: Paper, question: Question, chunks: list[Chunk]) -> str:
    passages = "\n\n".join(f"[{i}] {chunk.text}" for i, chunk in enumerate(chunks, start=1))
    return f"Paper: {paper.title}\n\nPassages:\n{passages}\n\nQuestion: {question.text}"


class ClaudeAnswerer:
    """Calls Claude through the Agent SDK, with a response cache on disk.

    By default a call is one turn with no tools. Agentic runs pass their own tool server,
    allowed tools and turn limit, plus a `cache_extra` describing them so that runs with
    different tools never share cache entries.
    """

    def __init__(
        self,
        model: str = DEFAULT_LLM,
        cache_dir: Path | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        if os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is set: it would bill the API instead of the subscription"
            )
        self.model = model
        self.system_prompt = system_prompt
        self.cache_dir = cache_dir or settings.results_dir / "llm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls_made = 0

    def _options(self, **overrides) -> ClaudeAgentOptions:
        options = {
            "system_prompt": self.system_prompt,
            "model": self.model,
            "tools": [],  # no built-in tools (allowed_tools only auto-approves; it removes nothing)
            "setting_sources": [],  # no user/project settings, CLAUDE.md or plugins
            "strict_mcp_config": True,  # ignore the user's MCP servers
            "mcp_servers": {},
            "max_turns": 1,
            "cwd": tempfile.gettempdir(),  # away from this repo's CLAUDE.md
        }
        return ClaudeAgentOptions(**{**options, **overrides})

    def cache_key(self, prompt: str, cache_extra: dict | None = None) -> str:
        # Key order matters for the hash: keep this layout so existing cache entries stay valid.
        payload = {"model": self.model, "system": self.system_prompt, "prompt": prompt}
        if cache_extra:
            payload["extra"] = json.dumps(cache_extra, sort_keys=True)
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()

    async def answer(self, prompt: str, cache_extra: dict | None = None, **overrides) -> dict:
        path = self.cache_dir / f"{self.cache_key(prompt, cache_extra)}.json"
        if path.exists():
            return {**json.loads(path.read_text()), "cached": True}

        text: list[str] = []
        tool_calls: list[dict] = []
        turns = 0
        result: ResultMessage | None = None
        async for message in query(prompt=prompt, options=self._options(**overrides)):
            if isinstance(message, AssistantMessage):
                turns += 1
                text += [b.text for b in message.content if isinstance(b, TextBlock)]
                tool_calls += [
                    {"name": b.name, "input": b.input}
                    for b in message.content
                    if isinstance(b, ToolUseBlock)
                ]
            elif isinstance(message, ResultMessage):
                result = message
        if result is None or result.is_error:
            raise RuntimeError(f"Claude call failed: {result and result.result}")

        self.calls_made += 1
        usage = result.usage or {}
        # With tools, intermediate turns also produce text; the final answer is the result.
        final = result.result if result.result is not None else "".join(text)
        record = {
            "text": final.strip(),
            "tool_calls": tool_calls,
            "assistant_messages": turns,
            "model": next(iter(result.model_usage or {}), self.model),
            "input_tokens": usage.get("input_tokens", 0),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cost_estimate_usd": result.total_cost_usd,
            "duration_ms": result.duration_ms,
        }
        path.write_text(json.dumps(record, indent=2) + "\n")
        return {**record, "cached": False}


def summarize(rows: dict[str, dict]) -> dict:
    by_type: dict[str, list[float]] = defaultdict(list)
    for row in rows.values():
        by_type[row["answer_type"]].append(row["answer_f1"])
    durations = [row["duration_ms"] for row in rows.values()]
    recalls = [r["evidence_recall"] for r in rows.values() if r["evidence_recall"] is not None]
    return {
        "answer_f1": round(statistics.mean(r["answer_f1"] for r in rows.values()), 4),
        "answer_f1_by_type": {
            t: {"f1": round(statistics.mean(v), 4), "questions": len(v)}
            for t, v in sorted(by_type.items())
        },
        "evidence_recall": round(statistics.mean(recalls), 4) if recalls else None,
        "tokens": {
            key: sum(r[key] for r in rows.values())
            for key in (
                "input_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "output_tokens",
            )
        },
        "cost_estimate_usd": round(sum(r["cost_estimate_usd"] or 0 for r in rows.values()), 4),
        # SDK wall time per call; includes Claude Code start-up, not just the model.
        "latency_ms": {
            "median": round(statistics.median(durations)),
            "p95": round(statistics.quantiles(durations, n=20)[-1]) if len(durations) > 1 else None,
        },
    }


async def run(args) -> dict:
    papers, questions = load_split(args.split)
    papers_by_id = {p.id: p for p in papers}
    chunker = CHUNKERS[args.chunker]
    chunks_by_paper = {p.id: chunker(p) for p in papers}
    subset = dev_subset(questions)[: args.limit or None]

    config = {
        "context": args.context,
        "llm": args.llm,
        "prompt_version": PROMPT_VERSION,
        "subset": {"split": args.split, "size": SUBSET_SIZE, "seed": SUBSET_SEED},
        "questions_run": len(subset),
        "chunker": args.chunker,
    }
    if args.context == "retrieved":
        all_chunks = [c for cs in chunks_by_paper.values() for c in cs]
        pipeline = build_pipeline(
            args.split, args.chunker, all_chunks, "hybrid", DEFAULT_MODEL, args.rerank
        )
        config.update(k=args.k, retrieval=pipeline.config)

    answerer = ClaudeAnswerer(args.llm)
    rows: dict[str, dict] = {}
    for i, question in enumerate(subset, start=1):
        if args.context == "retrieved":
            hits = pipeline.retrieve(question, args.k)
            by_id = {c.id: c for c in chunks_by_paper[question.paper_id]}
            context = [by_id[h.chunk_id] for h in hits]
        else:
            context = chunks_by_paper[question.paper_id]
        prompt = build_prompt(papers_by_id[question.paper_id], question, context)
        response = await answerer.answer(prompt)
        retrieval = score_question(question, [c.paragraph_ids for c in context], ks=(len(context),))
        rows[question.id] = {
            "prediction": response["text"],
            **score_answer(question, response["text"]),
            "evidence_recall": retrieval[f"recall@{len(context)}"] if retrieval else None,
            "context_chunks": len(context),
            **{k: v for k, v in response.items() if k not in ("text",)},
        }
        print(
            f"{i}/{len(subset)} f1={rows[question.id]['answer_f1']:.2f} {response['text'][:60]!r}"
        )

    return {
        "experiment": "generate",
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config": config,
        "new_llm_calls": answerer.calls_made,
        **summarize(rows),
        "per_question": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--context", choices=["retrieved", "full"], default="retrieved")
    parser.add_argument("--k", type=int, default=10, help="passages to give Claude (retrieved)")
    parser.add_argument("--rerank", default=DEFAULT_RERANKER, help="reranker model")
    parser.add_argument("--chunker", default="paragraph", choices=sorted(CHUNKERS))
    parser.add_argument("--llm", default=DEFAULT_LLM)
    parser.add_argument("--limit", type=int, help="run only the first N subset questions")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument(
        "--final", action="store_true", help="required for the test split: final numbers only"
    )
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        parser.error("the test split is for final reported numbers; pass --final to confirm")

    report = anyio.run(run, args)
    settings.results_dir.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    context = f"k{args.k}" if args.context == "retrieved" else "full"
    split = "" if args.split == "validation" else f"{args.split}-"
    path = (
        settings.results_dir
        / f"{stamp}-generate-{split}{context}-{report['config']['questions_run']}q.json"
    )
    path.write_text(json.dumps(report, indent=2) + "\n")
    summary = {k: v for k, v in report.items() if k != "per_question"}
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
