"""Agentic RAG: Claude searches the paper itself before answering.

Run a pilot first:  uv run python -m ragbench.agentic --limit 5
Then the full run:  uv run python -m ragbench.agentic

Instead of receiving fixed passages, Claude gets one tool, `search`, over the question's paper
(the same hybrid + reranker pipeline as the fixed setup). It chooses its own queries and may
search up to MAX_SEARCHES times, PASSAGES_PER_SEARCH passages each, so it sees at most as many
passages as the fixed top-20 setup: the comparison is at an equal passage budget.

Retrieval is deterministic, so the passages Claude saw are rebuilt from the logged queries;
that keeps the evidence numbers correct for cached runs too.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime

import anyio
from claude_agent_sdk import create_sdk_mcp_server, tool

from ragbench.answers import score_answer
from ragbench.chunking import CHUNKERS, Chunk
from ragbench.config import settings
from ragbench.data import Question, load_split
from ragbench.evaluate import git_commit, score_question
from ragbench.generate import DEFAULT_LLM, SUBSET_SEED, SUBSET_SIZE, ClaudeAnswerer, dev_subset
from ragbench.generate import summarize as summarize_generation
from ragbench.index import DEFAULT_MODEL
from ragbench.pipeline import Pipeline, build_pipeline
from ragbench.rerank import DEFAULT_RERANKER

MAX_SEARCHES = 4
PASSAGES_PER_SEARCH = 5
TOOL_NAME = "mcp__paper__search"

AGENT_PROMPT = f"""\
You answer a question about one scientific paper. You cannot see the paper: use the search \
tool, which returns numbered passages from it. Search with short, specific queries, and search \
again with different wording if the passages do not answer the question. You have at most \
{MAX_SEARCHES} searches.
When you are done, reply with only the final answer, as briefly as possible:
- If the answer can be copied from the passages, copy the shortest exact phrase or phrases, \
separated by ", ".
- For a yes/no question, answer only "Yes" or "No".
- If the passages do not contain the answer, answer only "Unanswerable".
- Otherwise answer in one short sentence.
No explanation, no passage numbers."""
AGENT_VERSION = hashlib.sha256(AGENT_PROMPT.encode()).hexdigest()[:8]
LIMIT_REACHED = "Search limit reached. Answer now with what you have."


class PaperSearch:
    """Runs searches for one question, scoped to its paper, under the search budget."""

    def __init__(self, pipeline: Pipeline, question: Question, chunks: list[Chunk]):
        self.pipeline = pipeline
        self.question = question
        self.by_id = {c.id: c for c in chunks}
        self.seen: dict[str, Chunk] = {}  # every passage returned, in first-seen order
        self.searches = 0

    def search(self, query: str) -> str:
        if self.searches >= MAX_SEARCHES:
            return LIMIT_REACHED
        self.searches += 1
        hits = self.pipeline.retrieve(replace(self.question, text=query), PASSAGES_PER_SEARCH)
        lines = []
        for hit in hits:
            chunk = self.by_id[hit.chunk_id]
            self.seen.setdefault(chunk.id, chunk)
            number = list(self.seen).index(chunk.id) + 1
            lines.append(f"[{number}] {chunk.text}")
        return "\n\n".join(lines) or "No passages found."

    def replay(self, tool_calls: list[dict]) -> None:
        """Rebuild `seen` and `searches` from logged tool calls (for cached answers)."""
        for call in tool_calls:
            if call["name"] == TOOL_NAME:
                self.search(call["input"].get("query", ""))


def search_server(paper_search: PaperSearch):
    @tool("search", "Search the paper. Returns the most relevant passages.", {"query": str})
    async def search(args):
        return {"content": [{"type": "text", "text": paper_search.search(args["query"])}]}

    return create_sdk_mcp_server("paper", tools=[search])


async def run(args) -> dict:
    papers, questions = load_split(args.split)
    papers_by_id = {p.id: p for p in papers}
    chunker = CHUNKERS[args.chunker]
    chunks_by_paper = {p.id: chunker(p) for p in papers}
    subset = dev_subset(questions)[: args.limit or None]
    all_chunks = [c for cs in chunks_by_paper.values() for c in cs]
    pipeline = build_pipeline(
        args.split, args.chunker, all_chunks, "hybrid", DEFAULT_MODEL, args.rerank
    )

    config = {
        "context": "agentic",
        "llm": args.llm,
        "prompt_version": AGENT_VERSION,
        "max_searches": MAX_SEARCHES,
        "passages_per_search": PASSAGES_PER_SEARCH,
        "subset": {"split": args.split, "size": SUBSET_SIZE, "seed": SUBSET_SEED},
        "questions_run": len(subset),
        "chunker": args.chunker,
        "retrieval": pipeline.config,
    }
    # Everything that changes what Claude can do is part of the cache key.
    cache_extra = {
        k: config[k] for k in ("max_searches", "passages_per_search", "retrieval", "chunker")
    }
    answerer = ClaudeAnswerer(args.llm, system_prompt=AGENT_PROMPT)
    rows: dict[str, dict] = {}
    for i, question in enumerate(subset, start=1):
        paper = papers_by_id[question.paper_id]
        live = PaperSearch(pipeline, question, chunks_by_paper[paper.id])
        response = await answerer.answer(
            f"Paper: {paper.title}\n\nQuestion: {question.text}",
            cache_extra=cache_extra,
            mcp_servers={"paper": search_server(live)},
            allowed_tools=[TOOL_NAME],
            # one turn per search, one to read the last result, one to answer
            max_turns=MAX_SEARCHES + 2,
        )
        # Same passages whether the answer was fresh or cached.
        seen = PaperSearch(pipeline, question, chunks_by_paper[paper.id])
        seen.replay(response["tool_calls"])
        context = list(seen.seen.values())
        retrieval = score_question(question, [c.paragraph_ids for c in context], ks=(len(context),))
        rows[question.id] = {
            "prediction": response["text"],
            **score_answer(question, response["text"]),
            "evidence_recall": (
                retrieval[f"recall@{len(context)}"] if retrieval and context else None
            ),
            "context_chunks": len(context),
            "searches": seen.searches,
            "queries": [c["input"].get("query") for c in response["tool_calls"]],
            **{k: v for k, v in response.items() if k not in ("text", "tool_calls")},
        }
        row = rows[question.id]
        print(
            f"{i}/{len(subset)} f1={row['answer_f1']:.2f} searches={row['searches']} "
            f"{response['text'][:50]!r}"
        )

    summary = summarize_generation(rows)
    summary["searches_mean"] = round(sum(r["searches"] for r in rows.values()) / len(rows), 2)
    summary["passages_seen_mean"] = round(
        sum(r["context_chunks"] for r in rows.values()) / len(rows), 2
    )
    return {
        "experiment": "agentic",
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config": config,
        "new_llm_calls": answerer.calls_made,
        **summary,
        "per_question": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    split = "" if args.split == "validation" else f"{args.split}-"
    path = (
        settings.results_dir
        / f"{stamp}-generate-{split}agentic-{report['config']['questions_run']}q.json"
    )
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_question"}, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
