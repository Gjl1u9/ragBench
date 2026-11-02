# ragBench

An evaluated RAG system over scientific papers, built on the
[QASPER](https://huggingface.co/datasets/allenai/qasper) dataset (questions about NLP papers,
with gold answers and evidence paragraphs).

The goal is not just to answer questions, but to measure what each technique buys:

| Setup | Recall@5 | MRR | Answer F1 | Cost / query | Latency |
|---|---|---|---|---|---|
| Dense retrieval (baseline) | – | – | – | – | – |
| + BM25 hybrid | – | – | – | – | – |
| + Cross-encoder reranker | – | – | – | – | – |
| + Query rewriting | – | – | – | – | – |
| Agentic RAG | – | – | – | – | – |

## Stack

Python 3.12 · uv · Qdrant · BM25 · sentence-transformers · Claude Agent SDK · pytest

Generation runs through the [Claude Agent SDK](https://docs.claude.com/en/docs/agent-sdk/overview),
authenticated with a Claude subscription rather than an API key. Retrieval metrics need no LLM
at all; generation is evaluated on a fixed dev subset with cached responses.

## Setup

Requires [uv](https://docs.astral.sh/uv/), Docker, and Claude Code logged in to a Claude
subscription.

```bash
uv sync
cp .env.example .env
docker compose up -d          # Qdrant on localhost:6333
uv run pytest
```

## Roadmap

- [x] Project setup: environment, Qdrant, tests
- [ ] Data loading and chunking
- [ ] Dense retrieval baseline
- [ ] Evaluation harness (retrieval + answer metrics)
- [ ] Hybrid retrieval (BM25 + dense)
- [ ] Cross-encoder reranking
- [ ] Agentic RAG and comparison

## License

MIT
