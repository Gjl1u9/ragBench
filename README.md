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

Python 3.12 · uv · Qdrant · BM25 · sentence-transformers · Claude API · pytest

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync
cp .env.example .env          # add your ANTHROPIC_API_KEY
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
