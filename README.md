# ragBench

An evaluated RAG system over scientific papers, built on the
[QASPER](https://huggingface.co/datasets/allenai/qasper) dataset (questions about NLP papers,
with gold answers and evidence paragraphs).

The goal is not just to answer questions, but to measure what each technique buys:

| Setup | Recall@5 | MRR | Answer F1 | Cost / query | Latency |
|---|---|---|---|---|---|
| Dense retrieval (baseline) | 0.573 | 0.463 | – | – | 14 ms |
| BM25 only | 0.456 | 0.350 | – | – | 0.05 ms |
| + BM25 hybrid (RRF, equal weights) | 0.573 | 0.464 | – | – | 17 ms |
| + Cross-encoder reranker | – | – | – | – | – |
| + Query rewriting | – | – | – | – | – |
| Agentic RAG | – | – | – | – | – |

Numbers are on the validation split (923 scorable questions), retrieving within each question's
paper, with `BAAI/bge-small-en-v1.5` over paragraph chunks. Retrieving 5 paragraphs at random
would cover about 11.7% of a paper. Latency is the median per query, including query embedding.

Equal-weight RRF fusion does not beat dense retrieval: the Recall@5 difference is +0.0001
(95% bootstrap CI −0.025 to +0.024). The two retrievers are complementary, though: BM25 wins
on 132 questions, and picking the better of the two per question would reach Recall@5 0.678.

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
- [x] Data loading and chunking (99.4% of train evidence resolved to paragraph ids)
- [x] Dense retrieval baseline
- [x] Evaluation harness: retrieval metrics
- [ ] Evaluation harness: answer metrics
- [x] Hybrid retrieval (BM25 + dense, RRF)
- [ ] Cross-encoder reranking
- [ ] Agentic RAG and comparison

## License

MIT
