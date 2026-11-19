# ragBench

An evaluated RAG system over scientific papers, built on the
[QASPER](https://huggingface.co/datasets/allenai/qasper) dataset (questions about NLP papers,
with gold answers and evidence paragraphs).

The goal is not just to answer questions, but to measure what each technique buys. Every
difference below is checked with a paired bootstrap 95% confidence interval
(`python -m ragbench.compare`); "not significant" means the interval includes 0.

### Retrieval

Validation split, 923 scorable questions, retrieving within each question's paper over
paragraph chunks. No LLM calls.

| Setup | Recall@5 | Recall@10 | MRR | Latency (median) |
|---|---|---|---|---|
| Dense (`BAAI/bge-small-en-v1.5`) | 0.573 | 0.737 | 0.463 | 14 ms |
| BM25 only | 0.456 | 0.642 | 0.350 | 0.05 ms |
| Hybrid (dense + BM25, RRF) | 0.573 | 0.749 | 0.464 | 18 ms |
| Hybrid + MiniLM reranker | **0.642** | **0.796** | **0.522** | 147 ms |
| Hybrid + bge-reranker-base | 0.598 | 0.759 | 0.472 | 990 ms |

- Random choice of 5 paragraphs would cover about 11.7% of a paper.
- RRF fusion does not beat dense alone (Recall@5 +0.0001, CI −0.025 to +0.024), although the
  two are complementary: the better of the two per question would reach Recall@5 0.678.
- The MiniLM cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), reranking the hybrid
  top 50, is a real gain over hybrid: Recall@5 +0.069 (CI +0.042 to +0.095), MRR +0.058
  (CI +0.035 to +0.081). The top 50 hold 98.7% of the evidence, its ceiling.
- The larger `BAAI/bge-reranker-base` is significantly worse than MiniLM (Recall@5 −0.044,
  CI −0.068 to −0.022), not significantly better than no reranker, and 6.7× slower.

### Generation

A fixed random subset of 100 validation questions (seed 0). Passages come from hybrid
retrieval + MiniLM reranking; Claude Haiku answers through the Agent SDK with no tools and no
project settings. Answer F1 is QASPER's official token F1, best over annotators.

| Context given to Claude | Answer F1 | Evidence in context | Input tokens / q | Cost est. / q | Latency (median) |
|---|---|---|---|---|---|
| Top 5 passages | 0.462 | 0.679 | 1,900 | $0.0004 | 1.6 s |
| Top 10 passages | 0.519 | 0.832 | 2,900 | $0.0006 | 1.8 s |
| Top 20 passages | 0.547 | 0.896 | 4,800 | $0.0010 | 1.8 s |
| Whole paper (no retrieval) | 0.551 | 1.000 | 9,100 | $0.0019 | 1.8 s |

- Top 5 is significantly worse than top 10 (−0.058, CI −0.105 to −0.014). Top 20 and the whole
  paper are not significantly better than top 10 on 100 questions (+0.028 and +0.032, CIs
  include 0).
- Top 20 matches the whole paper at about half the tokens.
- The main effect of less context is abstention: Claude answered "Unanswerable" to an
  answerable question 17 times with top 5, 11 with top 10, 8 with top 20, and 5 with the whole
  paper. It answered 4 of the 8 truly unanswerable questions in every setting.
- Token F1 is strict: "six experts" vs gold "five annotators", or "0.7 BLEU" vs "On average
  0.64", score 0. An LLM-judge check on a sample is still to do.
- Cost is the SDK's estimate (the subscription is not billed per call). Latency is dominated
  by Claude Code start-up, so it barely changes with context size.

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
- [x] Evaluation harness: answer metrics (official QASPER token F1)
- [x] Hybrid retrieval (BM25 + dense, RRF)
- [x] Cross-encoder reranking
- [x] Generation with Claude on a fixed 100-question dev subset
- [ ] LLM-judge check of answer F1 on a sample
- [ ] Agentic RAG and comparison

## License

MIT
