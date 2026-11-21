# ragBench

An evaluated RAG system over scientific papers, built on the
[QASPER](https://huggingface.co/datasets/allenai/qasper) dataset (questions about NLP papers,
with gold answers and evidence paragraphs).

The goal is not just to answer questions, but to measure what each technique buys. Every
difference below is checked with a paired bootstrap 95% confidence interval.
A difference is considered statistically significant when the interval excludes 0.

## Final results (test split)

All design choices were made on the validation split. The chosen system (hybrid retrieval +
MiniLM reranker, top 20 passages to Claude Haiku) was then run **once** on the held-out test
split; nothing was tuned afterwards.

| Retrieval (1,351 scorable questions) | Recall@5 | Recall@10 | MRR | Latency (median) |
|---|---|---|---|---|
| Dense baseline | 0.618 | 0.762 | 0.519 | 13 ms |
| Hybrid (dense + BM25, RRF) | 0.613 | 0.762 | 0.500 | 16 ms |
| **Hybrid + MiniLM reranker** | **0.696** | **0.824** | **0.575** | 168 ms |

| Generation (100 test questions, seed 0) | Answer F1 | Evidence in context | Cost est. / q | Latency (median) |
|---|---|---|---|---|
| Top 20 reranked passages, Claude Haiku | 0.639 (CI 0.572–0.708) | 0.896 | $0.0010 | 1.7 s |

- The reranker's gain holds on unseen data: Recall@5 +0.079 over the dense baseline
  (CI +0.055 to +0.102), MRR +0.056 (CI +0.035 to +0.076).
- Hybrid fusion again adds nothing on its own; on test its MRR is slightly below dense
  (−0.020, CI −0.038 to −0.002). It is kept only as the candidate source for the reranker.
- Test Answer F1 (0.639) is higher than validation's (0.547, CI 0.467–0.628), but the two are
  not directly comparable: test has more annotators per question (2.45 vs 1.76), and the score
  takes the best annotator, so more annotators raise it. The test sample also has more yes/no
  questions (17 vs 12), which Claude answered perfectly.

## Development results (validation split)

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
| Agentic: Claude searches, ≤ 4 × 5 passages | 0.492 | 0.804 | 5,900¹ | $0.0007 | 2.3 s |

¹ About half of it is earlier turns re-read from the prompt cache, which is why it costs less
than top 20 despite more tokens.

- Top 5 is significantly worse than top 10 (−0.058, CI −0.105 to −0.014). Top 20 and the whole
  paper are not significantly better than top 10 on 100 questions (+0.028 and +0.032, CIs
  include 0).
- Top 20 matches the whole paper at about half the tokens.
- The main effect of less context is abstention: Claude answered "Unanswerable" to an
  answerable question 17 times with top 5, 11 with top 10, 8 with top 20, and 5 with the whole
  paper. It answered 4 of the 8 truly unanswerable questions in every setting.
- Token F1 is strict: "six experts" vs gold "five annotators", or "0.7 BLEU" vs "On average
  0.64", score 0. The LLM-judge check below measures how much this matters.
- Cost is the SDK's estimate (the subscription is not billed per call). Latency is dominated
  by Claude Code start-up, so it barely changes with context size.

### Agentic RAG

Claude gets one tool, `search`, over the question's paper (the same hybrid + MiniLM pipeline)
and writes its own queries: up to 4 searches of 5 passages, so at most the 20 passages of the
fixed top-20 setup. It compares to top 20 on the same 100 questions:

- Answer F1 is lower (−0.055, CI −0.116 to +0.002), just short of significant.
- It stops early: 1.9 searches on average (1 search on 32 questions, 4 on only 4), so it sees
  7.3 distinct passages instead of 20, and the evidence is in what it saw 80% of the time
  (90% for top 20).
- Its answers are wordier (median 13 words vs 4), which token F1 penalises. Judged for
  meaning on the same 50 questions, the two are level (0.700 vs 0.670, +0.030, CI −0.07 to
  +0.14).
- It is slower (2.3 s vs 1.8 s median) because it takes 2–5 turns.

So on QASPER, where the right paper is known and a short question usually needs one lookup,
letting the model search buys nothing over a well-ranked fixed context. Retrieval quality
(the reranker) mattered far more than retrieval strategy.

### LLM judge

Claude Sonnet judged a fixed random sample of 50 of the 100 answers (seed 0): it reads the
question, every annotator's reference and the candidate, and returns correct / partial /
incorrect (scored 1 / 0.5 / 0). It does not see the paper, so it checks agreement with the
references. `python -m ragbench.judge results/<run>.json --sample 50`.

| Answers judged (same 50 questions) | Answer F1 | Judge score | Correct / partial / incorrect |
|---|---|---|---|
| Top 20 passages | 0.519 | 0.670 | 31 / 5 / 14 |
| Agentic | 0.468 | 0.700 | 32 / 6 / 12 |

- F1 understates answers that are right in other words: for top 20, 7 of the 11 answers with
  0 < F1 < 0.5 were judged correct ("SBERT can be tuned in less than 20 minutes" vs gold
  "20 minutes", F1 0.36). For the agentic run, 11 of its 27 answers below F1 0.5 were.
- F1 did not overstate: every answer with F1 = 0 was judged incorrect, and no answer with
  F1 ≥ 0.5 was.
- Read Answer F1 as a lower bound that is harsher on verbose systems. It is still the right
  metric for comparing runs with the same answer style, and it costs nothing to compute.
- Judging 50 answers costs about $0.21 (SDK estimate) with Sonnet.

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

## Compare evaluation runs

After setup, run this from the repository root to compare the dense baseline with the
MiniLM reranker on the test split:

```bash
.venv/bin/python -m ragbench.compare \
  results/20261119T033633-dense-test-paragraph.json \
  results/20261119T034151-hybrid-rerank-ms-marco-MiniLM-L-6-v2-test-paragraph.json
```

Evaluation outputs are stored locally in `results/`. If these files are absent, use two
saved runs that scored the same questions. Pass the baseline first and the candidate second.

## Roadmap

- [x] Project setup: environment, Qdrant, tests
- [x] Data loading and chunking (99.4% of train evidence resolved to paragraph ids)
- [x] Dense retrieval baseline
- [x] Evaluation harness: retrieval metrics
- [x] Evaluation harness: answer metrics (official QASPER token F1)
- [x] Hybrid retrieval (BM25 + dense, RRF)
- [x] Cross-encoder reranking
- [x] Generation with Claude on a fixed 100-question dev subset
- [x] Final evaluation on the held-out test split (run once)
- [x] LLM-judge check of answer F1 on a sample
- [x] Agentic RAG and comparison

## License

MIT
