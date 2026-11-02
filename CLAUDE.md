# ragBench

An evaluated RAG system over scientific papers (QASPER). The point of the project is
measured comparison: every retrieval or generation change is judged by the eval, not by demos.

## Stack

- Python 3.12, managed with `uv` (never `pip install`; use `uv add` / `uv add --dev`)
- Qdrant 1.20.0 in Docker (`docker-compose.yml`), BM25 via `rank-bm25`
- Embeddings and cross-encoder rerankers via `sentence-transformers`
- Generation via the Claude API (`anthropic`)
- Data via Hugging Face `datasets` (`allenai/qasper`)

## Commands

```bash
docker compose up -d          # start Qdrant (Docker runs on OrbStack)
uv run pytest                 # all tests; integration tests need Qdrant
uv run pytest -m "not integration"
uv run ruff check . && uv run ruff format .
```

## Layout

- `src/ragbench/` — library code; `config.py` reads settings from env / `.env`
- `tests/` — pytest; mark tests that need Qdrant with `@pytest.mark.integration`
- `data/` — downloaded datasets (gitignored)
- `results/` — eval outputs (gitignored except `.gitkeep`)

## Rules

- Evaluation discipline: tune only on the QASPER train/dev splits. Run the test split only
  for final reported numbers, and say so when you do.
- Every experiment records its config, Recall@k, answer score, cost and latency in `results/`.
- Secrets live in `.env` (gitignored). Never commit keys or print them.
- Tooling: use only Claude Code built-in skills and plugins from the Claude marketplace
  (Anthropic Directory). Do not download external skills or plugins. Hugging Face
  tooling is allowed for datasets and models.
