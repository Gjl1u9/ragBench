import json

import anyio
import pytest

from ragbench.chunking import paragraph_chunks
from ragbench.data import parse_paper
from ragbench.generate import ClaudeAnswerer, build_prompt, dev_subset, summarize


def test_dev_subset_is_fixed_and_independent_of_input_order(raw_paper):
    _, questions = parse_paper(raw_paper)
    many = [q for _ in range(60) for q in questions]  # duplicates are fine for ordering checks
    first = dev_subset(many, n=5, seed=0)
    again = dev_subset(list(reversed(many)), n=5, seed=0)
    assert [q.id for q in first] == [q.id for q in again]
    assert len(first) == 5


def test_build_prompt_numbers_passages_and_ends_with_the_question(raw_paper):
    paper, questions = parse_paper(raw_paper)
    prompt = build_prompt(paper, questions[0], paragraph_chunks(paper)[:2])
    assert prompt.startswith("Paper: A Tiny Paper\n\nPassages:\n[1] Introduction\n")
    assert "\n\n[2] " in prompt
    assert prompt.endswith("Question: How many layers does TinyNet have?")


def test_test_split_requires_final_flag(monkeypatch):
    from ragbench import generate

    monkeypatch.setattr("sys.argv", ["generate", "--split", "test"])
    monkeypatch.setattr(generate.anyio, "run", lambda *a: pytest.fail("must not run"))
    with pytest.raises(SystemExit):
        generate.main()


def test_answerer_refuses_to_run_with_an_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with pytest.raises(RuntimeError):
        ClaudeAnswerer(cache_dir=tmp_path)


def test_answerer_serves_cached_responses_without_calling_claude(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    answerer = ClaudeAnswerer(model="haiku", cache_dir=tmp_path)
    stored = {"text": "two layers", "duration_ms": 900, "cost_estimate_usd": 0.001}
    (tmp_path / f"{answerer.cache_key('prompt')}.json").write_text(json.dumps(stored))

    def fail(*args, **kwargs):
        raise AssertionError("Claude must not be called on a cache hit")

    monkeypatch.setattr("ragbench.generate.query", fail)
    response = anyio.run(answerer.answer, "prompt")
    assert response == {**stored, "cached": True}
    assert answerer.calls_made == 0


def test_cache_key_depends_on_model_and_prompt(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    haiku = ClaudeAnswerer(model="haiku", cache_dir=tmp_path)
    sonnet = ClaudeAnswerer(model="sonnet", cache_dir=tmp_path)
    assert haiku.cache_key("p") != sonnet.cache_key("p")
    assert haiku.cache_key("p") != haiku.cache_key("q")


def row(f1, answer_type, recall=1.0):
    return {
        "answer_f1": f1,
        "answer_type": answer_type,
        "evidence_recall": recall,
        "duration_ms": 1000,
        "cost_estimate_usd": 0.001,
        "input_tokens": 10,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 2,
    }


def test_summarize_breaks_scores_down_by_answer_type():
    summary = summarize({"a": row(1.0, "yes_no"), "b": row(0.0, "yes_no"), "c": row(0.5, "x")})
    assert summary["answer_f1"] == 0.5
    assert summary["answer_f1_by_type"]["yes_no"] == {"f1": 0.5, "questions": 2}
    assert summary["tokens"]["output_tokens"] == 6
    assert summary["cost_estimate_usd"] == 0.003
