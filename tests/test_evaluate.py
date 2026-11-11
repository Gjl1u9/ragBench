import pytest

from ragbench.chunking import paragraph_chunks
from ragbench.data import Answer, Question, parse_paper
from ragbench.evaluate import (
    chance_fraction,
    evaluate,
    gold_sets,
    recall_at_k,
    reciprocal_rank,
    score_question,
)
from ragbench.retrieve import Hit


def answer(evidence, type_="extractive"):
    return Answer(type_, "x", [], evidence)


def question(*answers):
    return Question("q", "paper", "?", list(answers))


def hit(*paragraph_ids):
    return Hit("c", 1.0, "text", paragraph_ids)


def test_recall_at_k_counts_gold_found_within_k():
    ranked = [("a",), ("b",), ("c",)]
    assert recall_at_k({"a", "c"}, ranked, 1) == 0.5
    assert recall_at_k({"a", "c"}, ranked, 3) == 1.0
    assert recall_at_k({"z"}, ranked, 3) == 0.0


def test_recall_counts_every_paragraph_a_chunk_covers():
    assert recall_at_k({"a", "b"}, [("a", "b")], 1) == 1.0


def test_reciprocal_rank_uses_the_first_hit():
    assert reciprocal_rank({"b", "c"}, [("a",), ("b",), ("c",)]) == 0.5
    assert reciprocal_rank({"z"}, [("a",)]) == 0.0


def test_gold_sets_skip_unanswerable_and_empty_evidence():
    q = question(answer(["a"]), answer([]), answer(["b"], "unanswerable"))
    assert gold_sets(q) == [{"a"}]


def test_score_takes_the_best_annotator():
    q = question(answer(["a", "z"]), answer(["b"]))
    scores = score_question(q, [("b",), ("a",)], ks=(1, 2))
    assert scores["recall@1"] == 1.0  # second annotator's only paragraph is ranked first
    assert scores["recall@2"] == 1.0
    assert scores["mrr"] == 1.0


def test_question_without_gold_is_not_scored():
    assert score_question(question(answer([], "unanswerable")), [("a",)]) is None


def test_evaluate_averages_scored_questions_only():
    good = Question("q1", "paper", "?", [answer(["a"])])
    miss = Question("q2", "paper", "?", [answer(["z"])])
    skipped = Question("q3", "paper", "?", [answer([], "unanswerable")])
    report = evaluate([good, miss, skipped], lambda q, k: [hit("a")], ks=(1,))
    assert report["questions_total"] == 3
    assert report["questions_scored"] == 2
    assert report["metrics"] == {"recall@1": 0.5, "mrr": 0.5}
    assert set(report["latency_ms"]) == {"median", "p95"}


def test_chance_fraction(raw_paper):
    paper, questions = parse_paper(raw_paper)
    assert len(paper.paragraphs) == 5
    assert chance_fraction([paper], questions, paragraph_chunks, 1) == 0.2
    assert chance_fraction([paper], questions, paragraph_chunks, 10) == 1.0


def test_test_split_requires_final_flag(monkeypatch):
    from ragbench import evaluate as module

    monkeypatch.setattr("sys.argv", ["evaluate", "--split", "test"])
    with pytest.raises(SystemExit):
        module.main()
