import pytest

from ragbench.answers import normalize_answer, score_answer, token_f1
from ragbench.data import Answer, Question


def test_normalize_drops_case_punctuation_and_articles():
    assert normalize_answer("The  BERT-base model, a baseline.") == "bertbase model baseline"


def test_token_f1_worked_example():
    # 4 shared tokens; precision 4/4, recall 4/6 -> F1 0.8
    gold = "a vocabulary of positive and negative predicates"
    assert token_f1("Positive and negative predicates.", gold) == pytest.approx(0.8)


def test_token_f1_is_zero_without_overlap_and_one_for_exact_match():
    assert token_f1("six layers", "two layers") == pytest.approx(0.5)
    assert token_f1("Yes", "No") == 0.0
    assert token_f1("Yes.", "yes") == 1.0


def question(*answers):
    return Question("q", "paper", "?", list(answers))


def test_score_takes_the_best_annotator_and_reports_its_type():
    q = question(
        Answer("abstractive", "it uses two layers of attention", [], []),
        Answer("extractive", "two layers", ["two layers"], []),
    )
    assert score_answer(q, "two layers") == {"answer_f1": 1.0, "answer_type": "extractive"}


def test_unanswerable_gold_is_the_word_unanswerable():
    q = question(Answer("unanswerable", "", [], []))
    assert score_answer(q, "Unanswerable")["answer_f1"] == 1.0
    assert score_answer(q, "The paper does not say.")["answer_f1"] == 0.0
