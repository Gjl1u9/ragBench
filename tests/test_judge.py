import pytest

from ragbench.data import Answer, Question
from ragbench.judge import bucket, judge_prompt, parse_verdict, sample_ids, summarize


def test_judge_prompt_lists_every_reference_and_unanswerable():
    q = Question(
        "q",
        "paper",
        "How many layers?",
        [
            Answer("extractive", "two layers", ["two layers"], []),
            Answer("unanswerable", "", [], []),
        ],
    )
    prompt = judge_prompt(q, "2")
    assert "Question: How many layers?" in prompt
    assert "- two layers\n- Unanswerable" in prompt
    assert prompt.endswith("Candidate answer: 2")


def test_parse_verdict_accepts_json_with_surrounding_text():
    text = 'Here: {"verdict": "partial", "reason": "misses one dataset"}'
    assert parse_verdict(text) == {"verdict": "partial", "reason": "misses one dataset"}


@pytest.mark.parametrize("text", ["correct", '{"verdict": "maybe"}'])
def test_parse_verdict_rejects_bad_replies(text):
    with pytest.raises(ValueError):
        parse_verdict(text)


@pytest.mark.parametrize(
    "f1, name",
    [(0.0, "f1 = 0"), (0.2, "0 < f1 < 0.5"), (0.5, "0.5 <= f1 < 1"), (1.0, "f1 = 1")],
)
def test_bucket(f1, name):
    assert bucket(f1) == name


def test_sample_ids_is_fixed_and_order_independent():
    ids = [f"q{i}" for i in range(100)]
    assert sample_ids(ids, 10) == sample_ids(list(reversed(ids)), 10)
    assert len(sample_ids(ids, 10)) == 10


def test_summarize_counts_disagreements():
    rows = {
        "a": {"answer_f1": 0.0, "verdict": "correct", "judge_score": 1.0},
        "b": {"answer_f1": 1.0, "verdict": "correct", "judge_score": 1.0},
        "c": {"answer_f1": 0.6, "verdict": "incorrect", "judge_score": 0.0},
        "d": {"answer_f1": 0.3, "verdict": "partial", "judge_score": 0.5},
    }
    s = summarize(rows)
    assert s["judge_score"] == 0.625
    assert (s["low_f1_judged_correct"], s["low_f1_total"]) == (1, 2)
    assert (s["high_f1_judged_incorrect"], s["high_f1_total"]) == (1, 2)
    assert s["verdicts_by_f1"]["f1 = 0"] == {"correct": 1}
