import pytest

from ragbench.compare import compare, paired_bootstrap


def run(experiment, scores):
    return {"experiment": experiment, "per_question": {q: {"mrr": s} for q, s in scores.items()}}


def test_compare_counts_wins_and_losses():
    base = run("a", {"q1": 0.0, "q2": 1.0, "q3": 0.5})
    cand = run("b", {"q1": 1.0, "q2": 1.0, "q3": 0.0})
    row = compare(base, cand)["mrr"]
    assert (row["baseline"], row["candidate"]) == (0.5, 0.6667)
    assert row["diff"] == pytest.approx(0.1667, abs=1e-4)
    assert (row["better"], row["worse"], row["same"]) == (1, 1, 1)


def test_bootstrap_interval_is_tight_for_a_constant_difference():
    assert paired_bootstrap([0.2] * 50) == pytest.approx((0.2, 0.2))


def test_bootstrap_interval_contains_zero_for_noise():
    low, high = paired_bootstrap([1.0, -1.0] * 100)
    assert low < 0 < high


def test_compare_rejects_runs_on_different_questions():
    with pytest.raises(ValueError):
        compare(run("a", {"q1": 0.0}), run("b", {"q2": 0.0}))
