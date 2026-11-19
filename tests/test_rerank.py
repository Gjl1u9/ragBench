from ragbench.rerank import order_by_scores
from ragbench.retrieve import Hit


def hits(*chunk_ids):
    return [Hit(c, 0.0, f"text {c}", (c,)) for c in chunk_ids]


def test_order_by_scores_sorts_best_first_and_replaces_scores():
    reranked = order_by_scores(hits("a", "b", "c"), [0.1, 0.9, 0.5], k=3)
    assert [h.chunk_id for h in reranked] == ["b", "c", "a"]
    assert [h.score for h in reranked] == [0.9, 0.5, 0.1]


def test_order_by_scores_can_lift_a_low_ranked_candidate_to_the_top():
    candidates = hits(*"abcdefghij")
    scores = [0.0] * 9 + [1.0]  # the last candidate is the only relevant one
    assert order_by_scores(candidates, scores, k=1)[0].chunk_id == "j"


def test_order_by_scores_truncates_to_k_and_breaks_ties_by_original_rank():
    reranked = order_by_scores(hits("a", "b", "c"), [0.5, 0.5, 0.1], k=2)
    assert [h.chunk_id for h in reranked] == ["a", "b"]
