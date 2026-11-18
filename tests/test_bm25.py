import pytest

from ragbench.bm25 import BM25Retriever, tokenize
from ragbench.chunking import paragraph_chunks
from ragbench.data import parse_paper
from ragbench.retrieve import Hit, reciprocal_rank_fusion


@pytest.fixture
def chunks(raw_paper):
    other = parse_paper({**raw_paper, "id": "9999.0000"})[0]
    return paragraph_chunks(parse_paper(raw_paper)[0]) + paragraph_chunks(other)


def test_tokenize_lowercases_and_keeps_model_names_whole():
    assert tokenize("BERT-base beats GPT-2 by 1.5 points.") == [
        "bert-base",
        "beats",
        "gpt-2",
        "by",
        "1.5",
        "points",
    ]


def test_bm25_ranks_the_keyword_match_first(chunks):
    hits = BM25Retriever(chunks).search("how many layers", "1234.5678", 3)
    assert hits[0].paragraph_ids == ("1234.5678:p2",)  # "TinyNet has two layers."
    assert len(hits) == 3
    assert hits[0].score > hits[1].score


def test_bm25_is_scoped_to_the_paper(chunks):
    hits = BM25Retriever(chunks).search("TinyNet", "9999.0000", 10)
    assert hits
    assert all(h.chunk_id.startswith("9999.0000:") for h in hits)


def test_bm25_query_with_no_known_terms_still_returns_k_hits(chunks):
    assert len(BM25Retriever(chunks).search("zzz", "1234.5678", 2)) == 2


def hit(chunk_id):
    return Hit(chunk_id, 0.0, "text", (chunk_id,))


def test_rrf_rewards_chunks_ranked_well_by_both_lists():
    dense = [hit("a"), hit("b"), hit("c")]
    keyword = [hit("b"), hit("c"), hit("d")]
    fused = reciprocal_rank_fusion([dense, keyword], k=4, rrf_k=60)
    # b: ranks 2 and 1; c: ranks 3 and 2; a and d appear in one list only.
    assert [h.chunk_id for h in fused] == ["b", "c", "a", "d"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_truncates_to_k_and_keeps_paragraph_ids():
    fused = reciprocal_rank_fusion([[hit("a"), hit("b")], [hit("a")]], k=1)
    assert [h.chunk_id for h in fused] == ["a"]
    assert fused[0].paragraph_ids == ("a",)
