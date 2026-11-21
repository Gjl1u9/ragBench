import pytest

from ragbench.agentic import (
    LIMIT_REACHED,
    MAX_SEARCHES,
    TOOL_NAME,
    PaperSearch,
    search_server,
)
from ragbench.chunking import paragraph_chunks
from ragbench.data import parse_paper
from ragbench.pipeline import Pipeline
from ragbench.retrieve import Hit


@pytest.fixture
def setup(raw_paper):
    paper, questions = parse_paper(raw_paper)
    chunks = paragraph_chunks(paper)
    asked = []

    def retrieve(question, k):
        # Fake retrieval: chunks whose text contains the query word, else the first chunk.
        asked.append(question.text)
        word = question.text.lower()
        matches = [c for c in chunks if word in c.text.lower()] or chunks[:1]
        return [Hit(c.id, 1.0, c.text, c.paragraph_ids) for c in matches[:k]]

    pipeline = Pipeline(retrieve, retrieve, config={})
    return pipeline, questions[0], chunks, asked


def test_search_scopes_by_question_and_uses_the_query_text(setup):
    pipeline, question, chunks, asked = setup
    result = PaperSearch(pipeline, question, chunks).search("layers")
    assert asked == ["layers"]
    assert result == "[1] Method ::: Model\nTinyNet has two layers."


def test_passage_numbers_stay_stable_across_searches(setup):
    pipeline, question, chunks, _ = setup
    search = PaperSearch(pipeline, question, chunks)
    search.search("layers")
    second = search.search("tinynet")  # returns the "layers" passage again among others
    assert "[1] Method ::: Model\nTinyNet has two layers." in second
    assert len(search.seen) == len({c for c in search.seen})


def test_search_budget_is_enforced(setup):
    pipeline, question, chunks, asked = setup
    search = PaperSearch(pipeline, question, chunks)
    for _ in range(MAX_SEARCHES):
        search.search("layers")
    assert search.search("layers") == LIMIT_REACHED
    assert search.searches == MAX_SEARCHES
    assert len(asked) == MAX_SEARCHES


def test_replay_rebuilds_the_same_passages(setup):
    pipeline, question, chunks, _ = setup
    live = PaperSearch(pipeline, question, chunks)
    live.search("layers")
    live.search("baseline")
    calls = [
        {"name": TOOL_NAME, "input": {"query": "layers"}},
        {"name": "other_tool", "input": {"query": "ignored"}},
        {"name": TOOL_NAME, "input": {"query": "baseline"}},
    ]
    replayed = PaperSearch(pipeline, question, chunks)
    replayed.replay(calls)
    assert list(replayed.seen) == list(live.seen)
    assert replayed.searches == 2


def test_search_server_exposes_one_tool(setup):
    pipeline, question, chunks, _ = setup
    server = search_server(PaperSearch(pipeline, question, chunks))
    assert server["name"] == "paper"
