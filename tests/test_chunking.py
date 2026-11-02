import pytest

from ragbench.chunking import paragraph_chunks, window_chunks
from ragbench.data import parse_paper


@pytest.fixture
def paper(raw_paper):
    return parse_paper(raw_paper)[0]


def test_paragraph_chunks_cover_one_paragraph_each(paper):
    chunks = paragraph_chunks(paper)
    assert len(chunks) == len(paper.paragraphs)
    assert chunks[2].paragraph_ids == ("1234.5678:p2",)
    assert chunks[2].text == "Method ::: Model\nTinyNet has two layers."


def test_paragraph_chunks_without_section(paper):
    assert paragraph_chunks(paper, with_section=False)[0].text == "Tiny things matter."


def test_window_chunks_overlap_and_cover_every_paragraph(paper):
    chunks = window_chunks(paper, max_words=6, overlap=2, with_section=False)
    words = [w for p in paper.paragraphs for w in p.text.split()]
    assert chunks[0].text.split() == words[:6]
    assert chunks[1].text.split()[:2] == words[4:6]  # overlap of 2
    assert chunks[-1].text.split()[-1] == words[-1]
    covered = {pid for c in chunks for pid in c.paragraph_ids}
    assert covered == {p.id for p in paper.paragraphs}
    assert len({c.id for c in chunks}) == len(chunks)


def test_window_chunk_spanning_paragraphs_lists_both(paper):
    first = window_chunks(paper, max_words=6, overlap=0)[0]
    # "Tiny things matter." (3 words) + "We propose TinyNet." (3 words)
    assert first.paragraph_ids == ("1234.5678:p0", "1234.5678:p1")
    assert first.section == "Introduction"


def test_window_chunks_rejects_bad_overlap(paper):
    with pytest.raises(ValueError):
        window_chunks(paper, max_words=5, overlap=5)
