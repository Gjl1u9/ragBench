"""Split papers into retrievable chunks.

Every chunk records which paragraph ids it covers, so retrieval can be scored against
QASPER's paragraph-level evidence whatever the chunking strategy.
"""

from dataclasses import dataclass

from ragbench.data import Paper


@dataclass(frozen=True)
class Chunk:
    id: str
    paper_id: str
    section: str
    text: str
    paragraph_ids: tuple[str, ...]


def _with_header(section: str, text: str, with_section: bool) -> str:
    return f"{section}\n{text}" if with_section and section else text


def paragraph_chunks(paper: Paper, with_section: bool = True) -> list[Chunk]:
    """One chunk per paragraph (or figure/table caption)."""
    return [
        Chunk(
            id=f"{paper.id}:c{i}",
            paper_id=paper.id,
            section=p.section,
            text=_with_header(p.section, p.text, with_section),
            paragraph_ids=(p.id,),
        )
        for i, p in enumerate(paper.paragraphs)
    ]


def window_chunks(
    paper: Paper, max_words: int = 200, overlap: int = 50, with_section: bool = True
) -> list[Chunk]:
    """Fixed-size word windows over the paper's paragraphs in order, overlapping by `overlap`.

    Windows may cross paragraph and section boundaries; the chunk is labelled with the
    section its first word belongs to.
    """
    if not 0 <= overlap < max_words:
        raise ValueError("need 0 <= overlap < max_words")

    words: list[str] = []
    owners: list[int] = []  # index into paper.paragraphs for each word
    for i, p in enumerate(paper.paragraphs):
        tokens = p.text.split()
        words.extend(tokens)
        owners.extend([i] * len(tokens))

    chunks: list[Chunk] = []
    step = max_words - overlap
    for start in range(0, len(words), step):
        end = min(start + max_words, len(words))
        covered = dict.fromkeys(owners[start:end])  # ordered, de-duplicated
        section = paper.paragraphs[owners[start]].section
        chunks.append(
            Chunk(
                id=f"{paper.id}:c{len(chunks)}",
                paper_id=paper.id,
                section=section,
                text=_with_header(section, " ".join(words[start:end]), with_section),
                paragraph_ids=tuple(paper.paragraphs[i].id for i in covered),
            )
        )
        if end == len(words):
            break
    return chunks
