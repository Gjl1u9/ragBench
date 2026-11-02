"""Load QASPER into typed records, with evidence mapped to stable paragraph ids.

QASPER evidence is given as paragraph text. We give every paragraph (and every figure or
table caption) an id like ``"<paper_id>:p12"`` and resolve each evidence string to those ids,
so retrieval can be scored exactly: a retrieved chunk is a hit if it covers a gold paragraph.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from datasets import load_dataset

from ragbench.config import settings

DATASET = "allenai/qasper"
# The dataset ships a loading script, which current `datasets` no longer runs; the Hub's
# auto-converted parquet branch has the same splits and schema.
REVISION = "refs/convert/parquet"
FLOAT_PREFIX = "FLOAT SELECTED: "

Split = Literal["train", "validation", "test"]
AnswerType = Literal["extractive", "abstractive", "yes_no", "unanswerable"]


@dataclass(frozen=True)
class Paragraph:
    id: str
    paper_id: str
    section: str
    text: str
    is_float: bool = False  # figure or table caption


@dataclass(frozen=True)
class Paper:
    id: str
    title: str
    abstract: str
    paragraphs: list[Paragraph]


@dataclass(frozen=True)
class Answer:
    """One annotator's answer to a question."""

    type: AnswerType
    text: str  # free-form answer, joined spans, "Yes"/"No", or "" when unanswerable
    spans: list[str]
    evidence_ids: list[str]  # resolved paragraph ids
    # Annotators sometimes selected a section heading as evidence. A heading has no content
    # to retrieve, so it is kept apart rather than counted as a miss.
    heading_evidence: list[str] = field(default_factory=list)
    unmatched_evidence: list[str] = field(default_factory=list)  # evidence we could not map


@dataclass(frozen=True)
class Question:
    id: str
    paper_id: str
    text: str
    answers: list[Answer]


def parse_paper(raw: dict) -> tuple[Paper, list[Question]]:
    paper_id = raw["id"]
    paragraphs: list[Paragraph] = []

    def add(section: str, text: str, is_float: bool = False) -> None:
        text = text.strip()
        if text:
            pid = f"{paper_id}:p{len(paragraphs)}"
            paragraphs.append(Paragraph(pid, paper_id, section, text, is_float))

    full_text = raw["full_text"]
    for section, paras in zip(full_text["section_name"], full_text["paragraphs"], strict=True):
        for text in paras:
            add(section or "", text)
    for caption in raw["figures_and_tables"]["caption"]:
        add("Figures and Tables", caption, is_float=True)

    by_text: dict[str, str] = {}
    for p in paragraphs:
        by_text.setdefault(p.text, p.id)
    # Nested sections are named "Parent ::: Child"; annotators may pick either form.
    headings = {s.strip() for s in full_text["section_name"] if s}
    headings |= {s.split(" ::: ")[-1] for s in headings}

    qas = raw["qas"]
    questions = [
        Question(
            id=qid,
            paper_id=paper_id,
            text=qtext,
            answers=[_parse_answer(a, by_text, headings) for a in answers["answer"]],
        )
        for qid, qtext, answers in zip(
            qas["question_id"], qas["question"], qas["answers"], strict=True
        )
    ]
    paper = Paper(paper_id, raw["title"], raw["abstract"], paragraphs)
    return paper, questions


def _parse_answer(raw: dict, by_text: dict[str, str], headings: set[str]) -> Answer:
    if raw["unanswerable"]:
        type_, text, spans = "unanswerable", "", []
    elif raw["extractive_spans"]:
        spans = list(raw["extractive_spans"])
        type_, text = "extractive", ", ".join(spans)
    elif raw["yes_no"] is not None:
        type_, text, spans = "yes_no", "Yes" if raw["yes_no"] else "No", []
    else:
        type_, text, spans = "abstractive", raw["free_form_answer"], []

    evidence_ids: list[str] = []
    heading_ev: list[str] = []
    unmatched: list[str] = []
    for ev in raw["evidence"]:
        key = ev.strip().removeprefix(FLOAT_PREFIX).strip()
        if not key:
            continue
        pid = by_text.get(key)
        if pid is not None:
            if pid not in evidence_ids:
                evidence_ids.append(pid)
        elif key in headings:
            heading_ev.append(ev)
        else:
            unmatched.append(ev)
    return Answer(type_, text, spans, evidence_ids, heading_ev, unmatched)


def load_split(split: Split) -> tuple[list[Paper], list[Question]]:
    """Download (first call only) and parse one QASPER split."""
    ds = load_dataset(
        DATASET, revision=REVISION, split=split, cache_dir=str(settings.data_dir / "hf_cache")
    )
    papers: list[Paper] = []
    questions: list[Question] = []
    for raw in ds:
        paper, qs = parse_paper(raw)
        papers.append(paper)
        questions.extend(qs)
    return papers, questions


def split_stats(papers: list[Paper], questions: list[Question]) -> dict:
    answers = [a for q in questions for a in q.answers]
    # Headings are excluded: they are not retrievable content.
    evidence_total = sum(len(a.evidence_ids) + len(a.unmatched_evidence) for a in answers)
    unmatched = sum(len(a.unmatched_evidence) for a in answers)
    return {
        "papers": len(papers),
        "paragraphs": sum(len(p.paragraphs) for p in papers),
        "questions": len(questions),
        "answers": len(answers),
        "answer_types": dict(Counter(a.type for a in answers)),
        "evidence_strings": evidence_total,
        "heading_evidence": sum(len(a.heading_evidence) for a in answers),
        "evidence_match_rate": round(1 - unmatched / evidence_total, 4) if evidence_total else 1.0,
    }


if __name__ == "__main__":
    for split in ("train", "validation"):
        print(split, split_stats(*load_split(split)))
