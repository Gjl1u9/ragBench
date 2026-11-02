from ragbench.data import parse_paper, split_stats


def test_paragraphs_get_stable_ids_and_skip_blanks(raw_paper):
    paper, _ = parse_paper(raw_paper)
    assert [p.id for p in paper.paragraphs] == [f"1234.5678:p{i}" for i in range(5)]
    assert [p.text for p in paper.paragraphs][:2] == ["Tiny things matter.", "We propose TinyNet."]
    assert paper.paragraphs[2].section == "Method ::: Model"


def test_captions_become_float_paragraphs(raw_paper):
    paper, _ = parse_paper(raw_paper)
    caption = paper.paragraphs[-1]
    assert caption.is_float
    assert caption.section == "Figures and Tables"


def test_evidence_resolves_to_paragraph_ids(raw_paper):
    _, questions = parse_paper(raw_paper)
    answer = questions[0].answers[0]
    assert answer.type == "extractive"
    assert answer.text == "two layers"
    assert answer.evidence_ids == ["1234.5678:p2"]
    assert answer.heading_evidence == ["Model"]  # last component of "Method ::: Model"
    assert answer.unmatched_evidence == []


def test_float_evidence_and_unmatched_evidence(raw_paper):
    _, questions = parse_paper(raw_paper)
    yes, unanswerable = questions[1].answers
    assert (yes.type, yes.text) == ("yes_no", "Yes")
    assert yes.evidence_ids == ["1234.5678:p4"]
    assert yes.unmatched_evidence == ["A sentence that is not in the paper."]
    assert (unanswerable.type, unanswerable.evidence_ids) == ("unanswerable", [])


def test_split_stats(raw_paper):
    paper, questions = parse_paper(raw_paper)
    stats = split_stats([paper], questions)
    assert stats["questions"] == 2
    assert stats["answers"] == 3
    assert stats["heading_evidence"] == 1
    assert stats["evidence_match_rate"] == round(2 / 3, 4)
