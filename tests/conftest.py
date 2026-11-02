import pytest


def _answer(**overrides) -> dict:
    answer = {
        "unanswerable": False,
        "extractive_spans": [],
        "yes_no": None,
        "free_form_answer": "",
        "evidence": [],
        "highlighted_evidence": [],
    }
    answer.update(overrides)
    return answer


@pytest.fixture
def raw_paper() -> dict:
    """A tiny record in QASPER's raw (column-oriented) shape."""
    return {
        "id": "1234.5678",
        "title": "A Tiny Paper",
        "abstract": "We study tiny things.",
        "full_text": {
            "section_name": ["Introduction", "Method ::: Model", "Results"],
            "paragraphs": [
                ["Tiny things matter.", "   ", "We propose TinyNet."],
                ["TinyNet has two layers."],
                ["TinyNet beats the baseline by 3 points."],
            ],
        },
        "figures_and_tables": {
            "caption": ["Table 1: Accuracy of TinyNet and the baseline."],
            "file": ["1-Table1-1.png"],
        },
        "qas": {
            "question": ["How many layers does TinyNet have?", "Does TinyNet beat the baseline?"],
            "question_id": ["q1", "q2"],
            "answers": [
                {
                    "answer": [
                        _answer(
                            extractive_spans=["two layers"],
                            evidence=["TinyNet has two layers.", "Model"],
                        )
                    ],
                    "annotation_id": ["a1"],
                    "worker_id": ["w1"],
                },
                {
                    "answer": [
                        _answer(
                            yes_no=True,
                            evidence=[
                                "FLOAT SELECTED: Table 1: Accuracy of TinyNet and the baseline.",
                                "A sentence that is not in the paper.",
                            ],
                        ),
                        _answer(unanswerable=True),
                    ],
                    "annotation_id": ["a2", "a3"],
                    "worker_id": ["w2", "w3"],
                },
            ],
        },
    }
