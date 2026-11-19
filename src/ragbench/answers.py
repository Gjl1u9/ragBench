"""Answer F1, computed the same way as QASPER's official evaluator.

`normalize_answer` and `token_f1` follow scripts/evaluator.py in allenai/qasper-led-baseline
(Apache-2.0), which in turn takes them from the SQuAD v1.1 evaluation script. Gold answers are
built as that script builds them: extractive spans joined by ", ", "Yes"/"No" for booleans,
and "Unanswerable" for unanswerable annotations. A question's score is the best F1 over its
annotators.
"""

import re
import string
from collections import Counter

from ragbench.data import Answer, Question

UNANSWERABLE = "Unanswerable"


def normalize_answer(text: str) -> str:
    """Lowercase, and drop punctuation, articles and extra whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    same = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def gold_answer(answer: Answer) -> str:
    return UNANSWERABLE if answer.type == "unanswerable" else answer.text


def score_answer(question: Question, prediction: str) -> dict:
    """Best F1 over annotators, and the answer type of the annotator that gave it."""
    f1, answer_type = max(
        ((token_f1(prediction, gold_answer(a)), a.type) for a in question.answers),
        key=lambda pair: pair[0],
    )
    return {"answer_f1": f1, "answer_type": answer_type}
