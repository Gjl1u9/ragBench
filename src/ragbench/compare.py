"""Compare two evaluation runs question by question.

Run with: uv run python -m ragbench.compare results/<baseline>.json results/<candidate>.json

For each metric, reports the mean difference (candidate - baseline) with a paired bootstrap
95% confidence interval, and how many questions got better, worse, or stayed the same. If the
interval includes 0, the runs are not distinguishable on this data.
"""

import argparse
import json
import random
from pathlib import Path


def paired_bootstrap(diffs: list[float], samples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% confidence interval of the mean of `diffs`, by resampling questions."""
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(samples))
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def compare(baseline: dict, candidate: dict) -> dict:
    base, cand = baseline["per_question"], candidate["per_question"]
    if base.keys() != cand.keys():
        raise ValueError("runs scored different questions; compare runs on the same split")
    ids = sorted(base)
    out = {}
    for metric in next(iter(base.values())):
        diffs = [cand[q][metric] - base[q][metric] for q in ids]
        low, high = paired_bootstrap(diffs)
        out[metric] = {
            "baseline": round(sum(base[q][metric] for q in ids) / len(ids), 4),
            "candidate": round(sum(cand[q][metric] for q in ids) / len(ids), 4),
            "diff": round(sum(diffs) / len(diffs), 4),
            "ci95": [round(low, 4), round(high, 4)],
            "better": sum(d > 0 for d in diffs),
            "worse": sum(d < 0 for d in diffs),
            "same": sum(d == 0 for d in diffs),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    baseline, candidate = (json.loads(p.read_text()) for p in (args.baseline, args.candidate))
    print(f"{candidate['experiment']} vs {baseline['experiment']}")
    for metric, row in compare(baseline, candidate).items():
        low, high = row["ci95"]
        print(
            f"{metric:10s} {row['baseline']:.4f} -> {row['candidate']:.4f}  "
            f"diff {row['diff']:+.4f} [{low:+.4f}, {high:+.4f}]  "
            f"better {row['better']} / worse {row['worse']} / same {row['same']}"
        )


if __name__ == "__main__":
    main()
