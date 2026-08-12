"""Оценка на собственном корпусе: confusion matrix по 3 классам.

Использование: python -m eval.own_corpus [--corpus data/eval/pairs.jsonl]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from core.classify import Thresholds, Verdict, classify_fragment
from core.similarity import CrossEncoderScorer, JaccardScorer
from eval.run import batched_score

LABELS = [Verdict.VERBATIM, Verdict.PARAPHRASE, Verdict.ORIGINAL]


def load_corpus(path: Path) -> tuple[list[tuple[str, str]], list[str], list[dict]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [(r["text_a"], r["text_b"]) for r in rows], [r["label"] for r in rows], rows


def confusion_matrix(y_true: list[str], y_pred: list[str]) -> np.ndarray:
    idx = {label.value: i for i, label in enumerate(LABELS)}
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        matrix[idx[t], idx[p.value]] += 1
    return matrix


def print_confusion(matrix: np.ndarray) -> None:
    names = [label.value for label in LABELS]
    header = "true\\pred".ljust(12) + "".join(n.rjust(12) for n in names)
    print(header)
    for i, name in enumerate(names):
        print(name.ljust(12) + "".join(str(v).rjust(12) for v in matrix[i]))
    accuracy = np.trace(matrix) / matrix.sum()
    # Wilson score interval for binomial proportion
    n = matrix.sum()
    z = norm.ppf(0.975)  # 95% CI
    p = accuracy
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half_width = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    ci_low, ci_high = centre - half_width, centre + half_width
    print(f"\nAccuracy: {accuracy:.3f}  (n={n}, 95% CI: {ci_low:.3f}–{ci_high:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/eval/pairs.jsonl"))
    parser.add_argument("--show-errors", action="store_true", help="показать ошибочные пары")
    args = parser.parse_args()

    pairs, y_true, rows = load_corpus(args.corpus)
    thresholds = Thresholds.load()
    print(f"Корпус: {len(pairs)} пар. Пороги: {thresholds}")

    sem_scores = batched_score(CrossEncoderScorer().score_pairs, pairs)
    # Лексический скор — тот же, что в продакшен-пайплайне (char n-gram Jaccard)
    lex_scores = JaccardScorer().score_pairs(pairs)
    y_pred = [
        classify_fragment(float(lex), float(sem), thresholds)
        for lex, sem in zip(lex_scores, sem_scores, strict=True)
    ]

    print_confusion(confusion_matrix(y_true, y_pred))

    if args.show_errors:
        print("\nОшибки:")
        for row, true_label, pred in zip(rows, y_true, y_pred, strict=True):
            if pred.value != true_label:
                print(f"- [{row['id']}] true={true_label} pred={pred.value}")
                print(f"  A: {row['text_a'][:150]}")
                print(f"  B: {row['text_b'][:150]}")


if __name__ == "__main__":
    main()
