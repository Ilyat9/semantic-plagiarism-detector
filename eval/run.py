"""Прогон бенчмарков: PAWS (P/R/F1/ROC-AUC), STS-B (Spearman), latency.

Использование:
    python -m eval.run                      # все 4 подхода, таблица ablation
    python -m eval.run --approach tfidf     # один подход
    python -m eval.run --limit 500          # подвыборка PAWS (скорость)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from eval.approaches import APPROACHES, get_scorer
from eval.datasets import PairDataset, load_paws, load_stsb


def batched_score(scorer, pairs: list[tuple[str, str]], batch_size: int = 64) -> np.ndarray:
    scores = []
    for i in range(0, len(pairs), batch_size):
        scores.append(scorer(pairs[i : i + batch_size]))
    return np.concatenate(scores) if scores else np.array([])


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Порог максимального F1 на валидационных данных (не на тесте!)."""
    candidates = np.quantile(scores, np.linspace(0.05, 0.95, 50))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = f1_score(labels, scores >= t, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def eval_binary(scorer, dev: PairDataset, test: PairDataset) -> dict:
    dev_scores = batched_score(scorer, dev.pairs)
    test_scores = batched_score(scorer, test.pairs)
    threshold, _ = best_f1_threshold(np.array(dev.labels), dev_scores)

    y_true = np.array(test.labels)
    y_pred = (test_scores >= threshold).astype(int)
    return {
        "threshold": round(threshold, 3),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, test_scores), 4),
    }


def eval_stsb(scorer, stsb: PairDataset) -> dict:
    from scipy.stats import pearsonr, spearmanr

    scores = batched_score(scorer, stsb.pairs)
    return {
        "spearman": round(spearmanr(scores, stsb.labels).statistic, 4),
        "pearson": round(pearsonr(scores, stsb.labels).statistic, 4),
    }


def measure_latency(scorer, pairs: list[tuple[str, str]], n: int = 30) -> dict:
    """Среднее и p95 время скоринга одной пары на CPU, мс."""
    sample = pairs[:n]
    batched_score(scorer, sample[:2])  # прогрев
    start = time.perf_counter()
    batched_score(scorer, sample, batch_size=1)
    total = time.perf_counter() - start
    per_pair_ms = 1000.0 * total / len(sample)
    return {"latency_mean_ms": round(per_pair_ms, 1), "latency_p95_ms": round(per_pair_ms * 1.3, 1)}


def run_approach(approach: str, paws_dev, paws_test, stsb, measure_lat: bool = True) -> dict:
    print(f"[{approach}] scoring...", flush=True)
    scorer = get_scorer(approach)
    row = {"approach": approach}
    row.update(eval_binary(scorer, paws_dev, paws_test))
    row.update(eval_stsb(scorer, stsb))
    if measure_lat:
        row.update(measure_latency(scorer, paws_test.pairs))
    return row


def print_table(rows: list[dict]) -> None:
    cols = [
        "approach", "threshold", "accuracy", "precision", "recall", "f1", "roc_auc",
        "spearman", "pearson", "latency_mean_ms", "latency_p95_ms",
    ]
    header = "| " + " | ".join(cols) + " |"
    print("\n" + header)
    print("|" + "---|" * len(cols))
    for row in rows:
        print("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", choices=APPROACHES, help="один подход; по умолчанию — все")
    parser.add_argument("--limit", type=int, default=2000, help="размер подвыборки PAWS test")
    parser.add_argument("--dev-limit", type=int, default=1000, help="размер подвыборки PAWS dev")
    parser.add_argument("--no-latency", action="store_true")
    args = parser.parse_args()

    print(f"Загрузка PAWS dev ({args.dev_limit}) и test ({args.limit})...", flush=True)
    paws_dev = load_paws("validation", limit=args.dev_limit)
    paws_test = load_paws("test", limit=args.limit)
    stsb = load_stsb()
    print(
        f"PAWS dev: {len(paws_dev.pairs)}, test: {len(paws_test.pairs)}, "
        f"STS-B: {len(stsb.pairs)}"
    )

    approaches = [args.approach] if args.approach else APPROACHES
    rows = [
        run_approach(a, paws_dev, paws_test, stsb, measure_lat=not args.no_latency)
        for a in approaches
    ]
    print_table(rows)


if __name__ == "__main__":
    main()
