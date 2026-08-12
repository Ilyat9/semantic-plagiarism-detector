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
    """Порог максимального F1 на валидационных данных (не на тесте!).

    Кандидаты — уникальные значения скоров: сетка по квантилям не покрывала
    хвосты распределения и могла выдать вырожденный порог (например 1.0).
    """
    candidates = np.unique(scores)
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
    """Среднее и p95 время скоринга одной пары на CPU, мс.

    Каждая пара замеряется отдельно (batch_size=1), p95 — честный перцентиль
    по per-pair таймингам, а не оценка от среднего.
    """
    sample = pairs[:n]
    batched_score(scorer, sample[:2])  # прогрев
    times_ms = []
    for pair in sample:
        start = time.perf_counter()
        batched_score(scorer, [pair], batch_size=1)
        times_ms.append(1000.0 * (time.perf_counter() - start))
    return {
        "latency_mean_ms": round(float(np.mean(times_ms)), 1),
        "latency_p95_ms": round(float(np.percentile(times_ms, 95)), 1),
    }


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
        "approach",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "spearman",
        "pearson",
        "latency_mean_ms",
        "latency_p95_ms",
    ]
    header = "| " + " | ".join(cols) + " |"
    print("\n" + header)
    print("|" + "---|" * len(cols))
    for row in rows:
        print("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")


# Человекочитаемые названия подходов для README-таблицы
_README_LABELS = {
    "tfidf": "TF-IDF + cosine (baseline)",
    "bi-minilm": "Bi-encoder MiniLM + cosine",
    "bi-e5": "Bi-encoder E5-base + cosine",
    "full": "Bi → Cross-encoder ms-marco",
    "full-stsb": "Bi → Cross-encoder **stsb-roberta** (дефолт пайплайна)",
    "full-ft": "Bi → Cross-encoder **stsb-roberta-ft**",
}


def print_readme_table(rows: list[dict], n_test: int, n_dev: int) -> None:
    """Печатает строки в том же формате, что таблица ablation в README.

    Нужно, чтобы перезамер сводился к копипасту: колонки и подписи совпадают
    с README, никакого ручного переноса цифр между форматами.
    """
    print("\n--- вставить в README, раздел Evaluation ---\n")
    print(
        "| Подход | Порог | Acc | Precision | Recall | F1 | ROC-AUC "
        "| STS-B Spearman | Latency/пара (mean / p95) |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        label = _README_LABELS.get(row["approach"], row["approach"])
        latency = (
            f"{row['latency_mean_ms']} / {row['latency_p95_ms']} мс"
            if "latency_mean_ms" in row
            else "—"
        )
        print(
            f"| {label} | {row['threshold']} | {row['accuracy']} | {row['precision']} "
            f"| {row['recall']} | {row['f1']} | {row['roc_auc']} | {row['spearman']} | {latency} |"
        )
    stamp = time.strftime("%Y-%m-%d")
    print(f"\n> Замер: {stamp}, PAWS test n={n_test}, dev n={n_dev}.")


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
        f"PAWS dev: {len(paws_dev.pairs)}, test: {len(paws_test.pairs)}, STS-B: {len(stsb.pairs)}"
    )

    approaches = [args.approach] if args.approach else APPROACHES
    rows = []
    for a in approaches:
        try:
            rows.append(run_approach(a, paws_dev, paws_test, stsb, measure_lat=not args.no_latency))
        except (OSError, ValueError) as exc:
            # full-ft требует обученного чекпойнта (eval.finetune); без него
            # остальные строки таблицы всё равно должны посчитаться.
            print(f"[{a}] пропущен: {exc}", flush=True)

    print_table(rows)
    print_readme_table(rows, n_test=len(paws_test.pairs), n_dev=len(paws_dev.pairs))


if __name__ == "__main__":
    main()
