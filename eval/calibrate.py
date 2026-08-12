"""Калибровка порогов T1 (lexical) и T2 (semantic) -> core/thresholds.json.

T2 (семантический порог вердикта) — grid search по PAWS-dev, максимум F1
бинарной задачи «перефраз vs не перефраз» на скорах кросс-энкодера.
T1 (лексический порог verbatim vs paraphrase) — по собственному размеченному
корпусу data/eval/pairs.jsonl: максимум F1 разделения verbatim/paraphrase
на том же лексическом скоре, что и в продакшене (char n-gram Jaccard).

Использование: python -m eval.calibrate [--dev-limit 1000] [--t1-only | --t2-only]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.classify import Thresholds
from core.similarity import CrossEncoderScorer, JaccardScorer
from eval.datasets import load_paws
from eval.run import batched_score, best_f1_threshold

OWN_CORPUS = Path("data/eval/pairs.jsonl")


def calibrate_t2(dev_limit: int = 1000) -> float:
    dev = load_paws("validation", limit=dev_limit)
    scorer = CrossEncoderScorer().score_pairs
    scores = batched_score(scorer, dev.pairs)
    t2, f1 = best_f1_threshold(np.array(dev.labels), scores)
    print(f"T2 (semantic) = {t2:.3f}  (PAWS-dev F1 = {f1:.4f}, n={len(dev.pairs)})")
    return t2


def calibrate_t1() -> float:
    """Порог lex-скора, разделяющий verbatim и paraphrase пары своего корпуса.

    NB: «полные» verbatim-пары — точные копии (Jaccard = 1.0), поэтому
    разделение было бы тривиальным; частичные verbatim (1–2 скопированных
    предложения из 3–4) делают калибровку ближе к продакшен-окну.
    """
    pairs, labels = [], []
    for line in OWN_CORPUS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["label"] in ("verbatim", "paraphrase"):
            pairs.append((row["text_a"], row["text_b"]))
            labels.append(1 if row["label"] == "verbatim" else 0)
    if not pairs:
        print("Свой корпус не найден/пуст — T1 оставлен по умолчанию.")
        return Thresholds().t1_lexical
    scores = JaccardScorer().score_pairs(pairs)
    t1, f1 = best_f1_threshold(np.array(labels), scores)
    print(f"T1 (lexical) = {t1:.3f}  (own corpus F1 = {f1:.4f}, n={len(pairs)})")
    return t1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-limit", type=int, default=1000)
    parser.add_argument("--t1-only", action="store_true", help="только T1 (не трогать T2)")
    parser.add_argument("--t2-only", action="store_true", help="только T2 (не трогать T1)")
    args = parser.parse_args()

    current = Thresholds.load()
    t2 = current.t2_semantic if args.t1_only else calibrate_t2(args.dev_limit)
    t1 = current.t1_lexical if args.t2_only else calibrate_t1()

    thresholds = Thresholds(t1_lexical=round(t1, 3), t2_semantic=round(t2, 3))
    thresholds.save()
    print(f"Сохранено в core/thresholds.json: {thresholds}")


if __name__ == "__main__":
    main()
