"""Скореры подходов для ablation: tfidf | bi-minilm | bi-e5 | full."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Scorer = Callable[[list[tuple[str, str]]], np.ndarray]


def get_scorer(approach: str) -> Scorer:
    if approach == "tfidf":
        from core.similarity import TfidfScorer

        return TfidfScorer().score_pairs
    if approach == "bi-minilm":
        from core.similarity import BI_ENCODER_MINILM, BiEncoderScorer

        return BiEncoderScorer(BI_ENCODER_MINILM).score_pairs
    if approach == "bi-e5":
        from core.similarity import BI_ENCODER_E5, BiEncoderScorer

        return BiEncoderScorer(BI_ENCODER_E5).score_pairs
    if approach == "full":
        # На парном бенчмарке финальный скор полного пайплайна — это скор кросс-энкодера
        # (би-энкодер на retrieval-стадии лишь отбирает кандидатов).
        from core.similarity import CROSS_ENCODER_MSMARCO, CrossEncoderScorer

        return CrossEncoderScorer(CROSS_ENCODER_MSMARCO).score_pairs
    if approach == "full-stsb":
        # Вариант полного пайплайна с кросс-энкодером, обученным на STS-B
        # (ms-marco оптимизирован под retrieval, stsb-roberta — под graded similarity).
        from core.similarity import CROSS_ENCODER_STSB, CrossEncoderScorer

        return CrossEncoderScorer(CROSS_ENCODER_STSB).score_pairs
    if approach == "full-ft":
        # Fine-tuned на PAWS (обучена через eval/finetune.py)
        from core.similarity import CROSS_ENCODER_FINE_TUNED, CrossEncoderScorer

        return CrossEncoderScorer(CROSS_ENCODER_FINE_TUNED).score_pairs
    raise ValueError(f"Неизвестный подход: {approach}")


APPROACHES = ["tfidf", "bi-minilm", "bi-e5", "full", "full-stsb", "full-ft"]
