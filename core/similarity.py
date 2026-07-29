"""Скореры сходства текстов: TF-IDF бейзлайн, би-энкодер, кросс-энкодер."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BI_ENCODER_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
BI_ENCODER_E5 = "intfloat/e5-base-v2"
BI_ENCODER_MULTILINGUAL = "paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MSMARCO = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_STSB = "cross-encoder/stsb-roberta-base"
CROSS_ENCODER_FINE_TUNED = "data/models/cross-encoder-paws-finetuned"


class TfidfScorer:
    """Лексическое сходство: TF-IDF (1,2)-граммы + косинус."""

    def __init__(self, max_features: int = 5000):
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Косинусное сходство для списка пар (a, b) -> [n]."""
        if not pairs:
            return np.array([])
        a_texts = [a for a, _ in pairs]
        b_texts = [b for _, b in pairs]
        matrix = self._vectorizer.fit_transform(a_texts + b_texts)
        n = len(a_texts)
        sims = cosine_similarity(matrix[:n], matrix[n:])
        return np.diag(sims)


_bi_encoders: dict[str, object] = {}


def get_bi_encoder(model_name: str = BI_ENCODER_MINILM):
    """Ленивая загрузка би-энкодера (кэш по имени модели)."""
    if model_name not in _bi_encoders:
        from sentence_transformers import SentenceTransformer

        _bi_encoders[model_name] = SentenceTransformer(model_name, device="cpu")
    return _bi_encoders[model_name]


class BiEncoderScorer:
    """Семантическое сходство: би-энкодер + косинус по эмбеддингам."""

    def __init__(self, model_name: str = BI_ENCODER_MINILM):
        self._model = get_bi_encoder(model_name)
        self._is_e5 = "e5" in model_name

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        # E5-модели требуют префиксов query:/passage:
        if self._is_e5:
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def cosine(self, a_emb: np.ndarray, b_emb: np.ndarray) -> np.ndarray:
        """Попарный косинус для нормализованных эмбеддингов [n, d] -> [n]."""
        return np.sum(a_emb * b_emb, axis=1)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.array([])
        a_emb = self.encode([a for a, _ in pairs], is_query=True)
        b_emb = self.encode([b for _, b in pairs])
        return self.cosine(a_emb, b_emb)


_cross_encoders: dict[str, object] = {}

# По итогам ablation (eval/run.py): stsb-roberta даёт заметно лучшую калибровку
# семантического скора (Spearman 0.92 на STS-B, ROC-AUC выше, чем у ms-marco),
# поэтому в продакшен-пайплайне используется он. ms-marco остаётся строкой ablation.
DEFAULT_CROSS_ENCODER = CROSS_ENCODER_STSB


def get_cross_encoder(model_name: str = DEFAULT_CROSS_ENCODER):
    """Ленивая загрузка кросс-энкодера."""
    if model_name not in _cross_encoders:
        from sentence_transformers import CrossEncoder

        _cross_encoders[model_name] = CrossEncoder(model_name, device="cpu")
    return _cross_encoders[model_name]


class CrossEncoderScorer:
    """Точная перепроверка пар: кросс-энкодер -> скор [0, 1].

    Учитывает разницу в выходах моделей: ms-marco выдаёт неограниченные логиты
    (нужен sigmoid), stsb-roberta обучена выдавать сходство уже в [0, 1]
    (sigmoid поверх неё разрушил бы калибровку).
    """

    # Модели, чьи выходы уже нормированы в [0, 1]
    _PRENORMALIZED = {CROSS_ENCODER_STSB}

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER):
        self._model = get_cross_encoder(model_name)
        self._prenormalized = model_name in self._PRENORMALIZED

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.array([])
        raw = np.asarray(self._model.predict(pairs), dtype=float)
        if self._prenormalized:
            return np.clip(raw, 0.0, 1.0)
        return 1.0 / (1.0 + np.exp(-raw))
