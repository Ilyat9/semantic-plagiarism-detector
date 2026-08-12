"""Скореры сходства текстов: лексический (char n-gram Jaccard), би-энкодер, кросс-энкодер."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.verbatim_matcher import char_ngrams

BI_ENCODER_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
BI_ENCODER_E5 = "intfloat/e5-base-v2"
BI_ENCODER_MULTILINGUAL = "paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MSMARCO = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_STSB = "cross-encoder/stsb-roberta-base"
CROSS_ENCODER_FINE_TUNED = "data/models/cross-encoder-paws-finetuned"


class JaccardScorer:
    """Лексическое сходство: character 5-gram Jaccard (как в VerbatimMatcher).

    Батч-независимый: скор пары не зависит от того, с какими ещё парами она
    передана (в отличие от TF-IDF, где IDF фитится на батче — одна и та же пара
    давала разброс до 36% в зависимости от состава батча). Поэтому именно этот
    скорер используется в продакшен-пайплайне и при калибровке T1.
    """

    def __init__(self, n: int = 5):
        self._n = n

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Jaccard-сходство для списка пар (a, b) -> [n]."""
        scores = []
        for a, b in pairs:
            ga, gb = char_ngrams(a, self._n), char_ngrams(b, self._n)
            union = len(ga | gb)
            scores.append(len(ga & gb) / union if union else 0.0)
        return np.asarray(scores)


class TfidfScorer:
    """Лексическое сходство: TF-IDF (1,2)-граммы + косинус.

    NB: векторизатор фитится на переданном батче (IDF зависит от состава
    батча), поэтому скоры сравнимы только внутри одного прогона. Используется
    как baseline-строка в ablation (eval/run.py); в продакшене — JaccardScorer.
    """

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

    predict() sentence-transformers сам применяет активацию из конфига модели:
    Identity для retrieval-моделей (ms-marco -> логиты, нужен sigmoid) и
    Sigmoid для stsb-roberta и fine-tuned чекпойнтов (уже вероятности).
    Определяем случай по model.activation_fn, а не по имени модели — иначе
    sigmoid поверх sigmoid'а сжимает скоры в 0.5–0.73 и убивает разделение
    (см. failure analysis #3 в README).
    """

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER):
        from torch import nn

        self._model = get_cross_encoder(model_name)
        self._needs_sigmoid = isinstance(self._model.activation_fn, nn.Identity)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.array([])
        raw = np.asarray(self._model.predict(pairs), dtype=float)
        if self._needs_sigmoid:
            raw = 1.0 / (1.0 + np.exp(-raw))
        return np.clip(raw, 0.0, 1.0)
