"""Тест мультиязычности: русский текст через multilingual модели."""

from __future__ import annotations

import pytest

from core.language import detect_language
from core.similarity import BiEncoderScorer

RU_TEXT = (
    "Фотосинтез — процесс преобразования световой энергии в химическую. "
    "Хлорофил поглощает свет в синей и красной частях спектра. "
    "В результате расщепления воды выделяется кислород."
)

RU_PARAPHRASE = (
    "Растения превращают солнечный свет в химическую энергию посредством фотосинтеза. "
    "Пигмент хлорофил улавливает излучение преимущественно в синем и красном диапазоне. "
    "При разложении молекул воды образуется кислород."
)

EN_TEXT = "Photosynthesis converts sunlight into chemical energy in plants."


class TestLanguageDetection:
    def test_detect_russian(self):
        assert detect_language(RU_TEXT) == "ru"

    def test_detect_english(self):
        assert detect_language(EN_TEXT) == "en"


class TestMultilingualScoring:
    @pytest.mark.slow
    def test_russian_paraphrase_semantic_score(self):
        """Русский перефраз должен давать высокий семантический скор."""
        scorer = BiEncoderScorer("paraphrase-multilingual-MiniLM-L12-v2")
        pairs = [(RU_TEXT, RU_PARAPHRASE)]
        scores = scorer.score_pairs(pairs)
        assert scores[0] > 0.5, f"semantic score = {scores[0]:.3f}, ожидалось > 0.5"

    @pytest.mark.slow
    def test_russian_vs_english_low_score(self):
        """Русский текст vs несвязанный английский — низкий скор.

        NB: cross-lingual модель специально сближает тексты с ОДИНАКОВЫМ смыслом
        на разных языках (RU-описание фотосинтеза vs EN-описание: ~0.80).
        Поэтому низкий скор проверяем на тематически несвязанной паре.
        """
        scorer = BiEncoderScorer("paraphrase-multilingual-MiniLM-L12-v2")
        unrelated_en = "The stock market closed higher on Tuesday amid a rally in tech shares."
        scores = scorer.score_pairs([(RU_TEXT, unrelated_en)])
        assert scores[0] < 0.5, f"semantic score = {scores[0]:.3f}, ожидалось < 0.5"

    @pytest.mark.slow
    def test_cross_lingual_same_meaning_high_score(self):
        """Cross-lingual пара с общим смыслом (RU vs EN о фотосинтезе) — высокий скор."""
        scorer = BiEncoderScorer("paraphrase-multilingual-MiniLM-L12-v2")
        scores = scorer.score_pairs([(RU_TEXT, EN_TEXT)])
        assert scores[0] > 0.5, f"semantic score = {scores[0]:.3f}, ожидалось > 0.5"
