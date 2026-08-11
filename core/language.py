"""Определение языка текста и выбор multilingual моделей."""

from __future__ import annotations

import re

# Простая эвристика: кириллица → ru, иначе en
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")

BI_ENCODER_MULTILINGUAL = "paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MULTILINGUAL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def detect_language(text: str) -> str:
    """Возвращает 'ru' если текст содержит кириллицу, иначе 'en'."""
    return "ru" if _CYRILLIC_RE.search(text) else "en"


def get_bi_encoder_name(lang: str | None = None, text: str | None = None) -> str:
    """Выбирает би-энкодер в зависимости от языка."""
    if lang is None and text:
        lang = detect_language(text)
    if lang == "ru":
        return BI_ENCODER_MULTILINGUAL
    from core.similarity import BI_ENCODER_MINILM

    return BI_ENCODER_MINILM


def get_cross_encoder_name(lang: str | None = None, text: str | None = None) -> str:
    """Выбирает кросс-энкодер в зависимости от языка."""
    if lang is None and text:
        lang = detect_language(text)
    if lang == "ru":
        return CROSS_ENCODER_MULTILINGUAL
    from core.similarity import DEFAULT_CROSS_ENCODER

    return DEFAULT_CROSS_ENCODER
