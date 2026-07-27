"""Генерация поисковых фраз: окна 5–10 слов с максимальной отличительностью (IDF)."""

from __future__ import annotations

import math
import re
from collections import Counter

_WORD_RE = re.compile(r"[A-Za-z0-9'-]+")

MIN_PHRASE_WORDS = 5
MAX_PHRASE_WORDS = 10


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _document_idf(sentences_tokens: list[list[str]]) -> dict[str, float]:
    """IDF по предложениям самого документа: редкие внутри документа слова отличительны."""
    n_docs = len(sentences_tokens)
    df: Counter[str] = Counter()
    for tokens in sentences_tokens:
        df.update(set(tokens))
    return {word: math.log((1 + n_docs) / (1 + count)) + 1.0 for word, count in df.items()}


def generate_queries(
    text: str,
    sentences: list[str] | None = None,
    words_per_100: int = 1,
    min_queries: int = 10,
    max_queries: int = 30,
    window: int = 8,
) -> list[str]:
    """Выбирает словные окна с максимальной суммой IDF, без пересечений.

    Окна берутся внутри предложений (не пересекают границы), длина — `window` слов
    (последнее окно предложения может быть короче, но не меньше MIN_PHRASE_WORDS).
    """
    from core.chunking import split_sentences

    if sentences is None:
        sentences = split_sentences(text)
    sentences_tokens = [_tokenize(s) for s in sentences]
    sentences_tokens = [t for t in sentences_tokens if t]
    if not sentences_tokens:
        return []

    idf = _document_idf(sentences_tokens)
    total_words = sum(len(t) for t in sentences_tokens)
    n_queries = max(
        min_queries, min(max_queries, total_words // 100 * words_per_100 or min_queries)
    )

    candidates: list[tuple[float, int, int, str]] = []  # (score, sent_idx, start, phrase)
    for s_idx, tokens in enumerate(sentences_tokens):
        for start in range(0, len(tokens), window):
            window_tokens = tokens[start : start + window]
            if len(window_tokens) < MIN_PHRASE_WORDS:
                continue
            score = sum(idf.get(t, 1.0) for t in window_tokens)
            phrase = " ".join(window_tokens)
            candidates.append((score, s_idx, start, phrase))

    candidates.sort(key=lambda c: c[0], reverse=True)
    selected: list[str] = []
    used_spans: list[tuple[int, int, int]] = []  # (sent_idx, start, end)
    for _score, s_idx, start, phrase in candidates:
        end = start + window
        if any(s == s_idx and start < e2 and end > s2 for s, s2, e2 in used_spans):
            continue
        selected.append(phrase)
        used_spans.append((s_idx, start, end))
        if len(selected) >= n_queries:
            break
    return selected
