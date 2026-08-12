"""Near-duplicate detection для verbatim: быстрый exact match через n-gram Jaccard.

Используется как pre-filter перед ML-пайплайном: если чанк документа —
буквальная копия чанка источника, ML не нужен (вердикт сразу verbatim).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Пунктуация и пробелы нормализуются перед извлечением n-gram
_NORM_RE = re.compile(r"[^\w\s]")


def char_ngrams(text: str, n: int = 5) -> set[str]:
    """Извлекает character n-gram из нормализованного текста."""
    text = _NORM_RE.sub("", text.lower())
    text = " ".join(text.split())  # collapse whitespace
    if len(text) < n:
        return set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


@dataclass(frozen=True)
class NearDuplicateResult:
    is_duplicate: bool
    jaccard: float
    matched_source_url: str | None = None


class VerbatimMatcher:
    """Near-duplicate matcher на character 5-gram Jaccard.

    Порог 0.6 эмпирически подобран: при полном копировании Jaccard ≈ 1.0,
    при глубоком перефразе — < 0.3, при переформулировании с сохранением
    структуры — 0.3–0.5.
    """

    def __init__(self, n: int = 5, threshold: float = 0.6):
        self._n = n
        self._threshold = threshold

    def grams(self, text: str) -> set[str]:
        return char_ngrams(text, self._n)

    def check(
        self,
        doc_chunk: str,
        source_chunks: list[tuple[str, str]],
        source_grams: list[tuple[str, set[str]]] | None = None,
    ) -> NearDuplicateResult:
        """Сравнивает чанк документа с чанками источников.

        Args:
            doc_chunk: текст чанка документа
            source_chunks: список (url, text) чанков источников
            source_grams: опционально предпосчитанные (url, n-grams) — при
                проверке многих чанков документа против одних и тех же источников
                граммы строятся один раз, а не на каждый чанк (было O(n_doc × n_src))

        Returns:
            NearDuplicateResult с флагом дубликата и лучшим Jaccard
        """
        if source_grams is None:
            source_grams = [(url, self.grams(t)) for url, t in source_chunks]
        doc_grams = self.grams(doc_chunk)
        if not doc_grams:
            return NearDuplicateResult(False, 0.0)

        best_jaccard = 0.0
        best_url = None
        for url, src_grams in source_grams:
            if not src_grams:
                continue
            intersection = len(doc_grams & src_grams)
            union = len(doc_grams | src_grams)
            jaccard = intersection / union if union > 0 else 0.0
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_url = url

        return NearDuplicateResult(
            is_duplicate=best_jaccard >= self._threshold,
            jaccard=round(best_jaccard, 3),
            matched_source_url=best_url if best_jaccard >= self._threshold else None,
        )
