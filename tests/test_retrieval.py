"""Тесты качества retrieval: находит ли би-энкодер релевантные чанки в top-k.

Проверяет ключевую инвариант системы: если Stage 1 (retrieval) промахивается,
Stage 2 (cross-encoder) не спасает — потому что не видит релевантный чанк.
"""

from __future__ import annotations

import pytest

from core.chunking import chunk_text
from core.similarity import BiEncoderScorer

# Фиксированный документ с хорошо известной темой
DOC_PHOTOSYNTHESIS = (
    "Photosynthesis converts sunlight into chemical energy in plants. "
    "Chlorophyll absorbs light mainly in the blue and red spectra. "
    "The Calvin cycle fixes carbon dioxide into organic molecules. "
    "Oxygen is released as a byproduct of splitting water molecules."
)

# Источники разной релевантности
SOURCE_RELEVANT = (
    "Photosynthesis is the process by which plants convert light energy into chemical energy. "
    "Chlorophyll pigments capture photons primarily in blue and red wavelengths. "
    "Through the Calvin cycle, carbon dioxide is assimilated into sugar molecules. "
    "Water photolysis produces molecular oxygen as a waste product."
)

SOURCE_PARTIAL = (
    "Cellular respiration breaks down glucose to produce ATP energy. "
    "Mitochondria are the powerhouses of eukaryotic cells. "
    "The Krebs cycle generates electron carriers for oxidative phosphorylation."
)

SOURCE_UNRELATED = (
    "The stock market closed higher amid technology sector rallies. "
    "Investors welcomed stronger quarterly earnings reports. "
    "Analysts expect continued volatility in commodity prices."
)


@pytest.fixture(scope="module")
def bi_scorer():
    return BiEncoderScorer()


def _recall_at_k(doc_chunks, src_chunks, bi_scorer, k: int = 3) -> float:
    """Для каждого чанка документа проверяет, есть ли идеальный матч (copy) в top-k."""
    if not doc_chunks or not src_chunks:
        return 0.0
    doc_emb = bi_scorer.encode([c.text for c in doc_chunks], is_query=True)
    src_emb = bi_scorer.encode([c.text for c in src_chunks])
    # cosine similarity matrix [n_doc, n_src]
    sims = doc_emb @ src_emb.T

    hits = 0
    for i in range(len(doc_chunks)):
        top_k_idx = sims[i].argsort()[-k:][::-1]
        # Считаем хитом, если top-k содержит чанк с semantic overlap
        # (упрощённо: если max similarity в top-k > 0.7)
        if sims[i, top_k_idx].max() > 0.7:
            hits += 1
    return hits / len(doc_chunks)


@pytest.mark.slow
def test_retrieval_relevant_source_in_topk(bi_scorer):
    """Релевантный источник: каждый чанк документа должен иметь близкий чанк в top-3."""
    doc_chunks = chunk_text(DOC_PHOTOSYNTHESIS, window=2, overlap=0)
    src_chunks = chunk_text(SOURCE_RELEVANT, window=2, overlap=0)
    recall = _recall_at_k(doc_chunks, src_chunks, bi_scorer, k=3)
    assert recall >= 0.8, f"recall@3 = {recall:.2f}, ожидалось ≥ 0.8"


@pytest.mark.slow
def test_retrieval_partial_source_low_recall(bi_scorer):
    """Частично релевантный источник: recall должен быть ниже, чем у релевантного."""
    doc_chunks = chunk_text(DOC_PHOTOSYNTHESIS, window=2, overlap=0)
    src_chunks = chunk_text(SOURCE_PARTIAL, window=2, overlap=0)
    recall = _recall_at_k(doc_chunks, src_chunks, bi_scorer, k=3)
    # Частичная релевантность → recall ниже, чем у полной
    assert recall < 0.8, f"recall@3 = {recall:.2f}, ожидалось < 0.8 для частичной релевантности"


@pytest.mark.slow
def test_retrieval_unrelated_source_near_zero(bi_scorer):
    """Нерелевантный источник: максимальная similarity должна быть низкой."""
    doc_chunks = chunk_text(DOC_PHOTOSYNTHESIS, window=2, overlap=0)
    src_chunks = chunk_text(SOURCE_UNRELATED, window=2, overlap=0)
    doc_emb = bi_scorer.encode([c.text for c in doc_chunks], is_query=True)
    src_emb = bi_scorer.encode([c.text for c in src_chunks])
    sims = doc_emb @ src_emb.T
    max_sim = sims.max()
    assert max_sim < 0.6, f"max similarity = {max_sim:.3f}, ожидалось < 0.6"
