"""Двухстадийный пайплайн сравнения документа с источниками."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.chunking import chunk_text
from core.classify import Thresholds, Verdict, classify_fragment
from core.similarity import BiEncoderScorer, CrossEncoderScorer, TfidfScorer
from core.verbatim_matcher import VerbatimMatcher

VERBATIM_WEIGHT = 1.0
PARAPHRASE_WEIGHT = 0.5

MAX_SOURCE_CHUNKS = 150  # верхний лимит чанков на источник (скорость stage 1)


@dataclass
class Source:
    url: str
    text: str | None  # None — источник недоступен
    error: str | None = None


@dataclass
class FragmentMatch:
    chunk_index: int
    chunk_text: str
    verdict: Verdict
    lexical: float
    semantic: float
    source_url: str | None = None
    source_excerpt: str | None = None


@dataclass
class ComparisonReport:
    similarity_percent: float
    fragments: list[FragmentMatch] = field(default_factory=list)
    unreachable_sources: list[Source] = field(default_factory=list)
    n_chunks: int = 0


def aggregate_percent(verdicts: list[Verdict]) -> float:
    """(verbatim*1.0 + paraphrase*0.5) / n_chunks * 100. Формула зафиксирована в README."""
    if not verdicts:
        return 0.0
    weights = {Verdict.VERBATIM: VERBATIM_WEIGHT, Verdict.PARAPHRASE: PARAPHRASE_WEIGHT}
    weight = sum(weights.get(v, 0.0) for v in verdicts)
    return round(100.0 * weight / len(verdicts), 1)


def compare_document(
    doc_text: str,
    sources: list[Source],
    bi_scorer: BiEncoderScorer | None = None,
    cross_scorer: CrossEncoderScorer | None = None,
    thresholds: Thresholds | None = None,
    top_k: int = 3,
    window: int = 4,
    overlap: int = 1,
) -> ComparisonReport:
    """Сравнивает документ с источниками и выдаёт вердикт по каждому чанку.

    Stage 1: би-энкодер, косинус по чанкам -> top-k кандидатов на чанк документа.
    Stage 2: кросс-энкодер перепроверяет кандидатов -> semantic score чанка.
    Лексический скор — TF-IDF косинус с лучшим кандидатом.
    """
    thresholds = thresholds or Thresholds.load()
    doc_chunks = chunk_text(doc_text, window=window, overlap=overlap)
    reachable = [s for s in sources if s.text]
    unreachable = [s for s in sources if not s.text]

    if not doc_chunks or not reachable:
        return ComparisonReport(
            similarity_percent=0.0,
            fragments=[
                FragmentMatch(c.index, c.text, Verdict.ORIGINAL, 0.0, 0.0) for c in doc_chunks
            ],
            unreachable_sources=unreachable,
            n_chunks=len(doc_chunks),
        )

    bi_scorer = bi_scorer or BiEncoderScorer()
    cross_scorer = cross_scorer or CrossEncoderScorer()
    verbatim_matcher = VerbatimMatcher()

    # Чанкуем источники, помня URL каждого чанка
    source_chunk_texts: list[str] = []
    source_chunk_urls: list[str] = []
    for src in reachable:
        for ch in chunk_text(src.text, window=window, overlap=overlap)[:MAX_SOURCE_CHUNKS]:
            source_chunk_texts.append(ch.text)
            source_chunk_urls.append(src.url)

    # Pre-filter: near-duplicate detection (быстрый verbatim)
    source_chunks_for_matcher = list(zip(source_chunk_urls, source_chunk_texts, strict=True))
    verbatim_flags: list[bool] = []
    verbatim_best_urls: list[str | None] = []
    for chunk in doc_chunks:
        result = verbatim_matcher.check(chunk.text, source_chunks_for_matcher)
        verbatim_flags.append(result.is_duplicate)
        verbatim_best_urls.append(result.matched_source_url if result.is_duplicate else None)

    # Stage 1: retrieval (только для чанков, не помеченных как verbatim)
    doc_emb = bi_scorer.encode([c.text for c in doc_chunks], is_query=True)
    src_emb = bi_scorer.encode(source_chunk_texts)
    sim_matrix = doc_emb @ src_emb.T  # [n_doc, n_src], эмбеддинги нормализованы

    # Stage 2: rerank top-k кандидатов каждого чанка
    k = min(top_k, len(source_chunk_texts))
    candidate_idx = np.argpartition(-sim_matrix, k - 1, axis=1)[:, :k]
    pairs: list[tuple[str, str]] = []
    for i in range(len(doc_chunks)):
        for j in candidate_idx[i]:
            pairs.append((doc_chunks[i].text, source_chunk_texts[j]))
    cross_scores = cross_scorer.score_pairs(pairs).reshape(len(doc_chunks), k)

    # Лексический скор против лучшего по кросс-энкодеру кандидата
    best_local = np.argmax(cross_scores, axis=1)
    best_src_idx = candidate_idx[np.arange(len(doc_chunks)), best_local]
    tfidf = TfidfScorer()
    lex_scores = tfidf.score_pairs(
        [(doc_chunks[i].text, source_chunk_texts[best_src_idx[i]]) for i in range(len(doc_chunks))]
    )

    fragments: list[FragmentMatch] = []
    for i, chunk in enumerate(doc_chunks):
        if verbatim_flags[i]:
            # Near-duplicate: сразу verbatim, без ML-скоров
            fragments.append(
                FragmentMatch(
                    chunk_index=chunk.index,
                    chunk_text=chunk.text,
                    verdict=Verdict.VERBATIM,
                    lexical=1.0,
                    semantic=1.0,
                    source_url=verbatim_best_urls[i],
                    source_excerpt=None,
                )
            )
            continue

        semantic = float(cross_scores[i, best_local[i]])
        lexical = float(lex_scores[i])
        verdict = classify_fragment(lexical, semantic, thresholds)
        j = int(best_src_idx[i])
        fragments.append(
            FragmentMatch(
                chunk_index=chunk.index,
                chunk_text=chunk.text,
                verdict=verdict,
                lexical=round(lexical, 3),
                semantic=round(semantic, 3),
                source_url=source_chunk_urls[j] if verdict != Verdict.ORIGINAL else None,
                source_excerpt=source_chunk_texts[j] if verdict != Verdict.ORIGINAL else None,
            )
        )

    return ComparisonReport(
        similarity_percent=aggregate_percent([f.verdict for f in fragments]),
        fragments=fragments,
        unreachable_sources=unreachable,
        n_chunks=len(doc_chunks),
    )
