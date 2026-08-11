"""Разбиение текста на пересекающиеся чанки из предложений."""

from __future__ import annotations

from dataclasses import dataclass

_nlp_cache: dict[str, object] = {}


def _get_nlp(lang: str = "en"):
    """Ленивая загрузка spaCy sentencizer (без тяжёлого pipeline)."""
    if lang not in _nlp_cache:
        import spacy

        _nlp_cache[lang] = spacy.blank(lang)
        _nlp_cache[lang].add_pipe("sentencizer")
    return _nlp_cache[lang]


@dataclass
class Chunk:
    index: int
    text: str
    start_sentence: int
    end_sentence: int  # exclusive


def split_sentences(text: str, lang: str = "en") -> list[str]:
    """Разбивает текст на предложения."""
    nlp = _get_nlp(lang)
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def chunk_text(
    text: str,
    window: int = 4,
    overlap: int = 1,
    min_words: int = 8,
    lang: str = "en",
) -> list[Chunk]:
    """Sliding window по предложениям: `window` предложений, overlap `overlap`.

    Слишком короткие хвостовые чанки (< min_words слов) сливаются с предыдущим.
    """
    sentences = split_sentences(text, lang=lang)
    if not sentences:
        return []
    window = max(1, min(window, len(sentences)))
    step = max(1, window - overlap)

    chunks: list[Chunk] = []
    start = 0
    while start < len(sentences):
        end = min(start + window, len(sentences))
        chunks.append(Chunk(len(chunks), " ".join(sentences[start:end]), start, end))
        if end == len(sentences):
            break
        start += step

    if len(chunks) > 1 and len(chunks[-1].text.split()) < min_words:
        tail = chunks.pop()
        prev = chunks[-1]
        chunks[-1] = Chunk(
            prev.index, prev.text + " " + tail.text, prev.start_sentence, tail.end_sentence
        )
    return chunks
