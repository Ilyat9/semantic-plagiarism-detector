"""Тесты извлечения текста, чанкинга и генерации поисковых фраз."""

import pytest

from core.chunking import chunk_text, split_sentences
from core.query_gen import generate_queries
from parsing.extract import _drop_repeated_lines, extract_text, normalize_text

SAMPLE = (
    "Machine learning is a field of artificial intelligence. "
    "It focuses on building systems that learn from data. "
    "Gradient descent optimizes model parameters iteratively. "
    "Overfitting happens when a model memorizes noise. "
    "Regularization techniques help prevent overfitting. "
    "Cross-validation estimates generalization performance reliably. "
    "Neural networks approximate complex nonlinear functions. "
    "Transformers revolutionized natural language processing recently. "
    "Attention mechanisms weight relevant tokens dynamically. "
    "Embeddings map words into dense vector spaces. "
    "Cosine similarity measures angle between embedding vectors. "
    "Semantic search retrieves documents by meaning rather than keywords."
)


def test_normalize_text_collapses_spaces_and_nfkc():
    assert normalize_text("a  \t b\n\n\n\nc") == "a b\n\nc"
    assert normalize_text("ﬁle") == "file"  # лигатура fi -> fi


def test_drop_repeated_lines_removes_headers():
    pages = [
        ["Header Inc.", "first content", "Page 1"],
        ["Header Inc.", "second content", "Page 2"],
        ["Header Inc.", "third content", "Page 3"],
    ]
    cleaned = _drop_repeated_lines(pages)
    flat = [ln for page in cleaned for ln in page]
    assert "Header Inc." not in flat
    assert "first content" in flat


def test_extract_txt(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("Hello   world.\n\n\n\nSecond  paragraph.", encoding="utf-8")
    assert extract_text(p) == "Hello world.\n\nSecond paragraph."


def test_extract_unsupported(tmp_path):
    p = tmp_path / "doc.xlsx"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Неподдерживаемый формат"):
        extract_text(p)


def test_split_sentences():
    sents = split_sentences(SAMPLE)
    assert len(sents) == 12
    assert sents[0].endswith("intelligence.")


def test_chunk_text_window_and_overlap():
    chunks = chunk_text(SAMPLE, window=4, overlap=1)
    assert chunks[0].start_sentence == 0
    assert chunks[0].end_sentence == 4
    assert chunks[1].start_sentence == 3  # шаг = window - overlap
    # индексы последовательны, покрытие до конца текста
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[-1].end_sentence == 12


def test_chunk_text_merges_short_tail():
    text = " ".join(f"Sentence number {i} has several words in it." for i in range(5))
    chunks = chunk_text(text, window=4, overlap=1, min_words=8)
    # хвост из 1 предложения (8 слов) не должен остаться отдельным коротким чанком
    assert all(len(c.text.split()) >= 8 for c in chunks)


def test_generate_queries_count_and_length():
    queries = generate_queries(SAMPLE, min_queries=5, max_queries=10, window=7)
    assert 1 <= len(queries) <= 10
    for q in queries:
        assert 5 <= len(q.split()) <= 10


def test_generate_queries_no_overlap():
    queries = generate_queries(SAMPLE, min_queries=5, max_queries=20, window=7)
    seen = set()
    for q in queries:
        words = tuple(q.split())
        assert words not in seen
        seen.add(words)


def test_generate_queries_prefers_distinctive():
    text = (
        "The cat sat on the mat. The dog sat on the rug. "
        "Photosynthetic cyanobacteria dominated Precambrian shallow oceans. "
        "The sun is bright. The sky is blue."
    )
    queries = generate_queries(text, min_queries=1, max_queries=1, window=5)
    assert any("cyanobacteria" in q or "Precambrian" in q for q in queries)
