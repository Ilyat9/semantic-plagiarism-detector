"""End-to-end оркестрация проверки документа. Общая для API и Streamlit-демо."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from core.chunking import chunk_text
from core.pipeline import ComparisonReport, Source, compare_document
from core.query_gen import generate_queries
from parsing.extract import extract_text
from scraping.fetch import fetch_page
from scraping.search import SearchResult, search_phrases


@dataclass
class FullReport:
    filename: str
    doc_text: str
    n_words: int
    n_chunks: int
    queries: list[str]
    search_results: list[SearchResult] = field(default_factory=list)
    comparison: ComparisonReport | None = None


def run_check(file_path: str | Path, max_queries: int = 15) -> FullReport:
    """Полный прогон: парсинг -> фразы -> поиск -> скрапинг -> сравнение."""
    file_path = Path(file_path)
    doc_text = extract_text(file_path)
    n_words = len(doc_text.split())
    n_chunks = len(chunk_text(doc_text))
    queries = generate_queries(doc_text, max_queries=max_queries)

    search_results = search_phrases(queries)
    urls: dict[str, None] = {}  # уникальные URL, порядок сохранён
    for r in search_results:
        for u in r.urls:
            urls.setdefault(u)

    sources = []
    for url in urls:
        page = fetch_page(url)
        sources.append(Source(url=url, text=page.text, error=page.error))

    comparison = compare_document(doc_text, sources)
    return FullReport(
        filename=file_path.name,
        doc_text=doc_text,
        n_words=n_words,
        n_chunks=n_chunks,
        queries=queries,
        search_results=search_results,
        comparison=comparison,
    )


def run_check_upload(filename: str, content: bytes, max_queries: int = 15) -> FullReport:
    """Обёртка для загруженных файлов: сохраняет во временный файл и зовёт run_check."""
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    report = run_check(tmp_path, max_queries=max_queries)
    report.filename = filename
    tmp_path.unlink(missing_ok=True)
    return report
