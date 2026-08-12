"""End-to-end оркестрация проверки документа. Общая для API и Streamlit-демо."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from core.chunking import chunk_text
from core.pipeline import ComparisonReport, Source, compare_document
from core.query_gen import generate_queries
from parsing.extract import SUPPORTED_EXTENSIONS, extract_text
from scraping.fetch import fetch_page
from scraping.search import SearchResult, search_phrases

# Magic bytes для проверки, что содержимое соответствует заявленному расширению
_MAGIC_BYTES = {
    ".pdf": [b"%PDF"],
    ".docx": [b"PK\x03\x04"],  # docx — zip-контейнер
}


def _validate_upload(filename: str, content: bytes) -> str:
    """Проверяет расширение (whitelist) и magic bytes. Возвращает suffix."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Неподдерживаемый формат: {suffix or '(без расширения)'}. "
            f"Ожидается: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    magics = _MAGIC_BYTES.get(suffix)
    if magics and not any(content.startswith(m) for m in magics):
        raise ValueError(f"Содержимое файла не соответствует формату {suffix}")
    return suffix


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
    """Обёртка для загруженных файлов: сохраняет во временный файл и зовёт run_check.

    Временный файл удаляется в finally — даже при ошибке парсинга/сети
    содержимое документа не остаётся на диске.
    """
    suffix = _validate_upload(filename, content)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        report = run_check(tmp_path, max_queries=max_queries)
    finally:
        tmp_path.unlink(missing_ok=True)
    report.filename = filename
    return report
