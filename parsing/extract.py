"""Извлечение текста из .docx/.pdf/.txt с нормализацией."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import docx
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".docx", ".pdf", ".txt"}


def normalize_text(text: str) -> str:
    """NFKC-нормализация и схлопывание пробельных последовательностей."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_docx(path: Path) -> str:
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def _extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[list[str]] = []
    for page in reader.pages:
        lines = [ln.strip() for ln in (page.extract_text() or "").splitlines()]
        pages.append([ln for ln in lines if ln])
    pages = _drop_repeated_lines(pages)
    return "\n".join(ln for page in pages for ln in page)


def _drop_repeated_lines(pages: list[list[str]], threshold: float = 0.5) -> list[list[str]]:
    """Удаляет строки (колонтитулы), встречающиеся на больше чем `threshold` доле страниц."""
    if len(pages) < 3:
        return pages
    freq: dict[str, int] = {}
    for page in pages:
        for ln in set(page):
            freq[ln] = freq.get(ln, 0) + 1
    cutoff = max(2, int(len(pages) * threshold) + 1)
    repeated = {ln for ln, count in freq.items() if count >= cutoff}
    return [[ln for ln in page if ln not in repeated] for page in pages]


def extract_text(path: str | Path) -> str:
    """Извлекает нормализованный текст из поддерживаемого файла."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Неподдерживаемый формат: {ext}. Ожидается: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    raw = {".txt": _extract_txt, ".docx": _extract_docx, ".pdf": _extract_pdf}[ext](path)
    return normalize_text(raw)
