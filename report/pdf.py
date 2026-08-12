"""Генерация PDF-отчёта через WeasyPrint (HTML -> PDF)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from jinja2 import Environment, PackageLoader, select_autoescape


@runtime_checkable
class ReportLike(Protocol):
    """Структурный тип отчёта для рендеринга.

    Удовлетворяют и core.service.FullReport (живой dataclass, Streamlit/свежая
    проверка), и объект, восстановленный из БД (db.get_report_full): шаблону
    нужны только эти поля.
    """

    filename: str
    n_words: int
    n_chunks: int
    queries: list[str]
    comparison: Any


def _env() -> Environment:
    return Environment(
        loader=PackageLoader("report", "templates"),
        autoescape=select_autoescape(["html"]),
    )


def render_pdf(report: ReportLike) -> bytes:
    """Рендерит отчёт в PDF-байты."""
    from weasyprint import HTML

    html = _env().get_template("report.html").render(report=report)
    return HTML(string=html).write_pdf()
