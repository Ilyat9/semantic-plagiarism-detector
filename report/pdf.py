"""Генерация PDF-отчёта через WeasyPrint (HTML -> PDF)."""

from __future__ import annotations

from jinja2 import Environment, PackageLoader, select_autoescape

from core.service import FullReport


def _env() -> Environment:
    return Environment(
        loader=PackageLoader("report", "templates"),
        autoescape=select_autoescape(["html"]),
    )


def render_pdf(report: FullReport) -> bytes:
    """Рендерит FullReport в PDF-байты."""
    from weasyprint import HTML

    html = _env().get_template("report.html").render(report=report)
    return HTML(string=html).write_pdf()
