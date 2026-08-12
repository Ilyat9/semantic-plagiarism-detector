"""FastAPI-приложение: загрузка документа, проверка, HTML- и PDF-отчёт."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

import config
import db
from api.ratelimit import RateLimitMiddleware
from core.service import run_check_upload
from report.pdf import render_pdf

app = FastAPI(title="Semantic Plagiarism & Paraphrase Detection")
app.add_middleware(RateLimitMiddleware)
templates = Jinja2Templates(directory="api/templates")


@app.get("/", response_class=HTMLResponse)
async def index():
    return templates.get_template("index.html").render()


@app.post("/check", response_class=HTMLResponse)
def check(request: Request, file: Annotated[UploadFile, File()]):
    """Принимает документ и запускает проверку.

    Эндпоинт намеренно синхронный (def, а не async def): пайплайн проверки —
    блокирующий (сеть, скрапинг, CPU-инференс на 1–3 минуты), и FastAPI уводит
    его в threadpool, не блокируя event loop.
    """
    # Ранний отказ по Content-Length, чтобы не читать заведомо большие файлы
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Файл слишком большой")
    content = file.file.read(config.MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Пустой файл")
    if len(content) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Файл слишком большой")
    try:
        report = run_check_upload(file.filename or "document", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    report_id = uuid.uuid4().hex[:12]
    db.save_report(report_id, report)
    return templates.get_template("result.html").render(report=report, report_id=report_id)


@app.get("/report/{report_id}.pdf")
def report_pdf(report_id: str):
    """PDF-отчёт по id. report_id — bearer-идентификатор: ссылку видит только
    тот, кому её показали после проверки (публичного листинга отчётов нет)."""
    report = db.get_report_full(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Отчёт не найден")
    pdf_bytes = render_pdf(report)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="report-{report_id}.pdf"'},
    )
