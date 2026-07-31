"""FastAPI-приложение: загрузка документа, проверка, HTML- и PDF-отчёт."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

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
async def check(file: Annotated[UploadFile, File()]):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Пустой файл")
    try:
        report = run_check_upload(file.filename or "document", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    report_id = uuid.uuid4().hex[:12]
    db.save_report(report_id, report)
    return templates.get_template("result.html").render(report=report, report_id=report_id)


@app.get("/report/{report_id}.pdf")
async def report_pdf(report_id: str):
    report = db.get_report_full(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Отчёт не найден")
    pdf_bytes = render_pdf(report)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="report-{report_id}.pdf"'},
    )


@app.get("/reports")
async def list_reports(limit: int = 20, offset: int = 0):
    """Список проверенных документов с пагинацией."""
    return {
        "total": db.count_reports(),
        "limit": limit,
        "offset": offset,
        "items": db.list_reports(limit=limit, offset=offset),
    }
