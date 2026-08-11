"""Персистентное хранение отчётов в SQLite.

Заменяет in-memory dict на SQLAlchemy ORM с пагинацией.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, func
from sqlalchemy.orm import DeclarativeBase, Session, mapped_column

DB_PATH = Path("data/reports.db")


class Base(DeclarativeBase):
    pass


class ReportModel(Base):
    __tablename__ = "reports"

    id = mapped_column(primary_key=True)
    report_uuid = mapped_column(unique=True, index=True)
    filename = mapped_column()
    created_at = mapped_column(default=datetime.utcnow)
    n_words = mapped_column(default=0)
    n_chunks = mapped_column(default=0)
    queries_json = mapped_column(default="[]")
    comparison_json = mapped_column(default="null")
    doc_text = mapped_column(default="")


def get_engine():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{DB_PATH}", echo=False)


def init_db():
    Base.metadata.create_all(get_engine())


def save_report(report_uuid: str, report) -> None:
    """Сохраняет FullReport в SQLite."""
    init_db()
    with Session(get_engine()) as session:
        # Удаляем старый если есть
        session.query(ReportModel).filter_by(report_uuid=report_uuid).delete()
        comp = report.comparison
        row = ReportModel(
            report_uuid=report_uuid,
            filename=report.filename,
            n_words=report.n_words,
            n_chunks=report.n_chunks,
            queries_json=json.dumps(report.queries, ensure_ascii=False),
            comparison_json=json.dumps(
                {
                    "similarity_percent": comp.similarity_percent if comp else 0.0,
                    "fragments": [
                        {
                            "chunk_index": f.chunk_index,
                            "chunk_text": f.chunk_text,
                            "verdict": f.verdict,
                            "lexical": f.lexical,
                            "semantic": f.semantic,
                            "source_url": f.source_url,
                            "source_excerpt": f.source_excerpt,
                        }
                        for f in (comp.fragments if comp else [])
                    ],
                    "unreachable_sources": [
                        {"url": s.url, "error": s.error}
                        for s in (comp.unreachable_sources if comp else [])
                    ],
                    "n_chunks": comp.n_chunks if comp else 0,
                },
                ensure_ascii=False,
            ),
            doc_text=report.doc_text[:50000],  # лимит 50KB
        )
        session.add(row)
        session.commit()


def get_report(report_uuid: str):
    """Возвращает dict с данными отчёта или None."""
    init_db()
    with Session(get_engine()) as session:
        row = session.query(ReportModel).filter_by(report_uuid=report_uuid).first()
        if row is None:
            return None
        return {
            "report_uuid": row.report_uuid,
            "filename": row.filename,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "n_words": row.n_words,
            "n_chunks": row.n_chunks,
            "queries": json.loads(row.queries_json),
            "comparison": json.loads(row.comparison_json),
            "doc_text": row.doc_text,
        }


def list_reports(limit: int = 20, offset: int = 0):
    """Список отчётов с пагинацией."""
    init_db()
    with Session(get_engine()) as session:
        rows = (
            session.query(ReportModel)
            .order_by(ReportModel.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        return [
            {
                "report_uuid": r.report_uuid,
                "filename": r.filename,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "n_words": r.n_words,
                "n_chunks": r.n_chunks,
                "similarity_percent": (
                    json.loads(r.comparison_json).get("similarity_percent", 0.0)
                    if r.comparison_json
                    else 0.0
                ),
            }
            for r in rows
        ]


def get_report_full(report_uuid: str):
    """Возвращает данные отчёта для рендеринга (dict-like объект)."""
    data = get_report(report_uuid)
    if data is None:
        return None
    # SimpleNamespace позволяет обращаться через точку (как dataclass)
    from types import SimpleNamespace

    def _to_ns(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [_to_ns(item) for item in obj]
        return obj

    ns = _to_ns(data)
    # comparison нужен как объект с полями
    if hasattr(ns, "comparison") and ns.comparison:
        ns.comparison = _to_ns(ns.comparison)
    return ns


def count_reports() -> int:
    init_db()
    with Session(get_engine()) as session:
        return session.query(func.count(ReportModel.id)).scalar() or 0
    init_db()
    with Session(get_engine()) as session:
        return session.query(func.count(ReportModel.id)).scalar() or 0
