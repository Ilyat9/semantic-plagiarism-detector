"""Тесты на hardening-правки из код-ревью.

Покрывают: батч-независимость лексического скора, валидацию загрузок,
guard пустых источников в пайплайне, 429 (а не 500) в rate limiting,
рендер PDF-шаблона из данных БД (verdict — строка, не enum).
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.ratelimit import _store
from core.classify import Verdict
from core.pipeline import Source, compare_document
from core.service import _validate_upload
from core.similarity import JaccardScorer

TEXT_A = "Photosynthesis converts sunlight into chemical energy in plants."
TEXT_B = "Plants transform sunlight into usable chemical energy through photosynthesis."


class TestJaccardScorer:
    def test_batch_independence(self):
        """Скор пары не зависит от состава батча (в отличие от TF-IDF)."""
        scorer = JaccardScorer()
        alone = scorer.score_pairs([(TEXT_A, TEXT_B)])[0]
        filler = [
            (f"unrelated text number {i} about stocks", f"another disjoint text {i} on cooking")
            for i in range(64)
        ]
        in_batch = scorer.score_pairs(filler + [(TEXT_A, TEXT_B)])[-1]
        assert alone == in_batch

    def test_identical_pair_is_one(self):
        assert JaccardScorer().score_pairs([(TEXT_A, TEXT_A)])[0] == 1.0

    def test_disjoint_pair_is_zero(self):
        score = JaccardScorer().score_pairs([("aaaa bbbb cccc", "xxxx yyyy zzzz")])[0]
        assert score == 0.0


class TestUploadValidation:
    def test_rejects_unknown_extension(self):
        with pytest.raises(ValueError, match="Неподдерживаемый формат"):
            _validate_upload("evil.exe", b"MZ\x90\x00")

    def test_rejects_fake_pdf(self):
        with pytest.raises(ValueError, match="не соответствует"):
            _validate_upload("doc.pdf", b"this is not a pdf")

    def test_accepts_pdf_magic(self):
        assert _validate_upload("doc.pdf", b"%PDF-1.7\n...") == ".pdf"

    def test_accepts_docx_magic(self):
        assert _validate_upload("doc.docx", b"PK\x03\x04...") == ".docx"

    def test_accepts_txt(self):
        assert _validate_upload("notes.txt", "текст".encode()) == ".txt"


class TestPipelineGuards:
    def test_sources_without_chunks_yield_all_original(self):
        """Источники, не давшие ни одного чанка, не роняют пайплайн."""
        doc = "First sentence is here. Second sentence follows it. Third one completes the text."
        sources = [Source(url="http://example.com", text="   \n  ")]
        report = compare_document(doc, sources)
        assert report.similarity_percent == 0.0
        assert all(f.verdict == Verdict.ORIGINAL for f in report.fragments)


class TestRateLimit:
    def test_returns_429_not_500(self):
        """BaseHTTPMiddleware: raise HTTPException давал 500 — теперь JSONResponse 429."""
        _store.clear()
        client = TestClient(app)
        for _ in range(60):  # дефолтный лимит: 60/мин на путь
            assert client.get("/").status_code == 200
        resp = client.get("/")
        assert resp.status_code == 429
        assert "retry-after" in resp.headers
        _store.clear()


class TestPdfTemplate:
    def test_renders_string_verdicts_from_db(self):
        """Регрессия: отчёт из БД (verdict — строка) не должен терять вердикты.

        Раньше шаблон обращался к verdict.value: для строки это Undefined,
        original-фрагменты попадали в Flagged, а метки/CSS-классы были пустыми.
        """
        from jinja2 import Environment, PackageLoader, select_autoescape

        env = Environment(
            loader=PackageLoader("report", "templates"),
            autoescape=select_autoescape(["html"]),
        )
        frag = NS(
            chunk_index=0,
            chunk_text="copied fragment",
            verdict="verbatim",
            lexical=1.0,
            semantic=0.95,
            source_url=None,
            source_excerpt=None,
        )
        original = NS(
            chunk_index=1,
            chunk_text="own text",
            verdict="original",
            lexical=0.1,
            semantic=0.2,
            source_url=None,
            source_excerpt=None,
        )
        report = NS(
            filename="doc.pdf",
            created_at=None,
            n_words=10,
            n_chunks=2,
            queries=[],
            comparison=NS(
                similarity_percent=50.0, fragments=[frag, original], unreachable_sources=[]
            ),
        )
        html = env.get_template("report.html").render(report=report)
        assert 'class="frag verbatim"' in html
        assert "No matching fragments found" not in html
        # original-фрагмент не попадает в секцию Flagged
        flagged_section = html.split("Flagged Fragments")[1].split("All Fragments")[0]
        assert "own text" not in flagged_section
