"""Streamlit-демо поверх core (без дублирования логики).

Запуск: uv run streamlit run demo/app.py
"""

import streamlit as st

from core.classify import Verdict
from core.service import run_check_upload
from report.pdf import render_pdf

st.set_page_config(page_title="Semantic Plagiarism Detection", layout="wide")
st.title("Semantic Plagiarism & Paraphrase Detection")
st.caption("verbatim · paraphrase · original — двухстадийный ML-пайплайн поверх веб-поиска")

uploaded = st.file_uploader("Загрузите документ", type=["docx", "pdf", "txt"])

if uploaded and st.button("Проверить", type="primary"):
    with st.spinner("Поиск источников и ML-сравнение (1–3 минуты)..."):
        report = run_check_upload(uploaded.name, uploaded.getvalue())
    comp = report.comparison

    col1, col2, col3 = st.columns(3)
    col1.metric("Схожесть", f"{comp.similarity_percent}%")
    col2.metric("Чанков", report.n_chunks)
    col3.metric("Недоступных источников", len(comp.unreachable_sources))

    st.download_button(
        "Скачать PDF-отчёт",
        data=render_pdf(report),
        file_name="plagcheck-report.pdf",
        mime="application/pdf",
    )

    flagged = [f for f in comp.fragments if f.verdict != Verdict.ORIGINAL]
    st.subheader(f"Совпадающие фрагменты ({len(flagged)})")
    for f in flagged:
        icon = "🔴" if f.verdict == Verdict.VERBATIM else "🟠"
        with st.expander(f"{icon} {f.verdict} · lex {f.lexical:.2f} · sem {f.semantic:.2f}"):
            st.write(f.chunk_text)
            if f.source_url:
                st.markdown(f"Источник: {f.source_url}")
            if f.source_excerpt:
                st.caption(f"Фрагмент источника: {f.source_excerpt[:400]}...")

    if comp.unreachable_sources:
        st.subheader("Недоступные источники")
        for s in comp.unreachable_sources:
            st.caption(f"{s.url} — {s.error}")

    with st.expander("Поисковые фразы"):
        for q in report.queries:
            st.text(q)
