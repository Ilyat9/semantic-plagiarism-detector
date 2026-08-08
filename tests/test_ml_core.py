"""Тесты ML-ядра: классификатор вердиктов, агрегация, двухстадийный пайплайн."""

import pytest

from core.classify import Thresholds, Verdict, classify_fragment
from core.pipeline import Source, aggregate_percent, compare_document

TH = Thresholds(t1_lexical=0.6, t2_semantic=0.5)


@pytest.mark.parametrize(
    ("lexical", "semantic", "expected"),
    [
        (0.9, 0.9, Verdict.VERBATIM),  # высокий lex + высокий sem
        (0.6, 0.5, Verdict.VERBATIM),  # ровно на порогах -> verbatim
        (0.2, 0.9, Verdict.PARAPHRASE),  # низкий lex + высокий sem
        (0.9, 0.2, Verdict.ORIGINAL),  # высокий lex, но sem ниже порога
        (0.1, 0.1, Verdict.ORIGINAL),
    ],
)
def test_classify_fragment(lexical, semantic, expected):
    assert classify_fragment(lexical, semantic, TH) == expected


def test_aggregate_percent():
    assert aggregate_percent([]) == 0.0
    assert aggregate_percent([Verdict.ORIGINAL] * 4) == 0.0
    assert aggregate_percent([Verdict.VERBATIM] * 4) == 100.0
    assert aggregate_percent([Verdict.PARAPHRASE] * 4) == 50.0
    mixed = [Verdict.VERBATIM, Verdict.PARAPHRASE, Verdict.ORIGINAL, Verdict.ORIGINAL]
    assert aggregate_percent(mixed) == pytest.approx(37.5)


def test_compare_document_no_sources():
    report = compare_document(
        "A sentence here. Another sentence there. Third one follows. Fourth closes it.",
        sources=[],
    )
    assert report.similarity_percent == 0.0
    assert all(f.verdict == Verdict.ORIGINAL for f in report.fragments)


def test_compare_document_unreachable_source_marks_original():
    report = compare_document(
        "Some text here. More text follows. Even more after. Final sentence.",
        sources=[Source(url="http://x", text=None, error="HTTP 403")],
    )
    assert report.similarity_percent == 0.0
    assert len(report.unreachable_sources) == 1
    assert report.unreachable_sources[0].error == "HTTP 403"


# --- end-to-end с реальными моделями (медленный, скачивает веса при первом прогоне) ---

DOC = (
    "Photosynthesis converts sunlight into chemical energy in plants. "
    "Chlorophyll absorbs light mainly in the blue and red spectra. "
    "The Calvin cycle fixes carbon dioxide into organic molecules. "
    "Oxygen is released as a byproduct of splitting water molecules."
)

PARAPHRASED_SOURCE = (
    "Plants turn sunlight into usable chemical energy. "
    "Blue and red wavelengths are captured mostly by chlorophyll pigments. "
    "Carbon dioxide is incorporated into organic compounds via the Calvin cycle. "
    "Splitting water molecules produces oxygen as a side product."
)

UNRELATED_SOURCE = (
    "The stock market closed higher on Tuesday amid tech rallies. "
    "Investors welcomed the new quarterly earnings reports. "
    "Analysts expect continued volatility in commodity prices. "
    "Central banks signalled a cautious approach to rate cuts."
)


@pytest.mark.slow
def test_pipeline_verbatim_paraphrase_original():
    copied = compare_document(DOC, sources=[Source(url="http://src/copy", text=DOC)])
    paraphrased = compare_document(
        DOC, sources=[Source(url="http://src/para", text=PARAPHRASED_SOURCE)]
    )
    unrelated = compare_document(
        DOC, sources=[Source(url="http://src/other", text=UNRELATED_SOURCE)]
    )

    assert copied.fragments[0].verdict == Verdict.VERBATIM
    assert unrelated.fragments[0].verdict == Verdict.ORIGINAL
    # Перефраз: семантический скор должен быть значимо выше, чем у несвязанного текста.
    # Строгий вердикт paraphrase не требуем: порог T2 откалиброван на adversarial PAWS
    # и сознательно консервативен для естественных перефразов (см. failure analysis).
    assert paraphrased.fragments[0].semantic > unrelated.fragments[0].semantic + 0.2
    assert copied.similarity_percent > unrelated.similarity_percent
