"""Тесты near-duplicate detection для verbatim."""

from __future__ import annotations

import pytest

from core.verbatim_matcher import VerbatimMatcher

MATCHER = VerbatimMatcher()


def test_exact_copy_is_duplicate():
    text = "Photosynthesis converts sunlight into chemical energy in plants."
    result = MATCHER.check(text, [("http://src", text)])
    assert result.is_duplicate is True
    assert result.jaccard == 1.0


def test_near_copy_with_typos_is_duplicate():
    doc = "Photosynthesis converts sunlight into chemical energy in plants."
    src = "Photosynthesis converts sunligt into chemcal energy in plants."
    result = MATCHER.check(doc, [("http://src", src)])
    assert result.is_duplicate is True
    assert result.jaccard > 0.6


def test_paraphrase_is_not_duplicate():
    doc = "Photosynthesis converts sunlight into chemical energy in plants."
    src = "Plants transform sunlight into usable chemical energy through photosynthesis."
    result = MATCHER.check(doc, [("http://src", src)])
    assert result.is_duplicate is False
    assert result.jaccard < 0.6


def test_unrelated_is_not_duplicate():
    doc = "Photosynthesis converts sunlight into chemical energy in plants."
    src = "The stock market closed higher on Tuesday amid tech rallies."
    result = MATCHER.check(doc, [("http://src", src)])
    assert result.is_duplicate is False
    assert result.jaccard < 0.3


def test_empty_doc():
    result = MATCHER.check("", [("http://src", "some text")])
    assert result.is_duplicate is False
    assert result.jaccard == 0.0


def test_best_url_returned():
    doc = "Photosynthesis converts sunlight into chemical energy in plants."
    sources = [
        ("http://irrelevant", "The stock market closed higher today."),
        ("http://correct", "Photosynthesis converts sunlight into chemical energy in plants."),
    ]
    result = MATCHER.check(doc, sources)
    assert result.is_duplicate is True
    assert result.matched_source_url == "http://correct"
