"""Классификация фрагмента по двум скорам: verbatim / paraphrase / original."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

_THRESHOLDS_PATH = Path(__file__).parent / "thresholds.json"

# Стартовые пороги до калибровки по PAWS-dev (см. eval/calibrate.py)
DEFAULT_T1_LEXICAL = 0.6
DEFAULT_T2_SEMANTIC = 0.5


class Verdict(StrEnum):
    VERBATIM = "verbatim"
    PARAPHRASE = "paraphrase"
    ORIGINAL = "original"


@dataclass(frozen=True)
class Thresholds:
    t1_lexical: float = DEFAULT_T1_LEXICAL
    t2_semantic: float = DEFAULT_T2_SEMANTIC

    @classmethod
    def load(cls, path: Path = _THRESHOLDS_PATH) -> Thresholds:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(t1_lexical=data["t1_lexical"], t2_semantic=data["t2_semantic"])
        return cls()

    def save(self, path: Path = _THRESHOLDS_PATH) -> None:
        path.write_text(
            json.dumps({"t1_lexical": self.t1_lexical, "t2_semantic": self.t2_semantic}, indent=2),
            encoding="utf-8",
        )


def classify_fragment(
    lexical: float, semantic: float, thresholds: Thresholds | None = None
) -> Verdict:
    """Прозрачные правила поверх двух скоров (пороги калибруются по PAWS-dev)."""
    th = thresholds or Thresholds.load()
    if semantic >= th.t2_semantic:
        return Verdict.VERBATIM if lexical >= th.t1_lexical else Verdict.PARAPHRASE
    return Verdict.ORIGINAL
