"""Загрузка eval-датасетов: PAWS (labeled_final) и STS-B."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PairDataset:
    name: str
    pairs: list[tuple[str, str]]
    labels: list[float]  # PAWS: 0/1; STS-B: 0..5


def load_paws(split: str = "test", limit: int | None = None) -> PairDataset:
    """PAWS labeled_final: label 1 = перефраз. Сплиты: train/validation/test."""
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/paws", "labeled_final", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    pairs = [(r["sentence1"], r["sentence2"]) for r in ds]
    labels = [float(r["label"]) for r in ds]
    return PairDataset(f"PAWS-{split}", pairs, labels)


def load_stsb(limit: int | None = None) -> PairDataset:
    """STS-B validation (у test-сплита нет публичных меток). Скор 0..5."""
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/glue", "stsb", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    pairs = [(r["sentence1"], r["sentence2"]) for r in ds]
    labels = [float(r["label"]) for r in ds]
    return PairDataset("STS-B-validation", pairs, labels)
