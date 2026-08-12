"""Fine-tuning cross-encoder на PAWS для повышения F1.

Использование:
    uv run python -m eval.finetune [--epochs 2] [--samples 3000] [--output dir]

Сохраняет fine-tuned модель в data/models/cross-encoder-paws-finetuned/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

from eval.datasets import load_stsb
from eval.run import eval_binary, eval_stsb, measure_latency


def load_paws_examples(split: str = "train", max_samples: int | None = None) -> list[InputExample]:
    """Загружает PAWS и конвертирует в InputExample для CrossEncoder.fit()."""
    ds = load_dataset("google-research-datasets/paws", "labeled_final", split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    examples = []
    for row in ds:
        # label 1 = перефраз, 0 = не перефраз
        # CrossEncoder ожидает label как float (0..1)
        label = float(row["label"])
        examples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=label))
    return examples


def finetune(
    output_dir: Path,
    base_model: str = "cross-encoder/stsb-roberta-base",
    epochs: int = 2,
    batch_size: int = 16,
    max_samples: int | None = 3000,
    warmup_steps: int = 100,
) -> Path:
    """Дообучает cross-encoder на PAWS train."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Загрузка PAWS train (макс {max_samples} примеров)...", flush=True)
    train_examples = load_paws_examples("train", max_samples=max_samples)
    print(f"Обучающих примеров: {len(train_examples)}", flush=True)

    print(f"Загрузка базовой модели: {base_model}", flush=True)
    model = CrossEncoder(base_model, device="cpu")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    print(f"Обучение: {epochs} эпох, batch={batch_size}, warmup={warmup_steps}...", flush=True)
    model.fit(
        train_dataloader=train_dataloader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        show_progress_bar=True,
    )

    # fit() в новых версиях sentence-transformers не сохраняет финальную модель
    # в output_path — сохраняем явно (иначе каталог остаётся пустым).
    model.save(str(output_dir))
    print(f"Fine-tuned модель сохранена: {output_dir}", flush=True)
    return output_dir


def evaluate_finetuned(model_path: Path) -> dict:
    """Оценивает fine-tuned модель на PAWS test + STS-B."""
    import numpy as np

    from core.similarity import CrossEncoderScorer
    from eval.datasets import load_paws

    print("\nОценка fine-tuned модели...", flush=True)
    paws_dev = load_paws("validation", limit=1000)
    paws_test = load_paws("test", limit=2000)
    stsb = load_stsb()

    # Модель загружается ОДИН раз (раньше грузилась заново на каждый батч —
    # ~1 с на roberta-base, что делало заявленную latency недостижимой).
    # Нормализация — та же, что в продакшене (CrossEncoderScorer): после fit()
    # с дефолтным BCEWithLogitsLoss голова выдаёт логиты -> sigmoid, а не clip.
    ce_scorer = CrossEncoderScorer(str(model_path))

    # Контроль диапазона сырых выходов (урок из failure analysis #3):
    # перед нормализацией проверяем, что модель выдаёт то, что мы ожидаем.
    raw = np.asarray(ce_scorer._model.predict(paws_test.pairs[:50]), dtype=float)
    print(f"Сырые выходы predict() на 50 парах: min={raw.min():.3f}, max={raw.max():.3f}")

    row = {"approach": "full-ft"}
    row.update(eval_binary(ce_scorer.score_pairs, paws_dev, paws_test))
    row.update(eval_stsb(ce_scorer.score_pairs, stsb))
    row.update(measure_latency(ce_scorer.score_pairs, paws_test.pairs))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=3000, help="макс примеров из PAWS train")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output", type=Path, default=Path("data/models/cross-encoder-paws-finetuned")
    )
    parser.add_argument("--evaluate", action="store_true", help="сразу оценить после обучения")
    args = parser.parse_args()

    model_path = finetune(
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.samples,
    )

    if args.evaluate:
        row = evaluate_finetuned(model_path)
        print("\nРезультаты fine-tuned модели:")
        for k, v in row.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
