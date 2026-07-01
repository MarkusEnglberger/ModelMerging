#!/usr/bin/env python
"""Fine-tune a single-task GLUE expert from a shared pretrained checkpoint.

Used to fill checkpoint gaps (e.g. DistilRoBERTa experts for the smoke test).
All experts trained this way MUST start from the same `--base` so the resulting
task vectors share theta_0.

Usage:
    python scripts/train_expert.py --base distilroberta-base --task mrpc \
        --out experts/distilroberta-mrpc --epochs 3
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          TrainingArguments, Trainer, DataCollatorWithPadding)

from apr.tasks import get_task
from apr.data import load_task_dataset
from apr.metrics import score_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()

    spec = get_task(args.task)
    tokenizer = AutoTokenizer.from_pretrained(args.base, cache_dir=args.cache_dir)
    problem_type = "regression" if spec.is_regression else "single_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=spec.num_labels, problem_type=problem_type,
        cache_dir=args.cache_dir)

    train_ds, eval_ds = load_task_dataset(spec, tokenizer, args.max_length, args.cache_dir)
    collator = DataCollatorWithPadding(tokenizer)

    def compute_metrics(p):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(logits) if spec.is_regression else np.argmax(logits, axis=-1)
        return score_predictions(spec, preds, p.label_ids)

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        learning_rate=args.lr, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64, eval_strategy="epoch", save_strategy="no",
        seed=args.seed, logging_steps=50, report_to=[], fp16=True,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=eval_ds, data_collator=collator,
                      compute_metrics=compute_metrics)
    trainer.train()
    metrics = trainer.evaluate()
    print("[eval]", {k: round(v, 4) for k, v in metrics.items() if "eval_" in k})

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
