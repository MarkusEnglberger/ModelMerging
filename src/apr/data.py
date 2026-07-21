"""Data loading, tokenization, and replay-buffer sampling.

The replay buffer (D^probe) is small (n in {4,8,...,128}) and is drawn from the
*training* split so the validation split used for reporting stays untouched. For
classification tasks we draw a class-balanced sample when possible.
"""

from typing import Dict, List, Optional
import random

import torch
from datasets import load_dataset

from .tasks import TaskSpec


def _tokenize_fn(tokenizer, spec: TaskSpec, max_length: int):
    keys = spec.text_keys

    def fn(batch):
        if len(keys) == 1:
            enc = tokenizer(batch[keys[0]], truncation=True, max_length=max_length)
        else:
            enc = tokenizer(batch[keys[0]], batch[keys[1]],
                            truncation=True, max_length=max_length)
        return enc

    return fn


def load_task_dataset(spec: TaskSpec, tokenizer, max_length: int,
                      cache_dir: Optional[str] = None):
    """Return tokenized (train, eval) HF datasets with a `labels` column."""
    ds = load_dataset("glue", spec.glue_config, cache_dir=cache_dir)
    tok = _tokenize_fn(tokenizer, spec, max_length)
    cols_to_remove = [c for c in ds["train"].column_names if c != spec.label_key]
    ds = ds.map(tok, batched=True, remove_columns=cols_to_remove)
    ds = ds.rename_column(spec.label_key, "labels")
    train = ds["train"]
    eval_ = ds[spec.eval_split]
    return train, eval_


def _collate(features: List[Dict], tokenizer, is_regression: bool):
    labels = [f["labels"] for f in features]
    input_feats = [{k: f[k] for k in f if k != "labels"} for f in features]
    batch = tokenizer.pad(input_feats, return_tensors="pt")
    batch["labels"] = torch.tensor(
        labels, dtype=torch.float if is_regression else torch.long)
    return batch


def make_collator(tokenizer, is_regression: bool):
    return lambda feats: _collate(feats, tokenizer, is_regression)


def _sample_indices(train_ds, spec: TaskSpec, n: int, seed: int,
                    class_balanced: bool) -> List[int]:
    """Draw n example indices (class-balanced when asked). Deterministic in seed."""
    rng = random.Random(seed)
    n = min(n, len(train_ds))
    if spec.is_regression or not class_balanced:
        idx = rng.sample(range(len(train_ds)), n)
    else:
        labels = train_ds["labels"]
        by_label: Dict[int, List[int]] = {}
        for i, y in enumerate(labels):
            by_label.setdefault(int(y), []).append(i)
        classes = sorted(by_label)
        per = max(1, n // len(classes))
        idx: List[int] = []
        for c in classes:
            pool = by_label[c]
            rng.shuffle(pool)
            idx.extend(pool[:per])
        # top up / trim to exactly n
        if len(idx) < n:
            remaining = [i for i in range(len(train_ds)) if i not in set(idx)]
            rng.shuffle(remaining)
            idx.extend(remaining[: n - len(idx)])
        idx = idx[:n]
        rng.shuffle(idx)
    return idx


def sample_replay_buffer(train_ds, spec: TaskSpec, n: int, seed: int,
                         class_balanced: bool) -> List[Dict]:
    """Sample n replay examples (as a list of tokenized feature dicts)."""
    return [train_ds[i] for i in _sample_indices(train_ds, spec, n, seed,
                                                 class_balanced)]


def sample_replay_buffer_split(train_ds, spec: TaskSpec, n_train: int, n_val: int,
                               seed: int, class_balanced: bool):
    """Draw n_train+n_val examples with the SAME index stream as
    sample_replay_buffer(n_train+n_val, seed), then split DISJOINTLY.

    Guarantees train/val never overlap (unlike two independent draws with
    different seeds). With n_train=n_val=32 and the usual probe_seed, the union
    is exactly the 64 examples earlier n_probe=64 runs used -- the honest
    '64 labeled samples/task total' budget, now 32 for sweeps + 32 for
    hyperparameter selection."""
    idx = _sample_indices(train_ds, spec, n_train + n_val, seed, class_balanced)
    train_idx, val_idx = idx[:n_train], idx[n_train:n_train + n_val]
    return ([train_ds[i] for i in train_idx], [train_ds[i] for i in val_idx])


def batches_from_buffer(buffer: List[Dict], collator, batch_size: int,
                        device: str):
    """Yield padded batches from a replay buffer on the target device."""
    for start in range(0, len(buffer), batch_size):
        chunk = buffer[start: start + batch_size]
        batch = collator(chunk)
        yield {k: v.to(device) for k, v in batch.items()}
