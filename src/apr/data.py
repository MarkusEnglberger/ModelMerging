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
                    class_balanced: bool, exclude_indices=None) -> List[int]:
    """Draw deterministic example indices, optionally excluding a prior draw."""
    rng = random.Random(seed)
    excluded = set(exclude_indices or [])
    available = [i for i in range(len(train_ds)) if i not in excluded]
    n = min(n, len(available))
    if spec.is_regression or not class_balanced:
        idx = rng.sample(available, n)
    else:
        labels = train_ds["labels"]
        by_label: Dict[int, List[int]] = {}
        for i in available:
            y = labels[i]
            by_label.setdefault(int(y), []).append(i)
        classes = sorted(by_label)
        if len(classes) > n:
            # Fine-grained task whose class count exceeds the budget (Cars has
            # 196 classes, SUN397 has 397). One example per class then yields
            # more indices than the budget, and the trim below would keep the
            # LOWEST class ids -- deterministically, for every seed, since the
            # shuffles here only pick which example within a class and reorder
            # the result. Choose WHICH classes at random instead, so the covered
            # subset varies with the buffer seed. Buffers for tasks with at most
            # n classes are unaffected: this branch does not run for them, so
            # their random stream is untouched.
            classes = sorted(rng.sample(classes, n))
        per = max(1, n // len(classes))
        idx: List[int] = []
        for c in classes:
            pool = by_label[c]
            rng.shuffle(pool)
            idx.extend(pool[:per])
        # top up / trim to exactly n
        if len(idx) < n:
            remaining = [i for i in available if i not in set(idx)]
            rng.shuffle(remaining)
            idx.extend(remaining[: n - len(idx)])
        idx = idx[:n]
        rng.shuffle(idx)
    return idx


def sample_replay_buffer(train_ds, spec: TaskSpec, n: int, seed: int,
                         class_balanced: bool, exclude_indices=None,
                         return_indices: bool = False):
    """Sample replay examples, optionally excluding a previously drawn subset.

    ``return_indices`` lets callers construct and verify disjoint replay and
    selection buffers.
    """
    idx = _sample_indices(train_ds, spec, n, seed, class_balanced,
                          exclude_indices=exclude_indices)
    buffer = [train_ds[i] for i in idx]
    return (buffer, idx) if return_indices else buffer


def sample_replay_buffer_split(train_ds, spec: TaskSpec, n_train: int, n_val: int,
                               seed: int, class_balanced: bool,
                               return_indices: bool = False):
    """Draw n_train+n_val examples with the SAME index stream as
    sample_replay_buffer(n_train+n_val, seed), then split DISJOINTLY.

    Guarantees train/val never overlap (unlike two independent draws with
    different seeds). With n_train=n_val=32 and the usual probe_seed, the union
    is exactly the 64 examples earlier n_probe=64 runs used -- the honest
    '64 labeled samples/task total' budget, now 32 for sweeps + 32 for
    hyperparameter selection."""
    idx = _sample_indices(train_ds, spec, n_train + n_val, seed, class_balanced)
    train_idx, val_idx = idx[:n_train], idx[n_train:n_train + n_val]
    train = [train_ds[i] for i in train_idx]
    val = [train_ds[i] for i in val_idx]
    if return_indices:
        return train, val, train_idx, val_idx
    return train, val


def batches_from_buffer(buffer: List[Dict], collator, batch_size: int,
                        device: str):
    """Yield padded batches from a replay buffer on the target device."""
    for start in range(0, len(buffer), batch_size):
        chunk = buffer[start: start + batch_size]
        batch = collator(chunk)
        yield {k: v.to(device) for k, v in batch.items()}
