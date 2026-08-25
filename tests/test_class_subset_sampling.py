"""Class-balanced sampling when a task has more classes than the budget.

Regression test for a sampler bug: with C > n, `per = max(1, n // C) = 1`
produced one index per class in ascending class-id order and the trim to n
kept the LOWEST n ids -- identically for every seed. On CLIP-8 at n=32 that
silently restricted Cars to 32 of its 196 classes and SUN397 to 32 of 397,
and made the covered subset seed-invariant, so multi-draw error bars never
varied coverage.
"""

from types import SimpleNamespace

from apr.data import sample_replay_buffer


class Rows:
    """Dataset with `n_classes` classes, `per_class` examples each."""

    def __init__(self, n_classes=196, per_class=5):
        self.rows = [{"input_ids": [i], "labels": i % n_classes}
                     for i in range(n_classes * per_class)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if key == "labels":
            return [row["labels"] for row in self.rows]
        return self.rows[key]


def _classes(dataset, n, seed):
    buf = sample_replay_buffer(dataset, SimpleNamespace(is_regression=False),
                               n, seed, True)
    return sorted({row["labels"] for row in buf})


def test_class_subset_is_not_the_lowest_ids():
    covered = _classes(Rows(n_classes=196), 32, 0)
    assert len(covered) == 32
    assert covered != list(range(32)), "still taking the lowest 32 class ids"
    assert max(covered) >= 32, "class subset confined to the low end"


def test_class_subset_varies_with_seed():
    dataset = Rows(n_classes=196)
    subsets = {tuple(_classes(dataset, 32, seed)) for seed in (0, 1, 2)}
    assert len(subsets) == 3, "covered classes identical across buffer seeds"


def test_budget_is_exact_and_classes_distinct():
    for n in (8, 16, 32):
        buf = sample_replay_buffer(Rows(n_classes=196),
                                   SimpleNamespace(is_regression=False),
                                   n, 0, True)
        labels = [row["labels"] for row in buf]
        assert len(buf) == n
        assert len(set(labels)) == n, "a class was sampled twice"


def test_balanced_draw_unchanged_when_classes_fit():
    # C <= n keeps the original behaviour: every class represented, budget exact
    buf = sample_replay_buffer(Rows(n_classes=10, per_class=50),
                               SimpleNamespace(is_regression=False),
                               32, 0, True)
    labels = [row["labels"] for row in buf]
    assert len(buf) == 32
    assert set(labels) == set(range(10))
    assert min(labels.count(c) for c in range(10)) >= 3
