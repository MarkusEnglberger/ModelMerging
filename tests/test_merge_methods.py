from collections import OrderedDict

import pytest
import torch

from apr.merge_methods import (
    _della_keep_probabilities,
    _della_sampled,
    della_merge,
)


def test_della_probabilities_follow_rowwise_magnitude_ranks():
    values = torch.tensor([[1.0, 4.0, 2.0, 3.0],
                           [40.0, 10.0, 30.0, 20.0]])

    probabilities = _della_keep_probabilities(
        values, density=0.5, window_size=0.2)

    expected = torch.tensor([[0.4, 0.6, 0.46666667, 0.53333333],
                             [0.6, 0.4, 0.53333333, 0.46666667]])
    torch.testing.assert_close(probabilities, expected)


def test_della_sampling_is_deterministic_and_rescales_each_survivor():
    values = torch.arange(1.0, 101.0).reshape(10, 10)
    probabilities = _della_keep_probabilities(
        values, density=0.6, window_size=0.2)

    first = _della_sampled(values, 0.6, 0.2, 7, "task", "weight")
    second = _della_sampled(values, 0.6, 0.2, 7, "task", "weight")
    other_seed = _della_sampled(values, 0.6, 0.2, 8, "task", "weight")

    torch.testing.assert_close(first, second)
    assert not torch.equal(first, other_seed)
    survivor_values = values / probabilities
    assert torch.all((first == 0) | torch.isclose(first, survivor_values))


def test_della_merge_elects_sign_and_averages_agreeing_deltas():
    base = OrderedDict(weight=torch.tensor([10.0, 20.0]))
    task_vectors = {
        "a": OrderedDict(weight=torch.tensor([1.0, -2.0])),
        "b": OrderedDict(weight=torch.tensor([-3.0, -4.0])),
        "c": OrderedDict(weight=torch.tensor([5.0, 3.0])),
    }

    merged = della_merge(base, task_vectors, lam=2.0, density=1.0,
                         window_size=0.14, seed=42)

    # coordinate 0 elects positive -> mean(1, 5) = 3; coordinate 1 elects
    # negative -> mean(-2, -4) = -3.
    torch.testing.assert_close(merged["weight"], torch.tensor([16.0, 14.0]))


def test_della_extends_rowwise_ranking_to_higher_rank_tensors():
    values = torch.tensor([[[1.0, 3.0], [4.0, 2.0]]])
    probabilities = _della_keep_probabilities(
        values, density=0.5, window_size=0.2)

    expected = torch.tensor([[[0.4, 0.53333333], [0.6, 0.46666667]]])
    torch.testing.assert_close(probabilities, expected)


@pytest.mark.parametrize(
    ("density", "window_size"),
    [(0.0, 0.0), (1.1, 0.0), (0.5, -0.1), (0.1, 0.3), (0.9, 0.3)],
)
def test_della_rejects_invalid_probability_ranges(density, window_size):
    with pytest.raises(ValueError):
        _della_keep_probabilities(torch.ones(2, 2), density, window_size)
