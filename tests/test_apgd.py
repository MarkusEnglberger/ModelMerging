from collections import OrderedDict

import torch

from apr.apgd import (
    _remove_shared_component_,
    _released_topk_threshold,
    _taskvector_loss,
    apgd_linear_keys,
    apgd_merge,
    prepare_apgd,
)


def _toy_vectors():
    return {
        "a": OrderedDict([
            ("model.encoder.layer.0.query.weight",
             torch.tensor([[1.0, 0.0], [0.0, 0.5]])),
            ("model.encoder.layer.0.query.bias", torch.tensor([0.2, -0.1])),
            ("model.embeddings.word_embeddings.weight", torch.ones(3, 2)),
        ]),
        "b": OrderedDict([
            ("model.encoder.layer.0.query.weight",
             torch.tensor([[0.0, 0.5], [-1.0, 0.0]])),
            ("model.encoder.layer.0.query.bias", torch.tensor([-0.2, 0.1])),
            ("model.embeddings.word_embeddings.weight", -torch.ones(3, 2)),
        ]),
    }


def test_apgd_targets_only_encoder_linear_weights():
    assert apgd_linear_keys(_toy_vectors()) == (
        "model.encoder.layer.0.query.weight",
    )


def test_projected_gradient_is_orthogonal_to_shared_basis():
    basis = torch.tensor([[1.0], [0.0]])
    gradient = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    _remove_shared_component_(gradient, basis)
    torch.testing.assert_close(basis.T @ gradient, torch.zeros(1, 2))
    torch.testing.assert_close(gradient, torch.tensor([[0.0, 0.0], [4.0, 5.0]]))


def test_taskvector_loss_matches_released_rowwise_expression():
    vectors = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[-1.0, 1.0], [2.0, -2.0]],
    ])
    delta = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
    lambdas = torch.tensor([0.2, 0.5])
    got = _taskvector_loss(vectors, delta, lambdas)

    merged = sum(lambdas[i] * (vectors[i] + delta) for i in range(2))
    expected = sum(((vectors[j] * (vectors[j] - merged)).sum(dim=1) ** 2).sum()
                   for j in range(2))
    torch.testing.assert_close(got, expected)


def test_topk_threshold_matches_released_inclusive_boundary():
    values = OrderedDict([("weight", torch.arange(10.0))])
    threshold = _released_topk_threshold(values, 0.3)
    assert threshold == 6.0
    # The released kth-value convention retains 4/10 entries here (6--9),
    # including the magnitude exactly at the 70th-percentile boundary.
    assert int((values["weight"].abs() >= threshold).sum()) == 4


def test_apgd_merge_is_deterministic_and_leaves_non_linear_weights_at_base():
    task_vectors = _toy_vectors()
    base = OrderedDict((key, torch.zeros_like(value))
                       for key, value in task_vectors["a"].items())
    prep = prepare_apgd(task_vectors, "cpu", subspace_divisor=2)

    first, info = apgd_merge(
        base, task_vectors, eta=0.1, device="cpu", preparation=prep,
        iterations=3, lr=1e-3, keep_density=1.0)
    second, _ = apgd_merge(
        base, task_vectors, eta=0.1, device="cpu", preparation=prep,
        iterations=3, lr=1e-3, keep_density=1.0)

    for key in base:
        torch.testing.assert_close(first[key], second[key])
    torch.testing.assert_close(first["model.encoder.layer.0.query.bias"],
                               base["model.encoder.layer.0.query.bias"])
    torch.testing.assert_close(first["model.embeddings.word_embeddings.weight"],
                               base["model.embeddings.word_embeddings.weight"])
    assert info["n_linear_keys"] == 1


def test_apgd_mask_is_selected_from_original_task_vector(monkeypatch):
    task_vectors = _toy_vectors()
    base = OrderedDict((key, torch.zeros_like(value))
                       for key, value in task_vectors["a"].items())
    prep = prepare_apgd(task_vectors, "cpu", subspace_divisor=2)
    seen = []

    def record_threshold(values, density):
        seen.append(OrderedDict((key, value.clone()) for key, value in values.items()))
        return 0.0

    monkeypatch.setattr("apr.apgd._released_topk_threshold", record_threshold)
    apgd_merge(
        base, task_vectors, eta=0.1, device="cpu", preparation=prep,
        iterations=3, lr=1e-3, keep_density=0.5)

    assert len(seen) == len(task_vectors)
    for recorded, name in zip(seen, task_vectors):
        torch.testing.assert_close(
            recorded["model.encoder.layer.0.query.weight"],
            task_vectors[name]["model.encoder.layer.0.query.weight"])
