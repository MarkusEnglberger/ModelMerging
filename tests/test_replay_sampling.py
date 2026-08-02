from types import SimpleNamespace

from apr.data import sample_replay_buffer


class Rows:
    def __init__(self, n=100):
        self.rows = [{"input_ids": [i], "labels": i % 2} for i in range(n)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if key == "labels":
            return [row["labels"] for row in self.rows]
        return self.rows[key]


def test_selection_sample_is_exactly_disjoint_from_probe_sample():
    dataset = Rows()
    spec = SimpleNamespace(is_regression=False)
    train, train_indices = sample_replay_buffer(
        dataset, spec, 32, 0, True, return_indices=True)
    selection, selection_indices = sample_replay_buffer(
        dataset, spec, 32, 1, True, exclude_indices=train_indices,
        return_indices=True)

    assert len(train) == len(selection) == 32
    assert set(train_indices).isdisjoint(selection_indices)
