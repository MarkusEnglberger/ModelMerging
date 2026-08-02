import torch
from transformers import T5Config, T5ForConditionalGeneration

from apr.mats_t5 import (_squad_scores, get_mats_task, inject_mats_ia3,
                         make_mats_collator, mats_checkpoint)


class FakeTokenizer:
    def pad(self, features, return_tensors=None):
        width = max(len(f["input_ids"]) for f in features)
        ids = torch.zeros((len(features), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, feature in enumerate(features):
            n = len(feature["input_ids"])
            ids[i, :n] = torch.tensor(feature["input_ids"])
            mask[i, :n] = torch.tensor(feature["attention_mask"])
        return {"input_ids": ids, "attention_mask": mask}


def test_registry_matches_mats_eight_task_mixture():
    names = {"cosmos_qa", "social_iqa", "paws", "quail", "wiki_qa",
             "quartz", "qasc", "ropes"}
    assert {get_mats_task(n).name for n in names} == names
    assert mats_checkpoint("paws").endswith("checkpoint_1299.pt")
    assert get_mats_task("social-iqa").label({"label": "3"}) == 2


def test_collator_pads_variable_answer_choices():
    features = [
        {"input_ids": [1, 2], "attention_mask": [1, 1], "target_ids": [4],
         "labels": 1, "choice_ids": [[5], [6, 7]]},
        {"input_ids": [3], "attention_mask": [1], "target_ids": [8, 9],
         "labels": 0, "choice_ids": [[10], [11], [12]]},
    ]
    batch = make_mats_collator(FakeTokenizer())(features)
    assert batch["choice_ids"].shape == (2, 3, 2)
    assert batch["choice_mask"].tolist() == [[True, True, False],
                                             [True, True, True]]
    assert batch["target_ids"].tolist() == [[4, -100], [8, 9]]


def test_squad_normalization_and_best_reference():
    exact, f1 = _squad_scores("The Eiffel Tower!", ["Eiffel Tower", "Paris"])
    assert exact == 1.0
    assert f1 == 1.0


def test_ia3_injection_uses_only_expected_merge_parameters():
    config = T5Config(
        d_model=16, d_kv=4, d_ff=32, num_layers=2,
        num_decoder_layers=3, num_heads=4, vocab_size=32,
        feed_forward_proj="gated-gelu",
    )
    t5 = inject_mats_ia3(T5ForConditionalGeneration(config))
    names = [name for name, _ in t5.named_parameters()
             if name.endswith(".ia3_scale")]
    assert len(names) == 3 * 2 + 5 * 3
    assert all(dict(t5.named_parameters())[name].requires_grad for name in names)
    assert all(not parameter.requires_grad for name, parameter in t5.named_parameters()
               if name not in names)
