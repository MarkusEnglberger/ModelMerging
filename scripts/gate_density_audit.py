"""Decompose APR's attribution-gate density at the pretrained model.

The refinement logs a single number per (sweep, task) -- the fraction of
mergeable coordinates with g*v < 0. That number is stable to three decimals
across sweeps, learning rates and tasks on GLUE, which invites the question of
what it is actually measuring. This script splits it into the two factors that
can produce a stable density:

  density = P(g != 0) * P(g*v < 0 | g != 0)

A gate that carried directional signal would show the conditional term far from
1/2. A gate whose density is set by *gradient sparsity* (encoder coordinates
that no replay example touches -- word-embedding rows for absent tokens) shows
the conditional term at 1/2 and the whole structure in P(g != 0).

Reports both terms overall, per task, and split by parameter group (embeddings
vs the rest), at theta_0 with the same budget/seed the protocol run used.
No evaluation split is read.

Usage:
  python scripts/gate_density_audit.py --config configs/glue8.yaml \
      --budget 32 --seed 100 --out results/compare/gate_density_glue8.json
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig                       # noqa: E402
from apr.pipeline import _make_backend, _log                  # noqa: E402
from apr.gradients import make_grad_fn                        # noqa: E402
from apr.data import sample_replay_buffer                     # noqa: E402
from apr.models import get_encoder_state, load_encoder_state  # noqa: E402


def group_of(name: str) -> str:
    """Coarse parameter group. Word/position embeddings are the tensors whose
    rows a finite replay buffer cannot all touch."""
    n = name.lower()
    if "word_embeddings" in n:
        return "word_embeddings"
    if "position_embeddings" in n or "token_type" in n:
        return "other_embeddings"
    if n.endswith(".bias") or "layernorm" in n:
        return "bias_layernorm"
    return "weights"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--seed", type=int, default=100,
                    help="buffer seed; 100+d matches protocol draw d")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    _log(f"[load] {cfg.base_model} (modality={cfg.modality}) on {device}")
    base_encoder, load_expert = _make_backend(cfg, device)

    report = {"config": args.config, "budget": args.budget, "seed": args.seed,
              "device": device, "tasks": {}, "groups": {}, "overall": {}}
    tot = dict(n=0, nz=0, gate=0)
    grp_tot = {}

    for e in cfg.experts:
        spec, model, train_ds, eval_ds, collator = load_expert(e)
        expert_encoder = get_encoder_state(model)
        buffer, _ = sample_replay_buffer(train_ds, spec, args.budget, args.seed,
                                         cfg.data.class_balanced,
                                         return_indices=True)
        grad_fn = make_grad_fn(model, buffer, collator,
                               cfg.data.grad_batch_size or cfg.data.eval_batch_size,
                               device)
        # gradient at theta_0: the encoder must HOLD the pretrained state
        model.to(device)
        load_encoder_state(model, base_encoder)
        g = grad_fn()
        model.to("cpu")

        t_n = t_nz = t_gate = 0
        per_group = {}
        for name in base_encoder:
            if name not in g:
                continue
            gi = g[name].detach().float().cpu()
            vi = (expert_encoder[name].float().cpu()
                  - base_encoder[name].float().cpu())
            nz = (gi != 0)
            gate = (gi * vi < 0)
            k = group_of(name)
            d = per_group.setdefault(k, dict(n=0, nz=0, gate=0, gate_nz=0))
            d["n"] += gi.numel()
            d["nz"] += int(nz.sum())
            d["gate"] += int(gate.sum())
            d["gate_nz"] += int((gate & nz).sum())
            t_n += gi.numel(); t_nz += int(nz.sum()); t_gate += int(gate.sum())
        del g

        rec = {"n_coords": t_n,
               "p_grad_nonzero": t_nz / t_n,
               "gate_density": t_gate / t_n,
               "p_gate_given_nonzero": t_gate / max(t_nz, 1),
               "by_group": {k: {"n": d["n"], "p_grad_nonzero": d["nz"] / d["n"],
                                "gate_density": d["gate"] / d["n"],
                                "p_gate_given_nonzero": d["gate_nz"] / max(d["nz"], 1)}
                            for k, d in per_group.items()}}
        report["tasks"][e.name] = rec
        tot["n"] += t_n; tot["nz"] += t_nz; tot["gate"] += t_gate
        for k, d in per_group.items():
            a = grp_tot.setdefault(k, dict(n=0, nz=0, gate=0, gate_nz=0))
            for f in ("n", "nz", "gate", "gate_nz"):
                a[f] += d[f]
        _log(f"[{e.name:6s}] density={rec['gate_density']:.4f}  "
             f"P(g!=0)={rec['p_grad_nonzero']:.4f}  "
             f"P(gate|g!=0)={rec['p_gate_given_nonzero']:.4f}")
        model.to("cpu")
        del model

    report["overall"] = {"p_grad_nonzero": tot["nz"] / tot["n"],
                         "gate_density": tot["gate"] / tot["n"],
                         "p_gate_given_nonzero": tot["gate"] / max(tot["nz"], 1)}
    report["groups"] = {k: {"n": d["n"], "p_grad_nonzero": d["nz"] / d["n"],
                            "gate_density": d["gate"] / d["n"],
                            "p_gate_given_nonzero": d["gate_nz"] / max(d["nz"], 1)}
                        for k, d in grp_tot.items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    o = report["overall"]
    _log(f"\n[overall] density={o['gate_density']:.4f} = "
         f"P(g!=0)={o['p_grad_nonzero']:.4f} x "
         f"P(gate|g!=0)={o['p_gate_given_nonzero']:.4f}")
    for k, d in sorted(report["groups"].items(), key=lambda kv: -kv[1]["n"]):
        _log(f"  {k:18s} n={d['n']:>11,d}  P(g!=0)={d['p_grad_nonzero']:.4f}  "
             f"density={d['gate_density']:.4f}  "
             f"P(gate|g!=0)={d['p_gate_given_nonzero']:.4f}")
    _log(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
