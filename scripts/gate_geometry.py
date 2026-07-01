#!/usr/bin/env python
"""Why is the APR gate density ~0.36 with tiny variance? Geometry control.

At the merge, for each task compute g = grad L_i and v = theta_i - merge, then the
fraction of coords with score<0 for several "directions":
  true v       : the real merge->expert direction      (the gate)   -> expect ~0.36
  -v           : away from the expert                                -> expect ~1-0.36
  sign-shuffled: v's magnitudes, signs randomly permuted             -> expect ~0.50
  random sign  : +/-|v| with random signs                            -> expect ~0.50
Also the magnitude-weighted negative fraction of g*v (does the kept minority carry
the loss-decrease mass?).
"""

import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from apr.config import ExperimentConfig
from apr.pipeline import MergeContext, _log
from apr.models import load_encoder_state, pd_to


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/compare/gate_geometry.json")
    args = ap.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    ctx = MergeContext.build(cfg)
    dev = ctx.device
    gen = torch.Generator(device=dev); gen.manual_seed(0)

    report = {"config": cfg.to_dict(), "tasks": {}}
    print(f"\n{'task':<6} {'true_v':>8} {'-v':>8} {'signshuf':>9} {'randsign':>9} "
          f"{'magwt_neg':>10}")
    for h in ctx.handles:
        h.model.to(dev); load_encoder_state(h.model, ctx.merged0)
        g = h.grad_fn()
        expert = pd_to(h.expert_encoder, dev)
        nt = {k: 0 for k in ["true", "negv", "shuf", "rand"]}
        tot = 0; magneg = 0.0; magtot = 0.0
        for n in g:
            gi = g[n]
            vi = expert[n] - ctx.merged0[n].to(dev)
            prod = gi * vi
            nt["true"] += int((prod < 0).sum())
            nt["negv"] += int((gi * (-vi) < 0).sum())
            # sign-shuffled: keep |v|, permute signs across this tensor's coords
            flat = vi.flatten()
            perm = torch.randperm(flat.numel(), generator=gen, device=dev)
            shuf = (flat.abs() * torch.sign(flat)[perm]).reshape(vi.shape)
            nt["shuf"] += int((gi * shuf < 0).sum())
            # random sign of matched magnitude
            rsign = torch.where(torch.rand(vi.shape, generator=gen, device=dev) < 0.5,
                                -1.0, 1.0)
            nt["rand"] += int((gi * (rsign * vi.abs()) < 0).sum())
            tot += vi.numel()
            magneg += float(prod[prod < 0].abs().sum()); magtot += float(prod.abs().sum())
        h.model.to("cpu")
        if dev.startswith("cuda"): torch.cuda.empty_cache()
        d = {k: nt[k] / tot for k in ["true", "negv", "shuf", "rand"]}
        d["magwt_neg"] = magneg / magtot
        report["tasks"][h.name] = d
        print(f"{h.name:<6} {d['true']:>8.4f} {d['negv']:>8.4f} {d['shuf']:>9.4f} "
              f"{d['rand']:>9.4f} {d['magwt_neg']:>10.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
