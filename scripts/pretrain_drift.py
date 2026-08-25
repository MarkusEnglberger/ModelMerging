"""Pretraining-capability drift probe.

Measures how much of the BASE MODEL'S OWN objective survives merging and
refinement, on text the experiments never touch (wikitext-2 validation):

  glue modality  : masked-LM loss of roberta-base's pretrained LM head over a
                   refined trunk (the head is untouched; only the trunk state
                   varies, so the delta is pure trunk drift).
  t5 modality    : prefix-LM loss of the full refined seq2seq model (flan-T5
                   descends from the LM-adapted T5, so continuation loss is
                   its pretraining-adjacent objective; the LM head is merged
                   with everything else on this track).

For each probed state it also records the parameter distance from theta_0, so
the (distance, capability-drift, task-score) triple can be reported together.

Probed states: theta_0, the config-lambda task-arithmetic merge, and the
split-selected from-pretrained winners of the three refinement arms,
re-derived deterministically from the same replay buffer (n_probe, probe_seed)
the grid runs used. States are re-derived rather than loaded because grid runs
do not persist weights.

Example (the Table-2 winners):
  python scripts/pretrain_drift.py --config configs/glue8.yaml \
      --n_probe 16 --probe_seed 0 \
      --arms apr:8:50 nogate:8:50 gd:0.001:50 \
      --out results/compare/pretrain_drift_glue8.json
"""

import argparse
import dataclasses
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from apr.config import ExperimentConfig
from apr.pipeline import MergeContext, _log
from apr.models import (load_encoder_state, pd_sub, pd_global_norm)

ARM_KW = {
    "apr": {},
    "nogate": {"gate_mode": "none"},
    "gd": {"gate_mode": "none", "update_mode": "grad", "clip_mode": "none"},
}


def wikitext_blocks(tokenizer, block_len, n_blocks, cache_dir):
    """Fixed token blocks from wikitext-2 validation (deterministic).

    cache_dir is the config's (usually None), so the dataset resolves through
    HF_HOME exactly as slurm/common.sh sets it -- the compute nodes run with
    HF_HUB_OFFLINE=1, so an explicit path here would miss the prefetched copy.
    """
    import datasets
    ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1",
                               split="validation", cache_dir=cache_dir)
    text = "\n\n".join(r["text"] for r in ds if r["text"].strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n = min(n_blocks, len(ids) // block_len)
    return ids[: n * block_len].view(n, block_len)


@torch.no_grad()
def mlm_loss(model, blocks, mask_id, vocab, device, batch=32, seed=0):
    """Masked-LM loss with the standard 15% / 80-10-10 corruption, PREBUILT
    once with a fixed seed so every probed state sees identical batches."""
    g = torch.Generator().manual_seed(seed)
    prob = torch.full(blocks.shape, 0.15)
    masked = torch.bernoulli(prob, generator=g).bool()
    labels = blocks.clone()
    labels[~masked] = -100
    inputs = blocks.clone()
    replace = torch.bernoulli(torch.full(blocks.shape, 0.8), generator=g).bool() & masked
    random = (torch.bernoulli(torch.full(blocks.shape, 0.5), generator=g).bool()
              & masked & ~replace)
    inputs[replace] = mask_id
    inputs[random] = torch.randint(0, vocab, (int(random.sum()),), generator=g)
    model.to(device).eval()
    tot, ntok = 0.0, 0
    for i in range(0, len(blocks), batch):
        out = model(input_ids=inputs[i:i+batch].to(device),
                    labels=labels[i:i+batch].to(device))
        k = int((labels[i:i+batch] != -100).sum())
        tot += float(out.loss) * k
        ntok += k
    model.to("cpu")
    return tot / ntok


@torch.no_grad()
def prefix_lm_loss(t5, blocks, device, batch=16):
    """Prefix-LM continuation loss: first half of each block conditions the
    second half. Deterministic; identical batches for every probed state."""
    half = blocks.shape[1] // 2
    src, tgt = blocks[:, :half], blocks[:, half:]
    t5.to(device).eval()
    tot, ntok = 0.0, 0
    for i in range(0, len(blocks), batch):
        out = t5(input_ids=src[i:i+batch].to(device),
                 labels=tgt[i:i+batch].to(device))
        k = tgt[i:i+batch].numel()
        tot += float(out.loss) * k
        ntok += k
    t5.to("cpu")
    return tot / ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_probe", type=int, required=True)
    ap.add_argument("--probe_seed", type=int, default=0)
    ap.add_argument("--arms", nargs="*", default=[],
                    help="arm:lr:steps, e.g. apr:8:50 -- re-derives the cell "
                         "from the replay buffer (legacy path; prefer "
                         "--state_files for protocol-v2 runs)")
    ap.add_argument("--state_files", nargs="*", default=[],
                    help="label=path.pt pairs: probe saved encoder states "
                         "directly (cv_protocol.py --save_winners output), so "
                         "the probed model IS the reported one and nothing is "
                         "re-derived or re-trained")
    ap.add_argument("--merges", nargs="*", default=[],
                    help="split-selected merge baselines to probe alongside "
                         "the config-lambda TA merge: ties:DENSITY:LAM | "
                         "dareties:DROP:TRIM:LAM | bc:DENSITY:OUTLIER:LAM | "
                         "ada:VARIANT (matched budget, reuses the merge cache)")
    ap.add_argument("--n_blocks", type=int, default=256)
    ap.add_argument("--block_len", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.data.n_probe = args.n_probe
    cfg.data.probe_seed = args.probe_seed
    ctx = MergeContext.build(cfg)
    device = ctx.device

    # ---- the probe model + loss function for this modality ----
    cache = cfg.data.cache_dir
    if cfg.modality == "glue":
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.base_model, cache_dir=cache)
        probe_model = AutoModelForMaskedLM.from_pretrained(cfg.base_model,
                                                           cache_dir=cache)
        blocks = wikitext_blocks(tok, args.block_len, args.n_blocks, cache)

        def probe(state):
            load_encoder_state(probe_model, state)
            return mlm_loss(probe_model, blocks, tok.mask_token_id,
                            probe_model.config.vocab_size, device)
    elif cfg.modality == "t5":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.base_model, cache_dir=cache)
        holder = ctx.per_task[ctx.task_names[0]]["model"]
        blocks = wikitext_blocks(tok, args.block_len, args.n_blocks, cache)

        def probe(state):
            load_encoder_state(holder, state)
            return prefix_lm_loss(holder.t5, blocks, device)
    else:
        raise SystemExit(f"no pretraining probe for modality {cfg.modality}")

    theta0 = ctx.base_encoder
    results = {"config": args.config, "n_probe": args.n_probe,
               "probe_seed": args.probe_seed, "wikitext_blocks": args.n_blocks,
               "block_len": args.block_len, "states": {}}

    def record(name, state, extra=None):
        d = pd_global_norm(pd_sub(state, theta0))
        loss = probe(state)
        results["states"][name] = {"dist_theta0": d, "drift_loss": loss,
                                   **(extra or {})}
        _log(f"[drift] {name:24s} dist={d:8.3f}  loss={loss:.4f}")

    record("theta0", theta0)
    record(f"merge:TA@l{cfg.experts[0].lam:g}", ctx.merged0)

    if args.merges:
        from apr.merge_methods import (ties_merge, dare_ties_merge,
                                       breadcrumbs_merge)
        names = ctx.task_names
        for spec in args.merges:
            fam, *ps = spec.split(":")
            if fam == "ties":
                d, lam = float(ps[0]), float(ps[1])
                state = ties_merge(ctx.base_encoder, ctx.task_vectors,
                                   lam=lam, density=d)
                nm = f"merge:TIES@d{d:g},l{lam:g}"
            elif fam == "dareties":
                dd, tr, lam = (float(x) for x in ps)
                state = dare_ties_merge(ctx.base_encoder, ctx.task_vectors,
                                        lam=lam, drop_density=dd,
                                        trim_density=tr, seed=0)
                nm = f"merge:DARETIES@dd{dd:g},t{tr:g},l{lam:g}"
            elif fam == "bc":
                d, o, lam = (float(x) for x in ps)
                state = breadcrumbs_merge(ctx.base_encoder, ctx.task_vectors,
                                          {n: lam for n in names},
                                          density=d, outlier_frac=o)
                nm = f"merge:BC@d{d:g},o{o:g},l{lam:g}"
            elif fam == "ada":
                variant = ps[0] if ps else "task"
                # same params the grid runs used, so the cache key matches
                # (grid_nn.sbatch: --ada_data probe_buffer, defaults elsewhere)
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                import merge_baselines as mb
                params = {"variant": variant, "steps": 300, "lr": 1e-3,
                          "bs": 16, "init_lam": 0.3, "seed": cfg.seed}
                extra = {"data_key": "probe_buffer"}
                state = mb.merge_cache_load(cfg, "ada", params, extra)
                if state is None:
                    from apr.adamerging import adamerging
                    state, _ = adamerging(
                        ctx.base_encoder, ctx.task_vectors, ctx.per_task,
                        names, ctx.device, layerwise=(variant == "layer"),
                        steps=300, lr=1e-3, batch_size=16, init_lam=0.3,
                        seed=cfg.seed, data_key="probe_buffer",
                        tv_on_gpu=(cfg.modality != "causal_lm"), logger=_log)
                    mb.merge_cache_save(cfg, "ada", params, state, extra)
                nm = f"merge:ADA-{variant}-matched"
            else:
                raise SystemExit(f"unknown merge family in --merges: {spec}")
            record(nm, state)
            del state

    for spec in args.state_files:
        label, path = spec.split("=", 1)
        st = torch.load(path, map_location="cpu")
        record(label, st, {"state_file": path})
        del st

    for spec in args.arms:
        arm, lr, steps = spec.split(":")
        rc = dataclasses.replace(cfg.refine, steps=int(steps), lr=float(lr),
                                 **ARM_KW[arm])
        _log(f"[refine] {arm} from theta0 @ lr{lr} S{steps}")
        st, _ = ctx.run_refine_from(theta0, rc, seed=cfg.seed)
        record(f"{arm}:from=theta0@lr{lr},S{steps}", st,
               {"arm": arm, "lr": float(lr), "steps": int(steps)})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    _log(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
