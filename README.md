# Attribution-Patching-Guided Replay Refinement for Task-Vector Merging

Implementation of the proposal of the same name. After a standard task-arithmetic
merge of fine-tuned experts, the merged encoder is refined with a few replay-buffer
gradient steps, where each task's update is **gated by a parameter-space attribution
score** (the sign of `g_{i,r} · v_{i,r}`) and applied **sequentially** (one task at a
time, immediately).

## Method (what the code implements)

Given a shared pretrained encoder `θ₀` and experts `θ_i` (task vectors `τ_i = θ_i − θ₀`):

1. **Task-arithmetic merge** `θ⁽⁰⁾ = θ₀ + Σ λ_i τ_i`  (Eq. 4).
2. **Attribution gate** for task `i` at the current state `θ`, with
   `g_i = ∇_θ L_i(θ; D_probe_i)` and `v_i = θ_i − θ`:
   - `m_{i,r} = 1[ g_{i,r} v_{i,r} < −ε_gate ]`  (move toward the expert only where it
     is locally loss-decreasing).
   - `ũ_{i,r} = −g_{i,r} |v_{i,r}| m_{i,r}`  (distance-scaled negative gradient, Eq. 12).
   - `u_{i,r} = clip(ũ_{i,r}, ±γ|v_{i,r}|)`  (per-coordinate trust region, Eq. 18).
3. **Sequential refinement** (Algorithm 1): within each sweep, apply each task's
   clipped update to the model *immediately* before computing the next task's gradient.

The aggregated-U variant (`refine.aggregated=true`) is the key ablation: compute all
`u_i` at the start-of-sweep state and apply their sum once.

## Layout

```
src/apr/
  tasks.py        GLUE-8 task registry (keys, num_labels, metric, clusters)
  config.py       YAML-backed experiment config (RefineConfig has all ablation knobs)
  data.py         GLUE loading, tokenization, replay-buffer sampling
  models.py       encoder/head split + param-dict vector algebra
  taskvec.py      task vectors + task-arithmetic merge (Eq. 1/4)
  gradients.py    replay-buffer task-gradient g_i (mean-loss, eval mode)
  refine.py       Algorithm 1: gate, clip, sequential/aggregated update
  metrics.py      GLUE scoring + normalized retention (Eq. 21)
  eval.py         eval-split scoring of an (encoder, head) pair
  interference.py cross-task interference matrix C_{j<-i} (Eq. 22)
  pipeline.py     end-to-end merge + refine + evaluate
scripts/
  run_merge.py    main entry: run a config, write results/<tag>.json
  train_expert.py fine-tune a single-task expert from a shared base
configs/          smoke.yaml (DistilRoBERTa MRPC+STS-B), poc3.yaml (RoBERTa x3)
slurm/            sbatch scripts for the GPU partition (mcs.gpu.q)
tests/            synthetic unit tests for the gate + Algorithm 1
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.2 datasets==2.21.0 evaluate==0.4.3 \
            scikit-learn scipy pyyaml accelerate==0.34.2 pandas
```

Assets (checkpoints + GLUE) are cached under `.hf_cache/` (`HF_HOME`). Pre-fetch on a
networked node; the SLURM scripts run offline (`HF_HUB_OFFLINE=1`).

## Running

```bash
# unit tests (CPU, fast)
python tests/test_refine.py

# local run
python scripts/run_merge.py --config configs/poc3.yaml

# on the cluster
sbatch slurm/poc3.sbatch
sbatch slurm/smoke.sbatch          # trains DistilRoBERTa experts first

# overrides (ablations):
python scripts/run_merge.py --config configs/poc3.yaml \
  --override refine.aggregated=true tag=poc3_aggU
python scripts/run_merge.py --config configs/poc3.yaml \
  --override refine.gate_mode=none tag=poc3_nogate
```

Results are written to `results/<tag>.json` with per-task `base/expert/merge/refined`
scores, normalized retention before/after refinement, and the per-sweep gate/update
history.

## CLIP/ViT vision track

The vision track (proposal H5) reuses the whole core (task vectors, Algorithm 1,
gradients, metrics, interference) via a modality switch (`modality: clip`). The
mergeable encoder is the CLIP **vision tower**; each task keeps a *frozen*
zero-shot head (base `visual_projection` + CLIP-text class embeddings +
`logit_scale`), exactly analogous to the fixed GLUE heads. Experts are the
FusionBench/`tanganke` fine-tuned `CLIPVisionModel` checkpoints (resolved from the
task name); datasets are the standard suite (EuroSAT, GTSRB, MNIST, SVHN, DTD,
RESISC45, SUN397, Cars). Vision-specific code lives in `src/apr/vision.py`.

```bash
# pre-fetch assets on a networked node, then run offline
python scripts/prefetch_clip.py --tasks eurosat gtsrb mnist   # or --all
sbatch slurm/clip_poc.sbatch
```

## Status

Implemented: core framework + Algorithm 1, GLUE multi-head merging, the DistilRoBERTa
smoke test and the 3-task RoBERTa POC. **CLIP/ViT vision track** with a validated
3-task POC (EuroSAT/GTSRB/MNIST): at the corrected main setting (S=5, gamma=1,
clip-after-lr, n=64) a fair sweep gives APR@lr32 mean NormRet 0.940 / worst 0.915,
beating ungated anchoring (0.917/0.876), ordinary replay GD (0.874/0.792) and the
unrefined merge (0.811/0.721); inverted-gate control -0.579. On the **full 8-task
suite** (SUN397/Cars/RESISC45/EuroSAT/SVHN/GTSRB/MNIST/DTD, `configs/clip8.yaml`) APR
again leads: APR@lr8 mean 0.598 / worst 0.159 vs nogate 0.536/-0.086, ordinary GD
0.458/-0.203, merge 0.270/-0.527 -- and APR is the only method holding worst-task
retention positive. Notably the AP gate helps on CLIP at both 3 and 8 tasks (it
rescues the fragile Cars/SUN397 tasks), unlike GLUE-8 where ungated anchoring tied it. The
ablation/baseline knobs (gate mode, update mode, aggregated-U, task order,
clipping/normalization, expansion point) are wired in `RefineConfig` and apply to both
modalities. Deferred: full GLUE-8 sweep, full 8-task CLIP suite (+ vision hyperparameter
sweep and baselines), the T5 shared-head track, and the full baseline suite.
