# SURF Small Compute application (NWO) — Snellius GPU

Form: servicedesk.surf.nl → "Small Compute applications (NWO)". New project (the current
work runs on the remaining budget of another project, EINF-16966, which is nearly
exhausted). Items in [brackets] need your input.

**Requested:** 500,000 SBU on Snellius GPU nodes (`gpu_h100`, `gpu_a100`), 1 year,
2 TB project space, default support.

---

## Applicant

Markus Englberger, Eindhoven University of Technology, [department/group].
[Temporary appointment → digital signature by supervisor with permanent appointment:
name, e-mail.] Project title: *Expert-anchored gradient descent for model merging with
small labeled replay buffers*.

## Scientific purpose

Model merging combines several fine-tuned experts of one pretrained model into a single
model by weight-space arithmetic, but merges degrade under task interference. We study
the setting where a small labeled replay buffer (8–16 examples per task) is available and
propose expert-anchored gradient descent: a replay refinement whose per-task updates are
gated by a first-order loss attribution, scaled by the distance to the expert and clipped
to a trust region (provably non-expansive toward the experts' box hull). The method is
evaluated on GLUE-8 (RoBERTa), two CLIP/ViT suites and a decoder-only LLM track following
MergeBench (Llama-3.2-3B with published math, coding, instruction-following and
multilingual experts).

The compute is needed for the LLM track, by far the most expensive part: evaluation is
generation-based (GSM8K, MBPP, IFEval, multilingual MC), so every hyperparameter cell
must be scored by decoding, and all hyperparameters are selected on a disjoint buffer.
Concretely: (i) complete the MergeBench grid (ordinary-GD and ungated-ablation
composition arms, DOGE/APGD and RegMean baselines — currently ~16 missing cells);
(ii) replicate it over five replay-buffer seeds as done on the other benchmarks;
(iii) score the selected cells on the full (non-subsampled) evaluation sets; (iv) add the
held-out-task retention and pretraining-drift probes to the LLM track; (v) verify that
the conclusions hold at the next model scale (MergeBench Llama-3.1-8B experts); and
(vi) revision/rebuttal experiments.

## SBU justification

All jobs are single-GPU (¼ node). H100 = 192 SBU/GPU-h, A100 = 128 SBU/GPU-h. Per-run
costs are measured on our current Snellius project: one 3B MergeBench grid run
(learning-rate × sweep grid for one initialization, selection, evaluation of the
selected cell) ≈ 5 H100-h; full-eval scoring of one 3B model ≈ 2 H100-h; an 8B grid run
≈ 3× the 3B cost.

| Work package | Runs × GPU-h × SBU/h | SBU |
|---|---|---|
| Missing 3B MergeBench cells (two budgets) | 32 × 5 × 192 | 31,000 |
| 5-seed replication, 3B (6 inits × 3 methods × 4 seeds) | 72 × 2 × 192 | 28,000 |
| Full-evaluation-set scoring of selected 3B cells | 40 × 2 × 192 | 15,000 |
| Held-out retention + drift probes, 3B | 40 × 1.5 × 192 | 12,000 |
| 8B-scale check: pretrained + 5 merges × 3 methods, 1 budget, + 5-seed replication of the headline cells | 18 × 15 × 192 + 20 × 6 × 192 | 75,000 |
| Re-bracketing of boundary selections (≈25 % of grids) and failed/timed-out jobs (≈15 %) | — | 40,000 |
| Encoder-track revision experiments (extra seeds, extended coefficient grids), A100 | 150 × 3 × 128 | 58,000 |
| Reserve for reviewer-requested experiments (≈ 100 grid runs) | 100 × 6 × 192 | 115,000 |
| **Total** | | **≈ 375,000** |

We request **500,000 SBU**: the itemized plan plus a ~30 % margin, because the
generation-based evaluation cost varies with the selected sweep horizon and decoding
length, and because only one small application per year is permitted.

## Memory

Standard GPU nodes; ¼ H100 node (16 cores, 180 GB) per job. The 8B track keeps base,
experts, task vectors and merge in bf16 in host memory (≈ 160 GB), still within a ¼–½
node. No fat nodes.

## Storage

Inputs: public checkpoints (3B + 8B MergeBench experts ≈ 120 GB, encoder checkpoints
≈ 30 GB) and datasets (< 20 GB); outputs are small JSON score files. Merged models are
not stored. Request **2 TB project space** (no backup needed) for the checkpoint/dataset
cache and the Python environment; home quota is too small and `/scratch-shared` is purged
after 14 days.

## Parallelisation and job duration

Single-GPU PyTorch jobs, run as SLURM arrays of 20–50 independent cells. Typical
duration 2–8 h, maximum ≈ 14 h (3B) / ≈ 40 h (8B grids); intermediate per-cell results
and a deterministic eval cache allow resumption. No multi-node jobs, no restart
workflow needed.

## Software

Own venv on `module load 2024 Python/3.12.3`: torch 2.4.1 (CUDA 12.1), transformers,
datasets, accelerate. Public HuggingFace checkpoints (MergeBench, FusionBench,
meta-llama under accepted license). No installation support needed.
