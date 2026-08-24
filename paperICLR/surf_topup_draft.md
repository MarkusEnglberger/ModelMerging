SURF Small Compute application (NWO) — Snellius GPU
====================================================

Form: servicedesk.surf.nl -> "Small Compute applications (NWO)". New project. Plain text
only. Items in [brackets] need your input.

Requested: 800,000 SBU on Snellius GPU nodes (gpu_h100 / gpu_a100), 1 year, default
storage and support.


APPLICANT
---------
Markus Englberger, Eindhoven University of Technology, [department/group].
[Temporary appointment -> digital signature by supervisor with permanent appointment:
name, e-mail.]
Project title: Expert-anchored gradient descent for model merging with small labeled
replay buffers.


SCIENTIFIC PURPOSE
------------------
Model merging combines several fine-tuned experts of one pretrained model into a single
model by merging, but merges degrade under task interference. We propose
a method that repairs a merged model using only 8-16 labeled examples per task: a few
gradient steps that move the merged model toward each expert, but only in the parameters
that a first-order attribution identifies as responsible for that task's loss, and never
beyond the expert itself. The method has been validated on encoder benchmarks (GLUE-8 with
RoBERTa, two CLIP/ViT suites).

The requested compute is for the decoder-only LLM track of the study, following the
MergeBench protocol with published full-parameter experts (math, coding,
instruction-following, multilingual) at two model scales, Llama-3.2-3B and Llama-3.1-8B.
For each scale we run the full grid of checkpoint-only and data-dependent merge baselines,
the proposed refinement and its ablations from every initialization, hyperparameter
selection on a disjoint buffer, replication over five replay-buffer seeds, and held-out
retention and pretraining-drift probes. This is by far the most expensive part of the
study because evaluation is generation-based.


SBU JUSTIFICATION
-----------------
All jobs are single-GPU. H100 = 192 SBU/GPU-hour, A100 = 128 SBU/GPU-hour. Costs are
extrapolated from our pilot runs on Snellius: one 3B grid run (learning-rate x sweep grid
for one initialization, selection, evaluation of the selected configuration) takes about
5 H100-hours, one 8B grid run about 15 H100-hours.

Each scale requires about 150 grid runs (baselines, refinement and ablations from 6
initializations, 2 data budgets, 5 seeds, probes, re-runs of boundary selections).

- LLM track, 3B: 150 x 5 h x 192 = 145,000 SBU
- LLM track, 8B: 150 x 15 h x 192 = 430,000 SBU
- Encoder-track revision experiments (A100): 250 x 3 h x 128 = 95,000 SBU

Total about 670,000 SBU. We request 800,000 SBU, because the
cost of generation-based evaluation varies with the selected sweep horizon and decoding
length.


MEMORY
------
Standard GPU nodes: one GPU with 16 cores and up to 180 GB host memory per job.


STORAGE
-------
Default storage is sufficient: public checkpoints (about 150 GB) and datasets are cached
on /scratch-shared.


PARALLELISATION AND JOB DURATION
--------------------------------
Single-GPU PyTorch jobs run as SLURM arrays of 20-50 independent configurations.
Typical duration 2-8 hours, maximum about 40 hours; per-configuration results are cached
so interrupted grids resume. No multi-node jobs necessary.


SOFTWARE
--------
HuggingFace transformers/datasets; public HuggingFace checkpoints. No installation support needed.
