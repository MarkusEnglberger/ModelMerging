SURF Small Compute application (NWO) — Snellius GPU
====================================================

Form: servicedesk.surf.nl -> "Small Compute applications (NWO)". New project. Plain text
only. [brackets] need your input.

Requested: 800,000 SBU on Snellius GPU nodes (gpu_h100 / gpu_a100), 1 year, default
storage and support.


APPLICANT
---------
Markus Englberger, Eindhoven University of Technology, [department/group].
[Temporary appointment -> digital signature by supervisor with permanent appointment.]
Project title: Expert-anchored gradient descent for model merging with small labeled
replay buffers.


SCIENTIFIC PURPOSE
------------------
Model merging combines several fine-tuned experts of one pretrained model into a single
model, but merges degrade under task interference. We propose a method that repairs a
merged model using only 8-16 labeled examples per task: a few gradient steps toward each
expert, restricted to the parameters a first-order attribution identifies as responsible
for that task's loss, and never moving beyond the expert itself. It is validated on
encoder benchmarks (GLUE-8 with RoBERTa, two CLIP/ViT suites).

The requested compute extends the study to large language models, using the MergeBench
suite (He et al., NeurIPS 2025): four fully fine-tuned experts (math, coding, instruction
following, multilingual) for both Llama-3.2-3B and Llama-3.1-8B, with generation-based
evaluation (GSM8K, MBPP, IFEval, multilingual QA). At both model sizes we merge the four
experts with the standard merging methods, apply our repair and its ablations on top of
each merge, select hyperparameters on held-out examples, and repeat every configuration
over five random draws of the labeled examples. This is the expensive part: every
configuration must be evaluated by generating answers with the model.


LOCAL / TIER-2 FACILITIES
-------------------------
The encoder experiments ran on the TU/e Umbrella cluster, whose GPUs are too small for
the LLM experiments: at most 24 GB (L4), and 16 GB on our departmental partition, against
the roughly 80 GB needed to merge an 8B model and compute full-model gradients. That
queue also allows only two GPUs per user, for a study of several hundred multi-hour jobs.
TU/e's DGX system (SPIKE-1) is granted only for short access windows after which all data
is deleted.


SBU JUSTIFICATION
-----------------
All jobs are single-GPU (H100 = 192 SBU/GPU-hour, A100 = 128 SBU/GPU-hour). From our
pilot runs, one grid run (learning-rate x sweep grid for one initialization, selection,
evaluation) takes about 5 H100-hours at 3B and 15 at 8B, and each model size needs about
150 grid runs (baselines, repair and ablations from 6 initializations, 2 data budgets,
5 seeds, re-runs of boundary selections).

- LLM track, 3B: 150 x 5 h x 192 = 145,000 SBU
- LLM track, 8B: 150 x 15 h x 192 = 430,000 SBU
- Encoder-track revision experiments (A100): 250 x 3 h x 128 = 95,000 SBU

Total about 670,000 SBU; we request 800,000 SBU, since the cost of generation-based
evaluation varies with the selected sweep horizon and decoding length.


MEMORY
------
One GPU, 16 cores and up to 180 GB host memory per job. The LLM jobs need an H100
(94 GB); the encoder jobs fit on an A100.


STORAGE
-------
Default storage is sufficient: public checkpoints (about 150 GB) and datasets are cached
on /scratch-shared.


PARALLELISATION AND JOB DURATION
--------------------------------
Single-GPU PyTorch jobs run as SLURM arrays of 20-50 independent configurations, 2-8
hours each and at most 40 hours; results are cached so interrupted grids resume. No
multi-node jobs.


SOFTWARE
--------
HuggingFace transformers/datasets and public HuggingFace checkpoints. No installation
support needed.
