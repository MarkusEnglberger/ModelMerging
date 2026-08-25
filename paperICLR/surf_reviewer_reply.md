Dear Douwe van der Wal,

Thank you for the quick review. Answers to your three questions:

1. Software and GPU utilization. We use plain PyTorch with HuggingFace transformers,
not vLLM: each job holds a single model in GPU memory that is alternately updated
(merging arithmetic and gradient steps on the full parameter vector) and evaluated, so
the weights change between every evaluation. Everything runs in bf16; evaluation decodes
in batches of 48 sequences, and deterministic evaluations (base model, experts, unrefined
merges) are cached and never recomputed. Sampling nvidia-smi during a generation job on
an H100 shows 98-100% utilization at 660-695 W of the 700 W limit. Your question also
prompted us to profile the evaluation more carefully, and we found that the published
MergeBench expert checkpoints ship with the KV cache disabled in their generation
config, which our evaluation had inherited; with the cache enabled, decoding is 6.5x
faster at 256 generated tokens (2,230 vs. 340 tokens/s) and more at longer outputs.
Since generation was about 85% of a run, the per-run cost in our estimate is
conservative by a factor of roughly 3, which makes the reduced budget below
comfortable.

2. Breakdown of the ~150 runs per model scale. One "grid run" covers one
initialization x one method, including its hyperparameter sweep, selection and final
evaluation:
   - construction and hyperparameter selection of the 5 merge baselines, at 2 data
     budgets: ~10 runs;
   - refinement grids: 6 initializations (pretrained + the 5 merges) x 3 methods (our
     repair + 2 ablation baselines) x 2 data budgets = 36 runs;
   - replication over 4 additional random draws of the labeled examples at the selected
     hyperparameters: 6 x 3 x 4 = 72 runs (cheaper than full grids, since only the
     selected configuration is re-run);
   - held-out retention probes for every selected configuration and re-runs of grids
     whose selected hyperparameter landed on the grid boundary (in our experience ~25%
     of grids): ~30 runs.
   That totals ~148 runs; since the replication runs cost less than the 5 (3B) /
   15 (8B) H100-hours of a full grid run, the estimate of 150 x full-grid cost already
   contains some internal margin.

3. Lowering to 700,000 SBU works for us — thank you for pointing out the extension
route via the servicedesk; we will use it with a concrete SBU calculation if the
remaining experiments require it.

Kind regards,
Markus Englberger
