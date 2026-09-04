Dear Douwe van der Wal,

Thank you for the response. Below are our answers to the three questions.

1. Each job keeps one model in GPU memory and alternates
between updating its weights (merging arithmetic and gradient steps on the full
parameter vector) and evaluating it. The updates run in PyTorch with HuggingFace
transformers; the generation-based evaluations run in vLLM. Weights are kept unquantized in bf16 throughout: the method needs exact
weight-space differences to the experts and gradients through the full model, and the
evaluations must score exactly the weights the method produces — quantized inference
would add score noise of the same magnitude as the method differences under study, and
would require re-quantizing after every weight update. Sampling nvidia-smi every two seconds across a run with the 3B model on an
H100, the GPU is at 90-100% utilization for two thirds of the compute time (81% on
average), with power peaking at 690 W of the 700 W limit.


2. One "run" is one job that sweeps the hyperparameter grid of one configuration,
selects the best cell on held-out examples, and evaluates only that selection:

   - construction and hyperparameter selection of the 5 merge baselines (i.e. merges obtained with existing methods), at 2 data
     budgets: ~10 runs;
   - refinement grids: 6 initializations (pretrained + 5 merge baselines) x 3 methods
     (our method + 2 ablation baselines) x 2 data budgets x 3 independent draws of the
     labeled examples = 108 runs. Each draw repeats the full select-then-evaluate
     procedure, so the reported draw-to-draw variance includes selection variance;
   - held-out retention probes (i.e. testing how good the method is at retaining unrelated capabilites of the pretrained model) for the selected configurations: estimated at around 30 runs.
   That leads to around ~150 runs per model scale.

3. Lowering to 700,000 SBU works for us.

Kind regards,
Markus Englberger
