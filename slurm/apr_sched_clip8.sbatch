#!/bin/bash
#SBATCH --job-name=apr-sched-c8
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=20:00:00
#SBATCH --output=logs/apr_sched_clip8-%j.out
#SBATCH --error=logs/apr_sched_clip8-%j.out
#
# Is APR better with a CONSTANT rate and FIXED task order, or with ANNEALING
# and RANDOM order? Rebuilds the whole CLIP-8 n=16+16 column with those two
# packages as the arms, across three replay seeds.
#
#   sbatch --export=ALL,SEED=0,MODE=merges slurm/apr_sched_clip8.sbatch
#   sbatch --export=ALL,SEED=0,MODE=base   slurm/apr_sched_clip8.sbatch
#
# Why this is unresolved: the two-initialization pilot (job 25482853) came back
# with OPPOSITE signs. From TIES the annealed package won on the selection
# buffer and on evaluation, 0.548 -> 0.615, overtaking GD's 0.599 and flipping
# that row. From task arithmetic it won the selection buffer by a clear margin
# and LOST on evaluation, 0.524 -> 0.495. So the effect is real, initialization-
# dependent, and currently searched on only two of six rows.
#
# The two arms are packages, not single knobs: merge_baselines.py consults
# --apr_order only for non-constant schedules, so "constant" implies the
# config's fixed order and "cosine" implies re-randomized order per sweep.
# That is the same pairing the n=64 long-horizon control used, so the result
# stays comparable to Table tab:longapr.
#
# The ungated arm is deliberately NOT re-run: those cells already exist at seed
# 0 and the compute is better spent on the open question. They stay in the
# table as single-seed values.
#
# S grid: cosine cannot reuse trajectories (its schedule depends on the
# horizon), so the annealed arm costs sum(S) per learning rate where constant
# costs max(S). S=100 buys the interior check on the S=50 selections that are
# currently daggered; S=200 would more than double the annealed arm for a
# horizon the pretrained S<=200 run already suggests is past the optimum.
#
# Learning rates 1..32 bracket both packages. The pilot's annealed optima were
# lr=16 (from TA) and lr=4 (from TIES) at S=50; offering S=100 halves the
# matched rate, so the grid extends DOWN rather than up.
#
# Seeds are 0/10/20, not 0/1/2: the selection buffer defaults to probe_seed+1,
# so consecutive probe seeds would make one run's training buffer another
# run's selection buffer. Seed 0 reproduces the existing column, which is a
# free check that selection is deterministic given the same draw.

set -euo pipefail
source slurm/common.sh

: "${SEED:=0}"
: "${MODE:?set MODE (merges|base)}"

OUT="results/compare/grid_nn_clip8_${MODE}_n16_sched_s${SEED}.json"

COMMON=(--config configs/clip8.yaml
        --n_probe 16 --n_select 16
        --probe_seed "$SEED"
        --steps 5 20 50 100
        --select_by mean_normret
        --apr_lrs 1 2 4 8 16 32
        --apr_schedules constant cosine
        --apr_order random
        --gd_lr_min_frac 0.05
        --ada_data probe_buffer
        --out "$OUT")

echo "[job] CLIP-8 schedule comparison MODE=$MODE seed=$SEED -> $OUT"

if [ "$MODE" = base ]; then
  # --ta_lams 0 makes merge:TA@l0 identical to theta_0, so refining "from ta"
  # refines from the PRETRAINED model.
  python scripts/merge_baselines.py "${COMMON[@]}" \
    --ta_lams 0 \
    --skip_families ties dare dareties della bc ls ada \
    --refine_from ta
else
  python scripts/merge_baselines.py "${COMMON[@]}" \
    --skip_families dare della ls \
    --refine_from ta ties dareties bc ada
fi

echo "[done] -> $OUT"
