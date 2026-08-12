# Archived job scripts

Job scripts from completed campaign phases (proof-of-concept, gate diagnostics,
MaTS/MergeBench tracks, the pre-n+n evaluation-selected grids, one-off
bracketing and multi-seed studies). They are kept for provenance -- the
results/compare/*.json files they produced are cited in the paper -- but they
are not part of the current protocol and several predate the n+n split, so do
not resubmit them without checking their flags against scripts/merge_baselines.py.

The active protocol lives one directory up: grid_nn.sbatch (canonical n+n
grid), *_matched_s200.* (the budget-matched S<=200 columns), and
nn_split_smoke.sbatch (pipeline validation).
