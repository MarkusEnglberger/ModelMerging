# Archived job scripts

Job scripts from completed campaign phases (proof-of-concept, gate diagnostics,
MaTS/MergeBench tracks, the pre-n+n evaluation-selected grids, one-off
bracketing and multi-seed studies). They are kept for provenance -- the
results/compare/*.json files they produced are cited in the paper. Superseded
tuning drivers that evaluated every grid cell have been removed, so some old
launchers are intentionally non-runnable. They document historical commands,
not supported entry points.

The active protocol lives one directory up: grid_nn.sbatch (canonical n+n
grid), *_matched_s200.* (the budget-matched S<=200 columns), and
nn_split_smoke.sbatch (pipeline validation).
