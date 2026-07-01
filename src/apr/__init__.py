"""Attribution-Patching-guided Replay refinement (APR) for task-vector merging.

Implements the method described in "Attribution-Patching-Guided Replay
Refinement for Task-Vector Merging":

  - task vectors and task-arithmetic merge (Eqs. 1-4)
  - the attribution-patching gate (Eqs. 9-12)
  - sequential expert-direction-gated replay refinement (Algorithm 1)
  - normalized-retention metrics (Eq. 21) and the cross-task interference
    matrix (Eq. 22).
"""

__version__ = "0.1.0"
