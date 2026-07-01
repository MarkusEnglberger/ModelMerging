"""Cross-task interference matrix (Eq. 22).

    C_{j<-i} = < grad_theta L_j(theta; D_probe_j),  u_i >

Negative entries => task i's update is locally predicted to *help* task j;
positive entries => predicted interference. This directly tests whether the
attribution gate is only self-protective or also aligns across related tasks.
"""

from typing import Dict, List
import torch

from .models import ParamDict


def dot(a: ParamDict, b: ParamDict) -> float:
    return float(sum((a[k] * b[k].to(a[k].device)).sum() for k in a))


def interference_matrix(grads: Dict[str, ParamDict],
                        updates: Dict[str, ParamDict]) -> Dict[str, Dict[str, float]]:
    """C[j][i] = <grad_j, u_i> for all task pairs.

    `grads[j]`   : encoder gradient for task j at the current state.
    `updates[i]` : the applied update u_i for task i at that state.
    """
    tasks = list(grads.keys())
    C: Dict[str, Dict[str, float]] = {j: {} for j in tasks}
    for j in tasks:
        for i in tasks:
            C[j][i] = dot(grads[j], updates[i])
    return C
