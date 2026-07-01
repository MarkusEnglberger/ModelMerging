"""Task vectors and the task-arithmetic merge.

    tau_i      = theta_i - theta_0                       (Eq. 1/3)
    theta^(0)  = theta_0 + sum_i lambda_i * tau_i        (Eq. 4)
"""

from typing import Dict, List

from .models import ParamDict, pd_clone, pd_sub, pd_axpy_


def task_vector(expert_encoder: ParamDict, base_encoder: ParamDict) -> ParamDict:
    """tau = theta_expert - theta_0 (Eq. 1)."""
    return pd_sub(expert_encoder, base_encoder)


def task_arithmetic_merge(base_encoder: ParamDict,
                          task_vectors: Dict[str, ParamDict],
                          lambdas: Dict[str, float]) -> ParamDict:
    """theta^(0) = theta_0 + sum_i lambda_i tau_i (Eq. 4)."""
    merged = pd_clone(base_encoder)
    for name, tau in task_vectors.items():
        pd_axpy_(merged, lambdas[name], tau)
    return merged
