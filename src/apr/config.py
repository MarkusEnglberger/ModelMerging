"""Experiment configuration.

A single YAML file fully specifies a merge+refine+eval run. Defaults below match
the method described in the proposal (gate threshold 0, clip fraction gamma,
sequential immediate updates).
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import yaml


@dataclass
class RefineConfig:
    steps: int = 5  # S, number of sweeps (S=0 => plain task-arithmetic merge)
    lr: float = 1.0  # eta
    clip_frac: float = 0.5  # gamma; clip each per-coord step to gamma*|v|
    gate_eps: float = 0.0  # epsilon_gate
    order: str = "fixed"  # task order per sweep: fixed | cyclic | random
    aggregated: bool = False  # if True, aggregated-U ablation (no immediate apply)
    gate_mode: str = "coordinate"  # coordinate | tensor | none | inverted | random
    # base update vector form (before gating/clipping):
    #   gated_grad : -g*|v|         (the proposed method, Eq. 12)
    #   grad       : -g             (ordinary replay gradient descent)
    #   interp     :  v             (direct expert interpolation, u ~ m*v)
    update_mode: str = "gated_grad"
    clip_mode: str = "vdist"  # vdist => clamp to +/- gamma*|v|; none => no clip
    rms_normalize: bool = False  # tensor-wise RMS normalize u before applying
    random_gate_density: Optional[float] = None  # for gate_mode=random; else match
    freeze_first_gates: bool = False  # reuse sweep-0 gates (expansion-point ablation)
    # learning-rate schedule across sweeps: "constant" | "cosine" | "linear".
    # cosine/linear decay the effective lr from `lr` at sweep 0 to lr*lr_min_frac
    # at the final sweep (long-horizon control: big early steps, fine late steps).
    lr_schedule: str = "constant"
    lr_min_frac: float = 0.05


@dataclass
class DataConfig:
    n_probe: int = 64  # replay examples per task
    max_length: int = 128
    eval_batch_size: int = 64
    # DataLoader workers for evaluation. 0 keeps GLUE behaviour unchanged; the
    # CLIP/ViT track sets this >0 so PIL decode+resize of images runs in parallel
    # and does not starve the GPU (the dominant cost for large image test sets).
    eval_num_workers: int = 0
    probe_seed: int = 0  # sampling seed for replay buffers
    class_balanced: bool = True
    cache_dir: Optional[str] = None
    # where to draw the probe split from: "train" (held-out from train) keeps the
    # eval/validation split untouched for reporting.
    probe_source: str = "train"
    max_eval: Optional[int] = None  # truncate eval split (debugging/fast iteration)


@dataclass
class ExpertConfig:
    name: str  # task id
    # HF model id or local path to a fine-tuned expert checkpoint. For the CLIP
    # vision track this may be left null to fall back to the standard tanganke
    # checkpoint for the task (see vision.VISION_EXPERT_CKPT).
    checkpoint: Optional[str] = None
    lam: float = 0.3  # task-arithmetic coefficient lambda_i


@dataclass
class ExperimentConfig:
    base_model: str  # shared pretrained checkpoint (theta_0), e.g. roberta-base
    experts: List[ExpertConfig]
    # which modality backend to use: "glue" (RoBERTa multi-head), "clip"
    # (CLIP/ViT vision track), "t5" (flan-T5 text-to-text GLUE, shared LM head)
    # or "causal_lm" (decoder-only MergeBench track). Selects pipeline loaders.
    modality: str = "glue"
    # parameter dtype for model loading / task vectors. "float32" everywhere on
    # the <=1B tracks; the 2-3B causal_lm track can use "bfloat16" to halve the
    # CPU-RAM footprint (5 experts x 3B fp32 ~ 60 GB) at some gate-sign noise.
    model_dtype: str = "float32"
    refine: RefineConfig = field(default_factory=RefineConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 0
    device: str = "cuda"
    output_dir: str = "results/run"
    tag: str = "apr"

    @property
    def task_names(self) -> List[str]:
        return [e.name for e in self.experts]

    @staticmethod
    def from_yaml(path: str) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        experts = [ExpertConfig(**e) for e in raw.pop("experts")]
        refine = RefineConfig(**raw.pop("refine", {}))
        data = DataConfig(**raw.pop("data", {}))
        return ExperimentConfig(experts=experts, refine=refine, data=data, **raw)

    def to_dict(self) -> Dict:
        return asdict(self)
