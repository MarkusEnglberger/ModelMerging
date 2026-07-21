"""CLIP/ViT vision track for the APR merging framework.

This is the second-modality benchmark from the proposal (H5): the standard
task-vector image-classification suite (SUN397, Cars, RESISC45, EuroSAT, SVHN,
GTSRB, MNIST, DTD) with a CLIP ViT-B/32 backbone.

The APR encoder/head split maps onto CLIP exactly as in the FusionBench
`HFCLIPClassifier` setup:

  * mergeable encoder  = the CLIP **vision tower** (``vision_model.*``), i.e. the
    part that is fine-tuned per task (the `tanganke/clip-vit-base-patch32_<task>`
    checkpoints are bare ``CLIPVisionModel`` fine-tunes);
  * frozen task head   = base CLIP ``visual_projection`` + a **zero-shot text
    classifier** (class-name embeddings from the CLIP text tower) + ``logit_scale``.

The head is fixed (never merged, never gradient-updated), exactly like the
per-task GLUE classification heads. Everything downstream -- task vectors, the
Algorithm-1 refinement, gradients, metrics, interference -- is reused unchanged
because it only ever touches the mergeable encoder param-dict and a model that
exposes ``.logits`` / ``.loss``.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (CLIPModel, CLIPVisionModel, CLIPImageProcessor,
                          CLIPTokenizer)

from .models import ParamDict

# ---------------------------------------------------------------------------
# Task registry: dataset ids, splits, and zero-shot prompt templates.
#
# Class *names* are read from the HF dataset's ClassLabel feature (they already
# match the FusionBench / task_vectors descriptive names). Templates are the
# canonical per-dataset prompts from Ilharco et al.'s task_vectors release.
# ---------------------------------------------------------------------------

_DEFAULT_TEMPLATES = ["a photo of a {}."]

VISION_TEMPLATES = {
    "eurosat": [
        "a centered satellite photo of {}.",
        "a centered satellite photo of a {}.",
        "a centered satellite photo of the {}.",
    ],
    "mnist": ['a photo of the number: "{}".'],
    "gtsrb": [
        'a zoomed in photo of a "{}" traffic sign.',
        'a centered photo of a "{}" traffic sign.',
        'a close up photo of a "{}" traffic sign.',
    ],
    "svhn": [
        'a photo of the number: "{}".',
        'a street sign of the number: "{}".',
    ],
    "dtd": [
        "a photo of a {} texture.",
        "a photo of a {} pattern.",
        "a photo of a {} object.",
    ],
    "resisc45": [
        "satellite imagery of {}.",
        "a satellite photo of {}.",
        "an aerial photo of {}.",
    ],
    "sun397": [
        "a photo of a {}.",
        "a photo of the {}.",
    ],
    "cars": [
        "a photo of a {}.",
        "a photo of the {}.",
        "i love my {}!",
    ],
    # --- 14/20-task extension (TALL-masks suite) ---
    "cifar10": ["a photo of a {}."],
    "cifar100": ["a photo of a {}."],
    "stl10": ["a photo of a {}.", "a photo of the {}."],
    "food101": ["a photo of {}, a type of food."],
    "fashion_mnist": ["a photo of a {}.", "an image of a {} (clothing)."],
    "flowers102": ["a photo of a {}, a type of flower."],
    "pets": ["a photo of a {}, a type of pet."],
    "emnist": ['a photo of the letter "{}".'],
    "kmnist": ["a photo of the character {}."],
    # clip-benchmark canonical prompts for the three no-ClassLabel (wds) tasks.
    "pcam": ["this is a photo of {}."],
    "fer2013": ["a photo of a {} looking face."],
    "rendered_sst2": ["a {} review of a movie."],
}


@dataclass(frozen=True)
class VisionTaskSpec:
    """Duck-types the fields of GLUE ``TaskSpec`` that the shared pipeline reads.

    The merge/refine/eval/metrics code only ever touches ``is_regression``,
    ``metric`` and ``num_labels`` on a spec, so a vision spec only needs those
    plus the dataset wiring used by the vision loader.
    """

    name: str               # canonical task id (eurosat, mnist, ...)
    dataset: str            # HF dataset id
    num_labels: int
    metric: str = "accuracy"
    dataset_config: Optional[str] = None  # e.g. svhn -> "cropped_digits"
    image_key: str = "image"
    label_key: str = "label"
    train_split: str = "train"
    eval_split: str = "test"
    # Explicit class names, ordered by integer label id. Only needed for datasets
    # whose label column is a plain int (no ClassLabel ``.names``) -- the three
    # webdataset tasks (pcam/fer2013/rendered_sst2). When None, names are read
    # from the dataset's ClassLabel feature (which already matches the expert).
    classnames: Optional[Tuple[str, ...]] = None

    @property
    def is_regression(self) -> bool:
        return False

    @property
    def templates(self) -> List[str]:
        return VISION_TEMPLATES.get(self.name, _DEFAULT_TEMPLATES)


# Explicit class names (ordered by label id) for the three webdataset tasks
# whose label column is a bare int with no ClassLabel .names. These are the
# canonical clip-benchmark names the tanganke experts were evaluated against;
# the prefetch accuracy-check (expert acc >> zero-shot) validates the ordering.
_PCAM_CLASSNAMES = ("lymph node",
                    "lymph node containing metastatic tumor tissue")
_FER2013_CLASSNAMES = ("angry", "disgusted", "fearful", "happy",
                       "neutral", "sad", "surprised")
_SST2_CLASSNAMES = ("negative", "positive")

# Canonical CLIP/ViT task-vector suite. `num_labels` is filled in from the
# dataset's ClassLabel feature at load time, so the literal here is just a hint.
VISION_TASKS = {
    # --- original 8-task task-arithmetic benchmark ---
    "sun397": VisionTaskSpec("sun397", "tanganke/sun397", 397),
    "cars": VisionTaskSpec("cars", "tanganke/stanford_cars", 196),
    "resisc45": VisionTaskSpec("resisc45", "tanganke/resisc45", 45),
    "eurosat": VisionTaskSpec("eurosat", "tanganke/eurosat", 10),
    "svhn": VisionTaskSpec("svhn", "svhn", 10, dataset_config="cropped_digits"),
    "gtsrb": VisionTaskSpec("gtsrb", "tanganke/gtsrb", 43),
    "mnist": VisionTaskSpec("mnist", "mnist", 10),
    "dtd": VisionTaskSpec("dtd", "tanganke/dtd", 47),
    # --- +6 -> 14-task suite (standard image + ClassLabel datasets) ---
    "cifar100": VisionTaskSpec("cifar100", "tanganke/cifar100", 100),
    "stl10": VisionTaskSpec("stl10", "tanganke/stl10", 10),
    "flowers102": VisionTaskSpec("flowers102", "dpdl-benchmark/oxford_flowers102", 102),
    "pets": VisionTaskSpec("pets", "timm/oxford-iiit-pet", 37),
    "pcam": VisionTaskSpec("pcam", "clip-benchmark/wds_vtab-pcam", 2,
                           image_key="webp", label_key="cls",
                           classnames=_PCAM_CLASSNAMES),
    "fer2013": VisionTaskSpec("fer2013", "clip-benchmark/wds_fer2013", 7,
                              image_key="jpg", label_key="cls",
                              classnames=_FER2013_CLASSNAMES),
    # --- +6 -> 20-task suite ---
    "emnist": VisionTaskSpec("emnist", "tanganke/emnist_letters", 26,
                             dataset_config="emnist-letters"),
    "cifar10": VisionTaskSpec("cifar10", "tanganke/cifar10", 10),
    "food101": VisionTaskSpec("food101", "ethz/food101", 101,
                              eval_split="validation"),
    "fashion_mnist": VisionTaskSpec("fashion_mnist", "zalando-datasets/fashion_mnist", 10,
                                    dataset_config="fashion_mnist"),
    "kmnist": VisionTaskSpec("kmnist", "tanganke/kmnist", 10,
                             dataset_config="kmnist"),
    "rendered_sst2": VisionTaskSpec("rendered_sst2", "clip-benchmark/wds_renderedsst2", 2,
                                    image_key="png", label_key="cls",
                                    classnames=_SST2_CLASSNAMES),
}

# Fine-tuned vision encoder (expert) checkpoints, FusionBench / tanganke release.
VISION_EXPERT_CKPT = {
    "sun397": "tanganke/clip-vit-base-patch32_sun397",
    "cars": "tanganke/clip-vit-base-patch32_stanford-cars",
    "resisc45": "tanganke/clip-vit-base-patch32_resisc45",
    "eurosat": "tanganke/clip-vit-base-patch32_eurosat",
    "svhn": "tanganke/clip-vit-base-patch32_svhn",
    "gtsrb": "tanganke/clip-vit-base-patch32_gtsrb",
    "mnist": "tanganke/clip-vit-base-patch32_mnist",
    "dtd": "tanganke/clip-vit-base-patch32_dtd",
    "cifar100": "tanganke/clip-vit-base-patch32_cifar100",
    "stl10": "tanganke/clip-vit-base-patch32_stl10",
    "flowers102": "tanganke/clip-vit-base-patch32_oxford_flowers102",
    "pets": "tanganke/clip-vit-base-patch32_oxford-iiit-pet",
    "pcam": "tanganke/clip-vit-base-patch32_pcam",
    "fer2013": "tanganke/clip-vit-base-patch32_fer2013",
    "emnist": "tanganke/clip-vit-base-patch32_emnist_letters",
    "cifar10": "tanganke/clip-vit-base-patch32_cifar10",
    "food101": "tanganke/clip-vit-base-patch32_food101",
    "fashion_mnist": "tanganke/clip-vit-base-patch32_fashion_mnist",
    "kmnist": "tanganke/clip-vit-base-patch32_kmnist",
    "rendered_sst2": "tanganke/clip-vit-base-patch32_rendered-sst2",
}

# Named suites (ordered as in the TALL-masks paper). The 8-task set is the
# original task-arithmetic benchmark; 14 and 20 are the standard extensions.
CLIP8_TASKS = ["sun397", "cars", "resisc45", "eurosat",
               "svhn", "gtsrb", "mnist", "dtd"]
CLIP14_TASKS = CLIP8_TASKS + ["cifar100", "stl10", "flowers102",
                              "pets", "pcam", "fer2013"]
CLIP20_TASKS = CLIP14_TASKS + ["emnist", "cifar10", "food101",
                               "fashion_mnist", "kmnist", "rendered_sst2"]
VISION_SUITES = {"8": CLIP8_TASKS, "14": CLIP14_TASKS, "20": CLIP20_TASKS}


def get_vision_task(name: str) -> VisionTaskSpec:
    key = name.lower().replace("-", "_")
    alias = {
        "stanford_cars": "cars", "stanfordcars": "cars",
        "oxford_flowers102": "flowers102", "flowers": "flowers102",
        "oxford_iiit_pet": "pets", "oxford_pet": "pets", "pet": "pets",
        "patch_camelyon": "pcam", "patchcamelyon": "pcam",
        "emnist_letters": "emnist",
        "fashionmnist": "fashion_mnist",
        "sst2": "rendered_sst2", "renderedsst2": "rendered_sst2",
        "rendered_sst": "rendered_sst2",
    }
    key = alias.get(key, key)
    if key not in VISION_TASKS:
        raise KeyError(f"Unknown vision task '{name}'. Known: {sorted(VISION_TASKS)}")
    return VISION_TASKS[key]


# ---------------------------------------------------------------------------
# Model wrapper: fine-tunable vision tower + frozen zero-shot head.
# ---------------------------------------------------------------------------

class ClipImageClassifier(nn.Module):
    """CLIP vision tower (mergeable) + frozen zero-shot classification head.

    forward(pixel_values, labels) -> object with ``.logits`` and ``.loss``,
    so the shared gradient/eval code treats it just like an HF classifier.

    Mergeable parameters live under ``base_model_prefix = "vision_model"``; the
    head (``visual_projection`` weight, the text-embedding buffer, ``logit_scale``)
    is frozen and excluded from merging by construction.
    """

    base_model_prefix = "vision_model"

    def __init__(self, vision_model: nn.Module, visual_projection: nn.Linear,
                 class_embeds: torch.Tensor, logit_scale: float):
        super().__init__()
        self.vision_model = vision_model              # CLIPVisionTransformer (mergeable)
        self.visual_projection = visual_projection    # frozen Linear (head)
        for p in self.visual_projection.parameters():
            p.requires_grad_(False)
        # zero-shot class text embeddings (C, proj_dim), L2-normalized; a buffer
        # so it is neither merged nor gradient-updated.
        self.register_buffer("class_embeds", class_embeds)
        self.logit_scale = float(logit_scale)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values=None, labels=None, **_):
        pooled = self.vision_model(pixel_values=pixel_values).pooler_output
        img = self.visual_projection(pooled)
        img = img / img.norm(p=2, dim=-1, keepdim=True)
        logits = self.logit_scale * img @ self.class_embeds.t()
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return _Output(logits=logits, loss=loss)


@dataclass
class _Output:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Shared CLIP assets: loaded once, reused to build every task's wrapper.
# ---------------------------------------------------------------------------

class VisionAssets:
    """Base CLIP pieces shared across tasks: vision tower (theta_0), frozen
    visual projection + logit scale, and the text tower used to build zero-shot
    heads. Created once per experiment in the pipeline."""

    def __init__(self, base_model: str, cache_dir: Optional[str] = None):
        self.base_model = base_model
        self.cache_dir = cache_dir
        clip = CLIPModel.from_pretrained(base_model, cache_dir=cache_dir)
        clip.eval()
        self._clip = clip
        self.processor = CLIPImageProcessor.from_pretrained(base_model, cache_dir=cache_dir)
        self.tokenizer = CLIPTokenizer.from_pretrained(base_model, cache_dir=cache_dir)
        self.proj_dim = clip.config.projection_dim
        self.logit_scale = float(clip.logit_scale.exp().item())
        # theta_0 = base vision tower state, keyed "vision_model.*" to match the
        # wrapper's encoder param names. Use named_parameters (not state_dict) so
        # the key set is exactly the mergeable parameters -- buffers such as
        # `position_ids` must NOT enter the merge or load_encoder_state breaks.
        self.base_vision_state: ParamDict = OrderedDict(
            (f"vision_model.{k}", v.detach().clone())
            for k, v in clip.vision_model.named_parameters())
        self._head_cache: Dict[str, torch.Tensor] = {}

    def _fresh_visual_projection(self) -> nn.Linear:
        proj = nn.Linear(self._clip.visual_projection.in_features,
                         self._clip.visual_projection.out_features, bias=False)
        proj.load_state_dict(self._clip.visual_projection.state_dict())
        return proj

    @torch.no_grad()
    def build_zero_shot_head(self, spec: VisionTaskSpec,
                             classnames: List[str]) -> torch.Tensor:
        """(C, proj_dim) L2-normalized class embeddings averaged over templates."""
        if spec.name in self._head_cache:
            return self._head_cache[spec.name]
        embeds = []
        for name in classnames:
            prompts = [t.format(name) for t in spec.templates]
            tok = self.tokenizer(prompts, padding=True, return_tensors="pt")
            feats = self._clip.get_text_features(**tok)          # (n_templates, dim)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            mean = feats.mean(dim=0)
            mean = mean / mean.norm(p=2)
            embeds.append(mean)
        head = torch.stack(embeds, dim=0)
        self._head_cache[spec.name] = head
        return head

    def build_vision_tower(self, checkpoint: str) -> nn.Module:
        """Load a fine-tuned (or base) CLIPVisionModel and return its inner
        CLIPVisionTransformer (params named ``vision_model.*`` once wrapped)."""
        vm = CLIPVisionModel.from_pretrained(checkpoint, cache_dir=self.cache_dir)
        return vm.vision_model

    def build_classifier(self, spec: VisionTaskSpec, classnames: List[str],
                         checkpoint: str) -> ClipImageClassifier:
        head = self.build_zero_shot_head(spec, classnames)
        tower = self.build_vision_tower(checkpoint)
        return ClipImageClassifier(tower, self._fresh_visual_projection(),
                                   head.clone(), self.logit_scale)


# ---------------------------------------------------------------------------
# Data: load an image task and build an image collator.
# ---------------------------------------------------------------------------

def load_vision_dataset(spec: VisionTaskSpec, cache_dir: Optional[str] = None):
    """Return (train_ds, eval_ds, classnames). Datasets keep an `image` column
    (PIL) and a renamed `labels` column; images are processed lazily in the
    collator so sampling/eval stay cheap."""
    kw = {} if spec.dataset_config is None else {"name": spec.dataset_config}
    ds = load_dataset(spec.dataset, cache_dir=cache_dir, **kw)
    # Class names come from the ClassLabel feature when present (order already
    # matches the expert); the webdataset tasks have a bare int label, so fall
    # back to the explicit names on the spec.
    label_feat = ds[spec.train_split].features[spec.label_key]
    classnames = getattr(label_feat, "names", None)
    if classnames is None:
        if spec.classnames is None:
            raise ValueError(
                f"Task '{spec.name}' label column '{spec.label_key}' has no "
                f"ClassLabel.names and no explicit classnames on the spec.")
        classnames = list(spec.classnames)
    keep = {spec.image_key, spec.label_key}
    drop = [c for c in ds[spec.train_split].column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    if spec.image_key != "image":
        ds = ds.rename_column(spec.image_key, "image")
    ds = ds.rename_column(spec.label_key, "labels")
    return ds[spec.train_split], ds[spec.eval_split], classnames


def make_image_collator(processor: CLIPImageProcessor):
    def collate(features: List[Dict]):
        images = [f["image"] for f in features]
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
        labels = torch.tensor([int(f["labels"]) for f in features], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}
    return collate
