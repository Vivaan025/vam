"""FrameExit's classifier as a `VideoActionModel` (experiments 017 / 018).

FrameExit (Ghodrati, Ehteshami Bejnordi, Habibian, CVPR 2021) is the shared backbone of the prior-work
comparison: OCSampler ports its classifier weights, and its released checkpoints define the ActivityNet-1.3
and Mini-Kinetics protocols the literature reports on (10 frames sampled uniformly at 1 fps / 5 fps, short
side 224, centre crop 224, ImageNet normalisation).

The network is a frozen ImageNet ResNet-50 (torchvision, classifier removed) -> MLP(2048 -> 4096, BN, ReLU)
per frame -> element-wise max over the frames seen so far -> one linear 200-way classifier per frame count
(`classifiers[t]` was trained on t + 1 frames). Early exit: the first frame exits on softmax confidence,
later frames on a learned gate over (current, previous) pooled features, the last frame always exits.

Two ways to drive it:

- `mode="fixed"` (default): the sampler decides which K frames arrive, the model pools them and answers with
  `classifiers[K - 1]`. This is how every sampler in `vam` (uniform, keyframe, budgeted, the re-implemented
  prior samplers) shares one classifier. `classifiers[K - 1]` was trained on FrameExit's own frame order, so
  this is the "classifier fixed, sampler swapped" protocol stated in `docs/experiments/018-prior-samplers.md`.
- `exit_forward(batch)`: FrameExit's own policy on its own 10-frame order (`INPUT_ORDER`), returning the
  logits at exit and how many frames were used. Reproduces the published 76.1 mAP (017 H1).

Checkpoints: `activitynet_checkpoint_25gmac.pth`, `minikinetics_checkpoint_19.7gmac.pth` from the FrameExit
GitHub release v1.0 (Qualcomm, their licence); pass `checkpoint=` or set `VAM_FRAMEEXIT_DIR`.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from vam.core import ActionOutput, FrameBatch
from vam.models.base import VideoActionModel
from vam.models.registry import register
from vam.runtime.preprocess import Preprocessor

INPUT_ORDER = [5, 0, 9, 2, 7, 4, 6, 3, 8, 1]  # FrameExit's order over its 10 uniform frames
FIRST_THRESHOLD = {"activitynet": 0.99, "minikinetics": 0.95}
GMACS_PER_FRAME = 4.12  # ResNet-50 at 224, the number FrameExit's inference script reports with


class _Gate(nn.Module):
    """FrameExit's exiting gate over (current, previous) pooled features. Output 0 means exit."""

    def __init__(self, in_planes: int = 4096):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, 128, 1, bias=True)
        self.conv2 = nn.Conv2d(128, 64, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.linear = nn.Linear(128, 1, bias=True)

    def forward(self, cur: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        def branch(x):
            x = F.relu(self.bn1(self.conv1(x.view(x.shape[0], -1, 1, 1))))
            x = F.relu(self.bn2(self.conv2(x)))
            return torch.flatten(x, 1)

        out = torch.sigmoid(self.linear(torch.cat([branch(cur), branch(prev)], dim=1)))
        return (out >= 0.5).float()


class FrameExitModel(VideoActionModel):
    def __init__(self, dataset: str = "activitynet", checkpoint: str | os.PathLike | None = None,
                 device: str | None = None, dtype: torch.dtype = torch.float16, num_frames: int = 10,
                 num_classes: int = 200, imagenet_init: bool = True, fixed_head: int | None = None):
        super().__init__()
        import torchvision

        # OCSampler classifies its K chosen frames with the 10-frame head regardless of K
        self.fixed_head = fixed_head

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dev == "cpu":
            dtype = torch.float32
        self._dev = torch.device(dev)
        self.dataset = dataset
        self.default_num_frames = num_frames
        self.num_classes = num_classes
        self.first_threshold = FIRST_THRESHOLD.get(dataset, 0.99)
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if (imagenet_init and checkpoint is None) else None
        r50 = torchvision.models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(r50.children())[:-1])  # ... -> avgpool, (B,2048,1,1)
        self.mlp = nn.Sequential(nn.Linear(2048, 4096), nn.BatchNorm1d(4096), nn.ReLU())
        self.classifiers = nn.ModuleList([nn.Linear(4096, num_classes) for _ in range(num_frames)])
        self.exit_selector = nn.ModuleList([_Gate(4096) for _ in range(num_frames - 1)])
        self.labels: list[str] | None = None
        ck = checkpoint or self._default_checkpoint(dataset)
        if ck is not None:
            self.load_checkpoint(ck)
        self.eval().to(self._dev, dtype=dtype)
        self._dtype = dtype
        self.pre = Preprocessor(shortest_edge=224, crop=(224, 224), mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225), device=dev)

    # --- weights ---------------------------------------------------------------------------------------
    @staticmethod
    def _default_checkpoint(dataset: str) -> Path | None:
        d = os.environ.get("VAM_FRAMEEXIT_DIR")
        if not d:
            return None
        name = {"activitynet": "activitynet_checkpoint_25gmac.pth", "minikinetics": "minikinetics_checkpoint_19.7gmac.pth"}[dataset]
        p = Path(d) / name
        return p if p.exists() else None

    def load_checkpoint(self, path: str | os.PathLike) -> "FrameExitModel":
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        # FrameExit: mlp.layers.{0,1}  ->  mlp.{0,1}
        state = {k.replace("mlp.layers.", "mlp."): v for k, v in state.items()}
        missing, unexpected = self.load_state_dict(state, strict=False)
        unexpected = [k for k in unexpected if not k.endswith("num_batches_tracked")]
        if missing or unexpected:
            raise RuntimeError(f"FrameExit checkpoint mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
        self.checkpoint = str(path)
        return self

    # --- frames -> features ----------------------------------------------------------------------------
    def _pixel_values(self, batch: FrameBatch) -> torch.Tensor:
        frames = batch.frames
        if batch.is_channels_first:
            frames = (frames * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        return self.pre(frames).to(self._dtype)  # (T,3,224,224)

    @torch.inference_mode()
    def frame_features(self, batch: FrameBatch) -> torch.Tensor:
        """Per-frame 4096-d features after the MLP, (T, 4096) float32."""
        x = self.backbone(self._pixel_values(batch)).flatten(1)
        return self.mlp(x).float()

    def logits_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        """Pool T frame features by max and classify with the head trained for T frames (capped at 10)."""
        t = int(feats.shape[0])
        z = feats.max(dim=0).values
        head = self.classifiers[(self.fixed_head or min(t, self.default_num_frames)) - 1]
        return head(z.to(head.weight.dtype)).float()

    @torch.inference_mode()
    def forward_frames(self, batch: FrameBatch) -> ActionOutput:
        feats = self.frame_features(batch)
        return ActionOutput(logits=self.logits_from_features(feats).cpu(), labels=self.labels, features=feats.cpu())

    # --- FrameExit's own policy -------------------------------------------------------------------------
    @torch.inference_mode()
    def exit_forward(self, batch: FrameBatch, order: list[int] | None = None) -> tuple[torch.Tensor, int]:
        """Run FrameExit's early exit over the batch in its training order. Returns (logits, frames used)."""
        feats = self.frame_features(batch).to(self._dtype)  # heads and gates run in the model dtype
        t_total = int(feats.shape[0])
        order = order if order is not None else (INPUT_ORDER if t_total == 10 else list(range(t_total)))
        z_prev = None
        for t, i in enumerate(order):
            z = feats[i]
            if t > 0:
                z = torch.maximum(z, z_prev)
            if t == 0:
                y = self.classifiers[0](z)
                exited = bool(torch.softmax(y.float(), 0).max() > self.first_threshold)
            elif t < len(order) - 1:
                exited = bool(self.exit_selector[t - 1](z[None], z_prev[None])[0, 0] == 0)
            else:
                exited = True
            if exited:
                y = self.classifiers[t](z) if t > 0 else y
                return y.float().cpu(), t + 1
            z_prev = z
        raise AssertionError("unreachable")

    @property
    def feature_dim(self) -> int:
        return 4096


@register("frameexit-activitynet")
def _fe_anet(**kw):
    return FrameExitModel("activitynet", **kw)


@register("frameexit-minikinetics")
def _fe_minik(**kw):
    return FrameExitModel("minikinetics", **kw)
