"""Adapter for torchvision's Kinetics-400 video classifiers (experiment 014).

A second model family with its own conventions: R(2+1)D wants 112x112 crops from a 128-short-edge resize,
MViT-v2 wants 224x224 from 256, both 16 frames, both ImageNet-style normalisation that differs from
VideoMAE's. The adapter maps those onto the shared `Preprocessor` and the `FrameBatch -> ActionOutput`
contract; nothing above the model changes.
"""

from __future__ import annotations

import torch

from vam.core import ActionOutput, FrameBatch
from vam.models.base import VideoActionModel
from vam.models.registry import register
from vam.runtime.preprocess import Preprocessor

# name -> (torchvision builder, weights enum name, shortest edge, crop, frames)
_SPECS = {
    "r2plus1d_18": ("r2plus1d_18", "R2Plus1D_18_Weights", 128, 112, 16),
    "r3d_18": ("r3d_18", "R3D_18_Weights", 128, 112, 16),
    "mvit_v2_s": ("mvit_v2_s", "MViT_V2_S_Weights", 256, 224, 16),
}


class TorchvisionVideoClassifier(VideoActionModel):
    def __init__(self, arch: str = "r2plus1d_18", device: str | None = None, dtype: torch.dtype = torch.float16):
        super().__init__()
        import torchvision.models.video as tvv

        if arch not in _SPECS:
            raise KeyError(f"unknown torchvision video arch {arch!r}; known: {sorted(_SPECS)}")
        builder, weights_name, short, crop, frames = _SPECS[arch]
        weights = getattr(tvv, weights_name).KINETICS400_V1
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dev == "cpu":
            dtype = torch.float32
        self.model = getattr(tvv, builder)(weights=weights).eval().to(dev, dtype=dtype)
        self._dev = torch.device(dev)
        self.arch = arch
        self.default_num_frames = frames
        meta = weights.meta
        self.labels = list(meta["categories"])
        tf = weights.transforms()  # carries mean/std/resize/crop for this checkpoint
        self.pre = Preprocessor(shortest_edge=int(getattr(tf, "resize_size", [short])[0]),
                                crop=tuple(getattr(tf, "crop_size", [crop, crop])[:2]) if hasattr(tf, "crop_size") else (crop, crop),
                                mean=tuple(float(x) for x in tf.mean), std=tuple(float(x) for x in tf.std), device=dev)
        # the classifier head is the last Linear; everything before it is the feature extractor
        self._head_name = "fc" if hasattr(self.model, "fc") else "head"

    def _pixel_values(self, batch: FrameBatch) -> torch.Tensor:
        """(T,H,W,3) uint8 -> (1, 3, T, H, W) in the model dtype."""
        frames = batch.frames
        if batch.is_channels_first:
            frames = (frames * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        t, n = frames.shape[0], self.default_num_frames
        if t > n:
            frames = frames[torch.linspace(0, t - 1, n).round().long()]
        elif t < n:
            frames = torch.cat([frames, frames[-1:].expand(n - t, *frames.shape[1:])], dim=0)
        x = self.pre(frames)                      # (T,3,H,W) float32
        return x.permute(1, 0, 2, 3).unsqueeze(0).to(self.model.fc.weight.dtype if hasattr(self.model, "fc")
                                                   else next(self.model.parameters()).dtype)

    @torch.inference_mode()
    def forward_frames(self, batch: FrameBatch) -> ActionOutput:
        logits = self.model(self._pixel_values(batch))
        return ActionOutput(logits=logits[0].float().cpu(), labels=self.labels)

    @torch.inference_mode()
    def features(self, batch: FrameBatch) -> torch.Tensor:
        """Pooled features before the classifier, for `LinearProbe`."""
        x = self._pixel_values(batch)
        m = self.model
        if hasattr(m, "fc"):  # resnet-style (r3d, r2plus1d): stem -> layers -> avgpool -> fc
            x = m.stem(x)
            for layer in (m.layer1, m.layer2, m.layer3, m.layer4):
                x = layer(x)
            return m.avgpool(x).flatten(1)[0].float()
        # MViT: run everything but the head's final linear
        head = m.head
        m.head = torch.nn.Identity()
        try:
            feat = m(x)
        finally:
            m.head = head
        return feat[0].float()

    @property
    def feature_dim(self) -> int:
        m = self.model
        lin = m.fc if hasattr(m, "fc") else [l for l in m.head.modules() if isinstance(l, torch.nn.Linear)][-1]
        return int(lin.in_features)


@register("r2plus1d-18-k400")
def _r2plus1d(**kw):
    return TorchvisionVideoClassifier("r2plus1d_18", **kw)


@register("r3d-18-k400")
def _r3d(**kw):
    return TorchvisionVideoClassifier("r3d_18", **kw)


@register("mvit-v2-s-k400")
def _mvit(**kw):
    return TorchvisionVideoClassifier("mvit_v2_s", **kw)
