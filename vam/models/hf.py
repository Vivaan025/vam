"""Adapter for Hugging Face video classification models (VideoMAE, TimeSformer, ViViT, ...).

vam never reimplements models. Anything with a video-classification head on the
Hub works through this one class.
"""

from __future__ import annotations

import torch

from vam.core import ActionOutput, FrameBatch
from vam.models.base import VideoActionModel
from vam.models.registry import register


class HFVideoClassifier(VideoActionModel):
    def __init__(self, name: str, device: str | None = None, dtype: torch.dtype = torch.float16):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModelForVideoClassification

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dev == "cpu":
            dtype = torch.float32
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForVideoClassification.from_pretrained(name, dtype=dtype)
        self.model.eval().to(dev)
        self._dev = torch.device(dev)
        cfg = self.model.config
        self.default_num_frames = int(getattr(cfg, "num_frames", 16))
        self.labels = [cfg.id2label[i] for i in range(len(cfg.id2label))]

    def forward_frames(self, batch: FrameBatch) -> ActionOutput:
        frames = batch.frames
        if batch.is_channels_first:
            frames = (frames * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        frames = self._fit_length(frames)
        inputs = self.processor(list(frames.cpu().numpy()), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._dev, dtype=self.model.dtype)
        out = self.model(pixel_values=pixel_values)
        return ActionOutput(logits=out.logits[0].float().cpu(), labels=self.labels)

    def _fit_length(self, frames: torch.Tensor) -> torch.Tensor:
        """HF video models need exactly default_num_frames; subsample or pad to match."""
        t, n = frames.shape[0], self.default_num_frames
        if t == n:
            return frames
        if t > n:
            idx = torch.linspace(0, t - 1, n).round().long()
            return frames[idx]
        pad = frames[-1:].expand(n - t, *frames.shape[1:])
        return torch.cat([frames, pad], dim=0)


@register("videomae-base-kinetics")
def _videomae(**kw):
    return HFVideoClassifier("MCG-NJU/videomae-base-finetuned-kinetics", **kw)


@register("videomae-small-kinetics")
def _videomae_small(**kw):
    return HFVideoClassifier("MCG-NJU/videomae-small-finetuned-kinetics", **kw)


@register("timesformer-base-k400")
def _timesformer(**kw):
    return HFVideoClassifier("facebook/timesformer-base-finetuned-k400", **kw)
