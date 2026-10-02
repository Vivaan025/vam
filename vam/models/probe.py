"""Linear probe: a frozen backbone's pooled features + one linear layer.

Why this exists: pretrained video models ship with Kinetics heads, but the data we
can download and label is UCF101. Fitting a linear head on frozen features gives a
real accuracy axis. Crucially the head is trained once with one sampler and then held
fixed, so accuracy differences between samplers are caused only by frame selection.
"""

from __future__ import annotations

from pathlib import Path

import torch

from vam.core import ActionOutput, FrameBatch
from vam.models.base import VideoActionModel
from vam.models.hf import HFVideoClassifier


class HFFeatureExtractor(HFVideoClassifier):
    """Same preprocessing as the HF classifier, but returns pooled backbone features."""

    @torch.inference_mode()
    def features(self, batch: FrameBatch) -> torch.Tensor:
        frames = batch.frames
        if batch.is_channels_first:
            frames = (frames * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        frames = self._fit_length(frames)
        inputs = self.processor(list(frames.cpu().numpy()), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._dev, dtype=self.model.dtype)
        backbone = getattr(self.model, self.model.base_model_prefix)
        hidden = backbone(pixel_values=pixel_values).last_hidden_state  # (1, tokens, dim)
        return hidden.mean(dim=1)[0].float()  # (dim,)


class LinearProbe(VideoActionModel):
    def __init__(self, extractor: HFFeatureExtractor, num_classes: int, class_names: list[str] | None = None):
        super().__init__()
        self.extractor = extractor
        self.default_num_frames = extractor.default_num_frames
        dim = extractor.model.config.hidden_size
        self.head = torch.nn.Linear(dim, num_classes).to(extractor._dev)
        self.labels = class_names

    def forward_frames(self, batch: FrameBatch) -> ActionOutput:
        feat = self.extractor.features(batch)
        logits = self.head(feat.to(self.head.weight.device))
        return ActionOutput(logits=logits.float().cpu(), labels=self.labels, features=feat.cpu())

    @staticmethod
    def fit_head(features: torch.Tensor, labels: torch.Tensor, num_classes: int, epochs: int = 100,
                 lr: float = 1e-2, weight_decay: float = 1e-4, device: str = "cpu") -> torch.nn.Linear:
        """Full-batch logistic regression on cached features. Fast: features are small."""
        x = features.to(device)
        y = labels.to(device)
        mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True) + 1e-6
        head = torch.nn.Linear(x.shape[1], num_classes).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        xn = (x - mean) / std
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(head(xn), y)
            loss.backward()
            opt.step()
        # fold the normalisation into the layer so inference needs no stats
        with torch.no_grad():
            head.weight.div_(std)
            head.bias.sub_((head.weight * mean).sum(1))
        return head

    def save_head(self, path: str | Path) -> None:
        torch.save({"state": self.head.state_dict(), "labels": self.labels}, path)

    def load_head(self, path: str | Path) -> "LinearProbe":
        ckpt = torch.load(path, map_location=self.head.weight.device)
        self.head.load_state_dict(ckpt["state"])
        self.labels = ckpt.get("labels", self.labels)
        return self
