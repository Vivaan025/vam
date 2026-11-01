"""Tensor preprocessing on the device, replacing per-image PIL loops.

Experiment 004: Hugging Face image processors resize/normalise each frame with PIL on the
CPU, which cost 250-380 ms per video for VideoMAE (16 frames) and ~170 ms for CLIP (~57
thumbnails) against a ~7 ms GPU forward. Decoded frames are already uint8 arrays, so the
same recipe (resize shortest edge -> centre crop -> rescale -> normalise) is a few batched
tensor ops. Parameters are read from the HF processor so each model keeps its exact numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

_RESAMPLE = {0: "nearest", 2: "bilinear", 3: "bicubic", "nearest": "nearest", "bilinear": "bilinear",
             "bicubic": "bicubic"}


@dataclass
class Preprocessor:
    shortest_edge: int
    crop: tuple[int, int]            # (height, width)
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    mode: str = "bilinear"           # interpolation
    rescale: float = 1.0 / 255.0
    device: str = "cpu"

    @classmethod
    def from_hf(cls, processor, device: str = "cpu") -> "Preprocessor":
        d = processor.to_dict() if hasattr(processor, "to_dict") else dict(vars(processor))
        size = d.get("size") or {}
        if "shortest_edge" in size:
            se = int(size["shortest_edge"])
        else:  # fixed height/width: treat the smaller as the shortest edge
            se = int(min(size.get("height", 224), size.get("width", 224)))
        cs = d.get("crop_size") or {"height": se, "width": se}
        crop = (int(cs["height"]), int(cs["width"])) if d.get("do_center_crop", True) else (se, se)
        rs = d.get("resample", 2)
        mode = _RESAMPLE.get(rs if not hasattr(rs, "value") else rs.value, "bilinear")
        return cls(
            shortest_edge=se,
            crop=crop,
            mean=tuple(float(x) for x in d.get("image_mean", (0.485, 0.456, 0.406))),
            std=tuple(float(x) for x in d.get("image_std", (0.229, 0.224, 0.225))),
            mode=mode,
            rescale=float(d.get("rescale_factor", 1.0 / 255.0)),
            device=device,
        )

    def __call__(self, frames) -> torch.Tensor:
        """uint8 (T,H,W,3) numpy or tensor -> float32 (T,3,crop_h,crop_w) on the device."""
        x = torch.as_tensor(np.asarray(frames)) if not torch.is_tensor(frames) else frames
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float()  # (T,3,H,W)
        t, c, h, w = x.shape
        scale = self.shortest_edge / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        if (nh, nw) != (h, w):
            x = F.interpolate(x, size=(nh, nw), mode=self.mode, align_corners=False, antialias=True)
        ch, cw = self.crop
        top, left = (nh - ch) // 2, (nw - cw) // 2
        x = x[:, :, top:top + ch, left:left + cw]
        mean = torch.tensor(self.mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=x.device).view(1, 3, 1, 1)
        return (x * self.rescale - mean) / std
