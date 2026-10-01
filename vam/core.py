"""Core data types shared by every model and runtime component.

These are deliberately small. The goal is one vocabulary that works for both
meanings of "video action model":

* recognition: video -> label distribution over actions
* prediction:  video (+ state) -> action sequence for an agent

Both produce an ActionOutput; the difference is in which fields are populated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

import cv2
import numpy as np
import torch


@dataclass
class FrameBatch:
    """A batch of decoded frames with their timestamps.

    frames: uint8 tensor (T, H, W, C) in RGB, or float (T, C, H, W) after preprocessing.
    timestamps: float tensor (T,) seconds since video start.
    indices: long tensor (T,) original frame indices in the source video.
    """

    frames: torch.Tensor
    timestamps: torch.Tensor
    indices: torch.Tensor

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def to(self, device) -> "FrameBatch":
        return FrameBatch(self.frames.to(device), self.timestamps.to(device), self.indices.to(device))

    @property
    def is_channels_first(self) -> bool:
        return self.frames.ndim == 4 and self.frames.shape[1] in (1, 3) and self.frames.shape[-1] not in (1, 3)

    def channels_first(self) -> "FrameBatch":
        """Return (T, C, H, W) float32 in [0, 1]. No-op if already converted."""
        if self.is_channels_first:
            return self
        f = self.frames.permute(0, 3, 1, 2).float().div_(255.0)
        return FrameBatch(f, self.timestamps, self.indices)


@dataclass
class Video:
    """A video source. Decoding is lazy so long videos are never fully materialised.

    Use load_video() rather than constructing directly.
    """

    path: str
    fps: float
    num_frames: int
    width: int
    height: int
    _cap: Optional[cv2.VideoCapture] = field(default=None, repr=False, compare=False)

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps if self.fps else 0.0

    def _capture(self) -> cv2.VideoCapture:
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.path)
            if not self._cap.isOpened():
                raise IOError(f"could not open video: {self.path}")
        return self._cap

    def read(self, indices: Sequence[int]) -> FrameBatch:
        """Decode the given frame indices."""
        cap = self._capture()
        frames = []
        got = []
        for idx in sorted(set(int(i) for i in indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok:
                continue
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            got.append(idx)
        if not frames:
            raise ValueError("no frames could be decoded for the requested indices")
        arr = torch.from_numpy(np.stack(frames))
        idx_t = torch.tensor(got, dtype=torch.long)
        return FrameBatch(arr, idx_t.float() / float(self.fps), idx_t)

    def iter_frames(self, start: int = 0, stop: Optional[int] = None) -> Iterator[tuple[int, np.ndarray]]:
        """Sequentially decode frames (much faster than random access)."""
        cap = self._capture()
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        stop = self.num_frames if stop is None else min(stop, self.num_frames)
        while idx < stop:
            ok, bgr = cap.read()
            if not ok:
                break
            yield idx, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            idx += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def load_video(path: str) -> Video:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"could not open video: {path}")
    return Video(
        path=path,
        fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
        num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        _cap=cap,
    )


@dataclass
class ActionOutput:
    """Unified output for recognition and prediction models.

    logits: (num_classes,) for recognition.
    labels: class names aligned with logits, if known.
    actions: (T, action_dim) continuous or token actions for prediction models.
    features: optional per-frame features the runtime may cache for reuse.
    """

    logits: Optional[torch.Tensor] = None
    labels: Optional[list[str]] = None
    actions: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None

    def top_k(self, k: int = 5) -> list[tuple[str, float]]:
        if self.logits is None:
            raise ValueError("no logits in this output")
        probs = torch.softmax(self.logits.float().flatten(), dim=-1)
        vals, idx = probs.topk(min(k, probs.numel()))
        names = self.labels or [str(i) for i in range(probs.numel())]
        return [(names[int(i)], float(v)) for v, i in zip(vals, idx)]
