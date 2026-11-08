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

from vam.errors import DecodeError, VideoOpenError


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
                raise VideoOpenError(self.path)
        return self._cap

    def read(self, indices: Sequence[int], seq_threshold: int = 64) -> FrameBatch:
        """Decode the given frame indices.

        A seek (cap.set) lands on the previous keyframe and decodes forward, so it costs
        up to a whole GOP. When the next wanted index is within `seq_threshold` frames of
        the current position we read forward instead of seeking, which turns clustered
        requests (e.g. a window after a keyframe) into one cheap sequential run.
        """
        cap = self._capture()
        frames = []
        got = []
        pos = -1  # index of the frame cap.read() would return next, -1 = unknown
        for idx in sorted(set(int(i) for i in indices)):
            if pos < 0 or idx < pos or idx - pos > seq_threshold:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                pos = idx
            ok = True
            while pos < idx and ok:
                ok = cap.grab()
                pos += 1
            if not ok:
                continue
            ok, bgr = cap.read()
            pos += 1
            if not ok:
                continue
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            got.append(idx)
        if not frames:
            raise DecodeError(self.path, "none of the requested indices decoded")
        arr = torch.from_numpy(np.stack(frames))
        idx_t = torch.tensor(got, dtype=torch.long)
        return FrameBatch(arr, idx_t.float() / float(self.fps), idx_t)

    def iter_frames(self, start: int = 0, stop: Optional[int] = None, step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
        """Sequentially decode frames (much faster than random access).

        With `step > 1` every frame is still decoded (a P-frame needs its predecessors) but only
        every `step`-th one is converted to RGB and yielded; the rest cost a bare `grab()`.
        """
        cap = self._capture()
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        stop = self.num_frames if stop is None else min(stop, self.num_frames)
        while idx < stop:
            if (idx - start) % step:
                if not cap.grab():
                    break
                idx += 1
                continue
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
    import os

    if not os.path.exists(path):
        raise VideoOpenError(path, "no such file")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise VideoOpenError(path, "not a readable video")
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames <= 0:  # still image, or a container without an index: count by grabbing (013)
        num_frames = 0
        while cap.grab() and num_frames < 100_000:
            num_frames += 1
        cap.release()
        if num_frames == 0:
            raise VideoOpenError(path, "no decodable frames")
        cap = cv2.VideoCapture(path)
    return Video(
        path=path,
        fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
        num_frames=num_frames,
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
    text: the reply of a video-LLM (020), when the model answers in words.
    """

    logits: Optional[torch.Tensor] = None
    labels: Optional[list[str]] = None
    actions: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None
    text: Optional[str] = None

    def top_k(self, k: int = 5) -> list[tuple[str, float]]:
        if self.logits is None:
            raise ValueError("no logits in this output")
        probs = torch.softmax(self.logits.float().flatten(), dim=-1)
        vals, idx = probs.topk(min(k, probs.numel()))
        names = self.labels or [str(i) for i in range(probs.numel())]
        return [(names[int(i)], float(v)) for v, i in zip(vals, idx)]
