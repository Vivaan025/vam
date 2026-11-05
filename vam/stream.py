"""Streaming input: process live or long video as a sequence of overlapping windows.

v0.1 is synchronous. The runtime research (async decode, overlapping transfer
with compute, micro-batching) builds on this interface without changing it.
"""

from __future__ import annotations

from typing import Iterator, Union

import cv2
import numpy as np
import torch

from vam.core import FrameBatch
from vam.errors import DecodeError, VideoOpenError


class Stream:
    """Yield FrameBatch windows from a file path or camera index.

    window: frames per batch handed to the model.
    hop:    frames to advance between windows (hop < window gives overlap).
    """

    def __init__(self, source: Union[str, int], window: int = 16, hop: int | None = None):
        self.source = source
        self.window = window
        self.hop = hop or window

    def __iter__(self) -> Iterator[FrameBatch]:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise VideoOpenError(f"could not open stream: {self.source}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        buf: list[np.ndarray] = []
        idxs: list[int] = []
        i = 0
        skip = 0  # frames to drop after a window when hop > window (014: a policy acting every k-th frame)
        try:
            while True:
                if skip > 0:
                    if not cap.grab():
                        break
                    i += 1
                    skip -= 1
                    continue
                ok, bgr = cap.read()
                if not ok:
                    break
                buf.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                idxs.append(i)
                i += 1
                if len(buf) == self.window:
                    yield self._batch(buf, idxs, fps)
                    buf, idxs = buf[self.hop:], idxs[self.hop:]
                    skip = max(0, self.hop - self.window)
            if buf:
                yield self._batch(buf, idxs, fps)
            elif i == 0:  # opened but nothing decoded, e.g. a codec this OpenCV build lacks (AV1 in 014)
                raise DecodeError(f"no frames decoded from {self.source}: codec unsupported or file corrupt")
        finally:
            cap.release()

    @staticmethod
    def _batch(buf, idxs, fps) -> FrameBatch:
        idx = torch.tensor(idxs, dtype=torch.long)
        return FrameBatch(torch.from_numpy(np.stack(buf)), idx.float() / fps, idx)
