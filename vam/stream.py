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
            raise IOError(f"could not open stream: {self.source}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        buf: list[np.ndarray] = []
        idxs: list[int] = []
        i = 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                buf.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                idxs.append(i)
                i += 1
                if len(buf) == self.window:
                    yield self._batch(buf, idxs, fps)
                    buf, idxs = buf[self.hop:], idxs[self.hop:]
            if buf:
                yield self._batch(buf, idxs, fps)
        finally:
            cap.release()

    @staticmethod
    def _batch(buf, idxs, fps) -> FrameBatch:
        idx = torch.tensor(idxs, dtype=torch.long)
        return FrameBatch(torch.from_numpy(np.stack(buf)), idx.float() / fps, idx)
