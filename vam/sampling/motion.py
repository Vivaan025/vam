"""Motion-based sampling: spend the frame budget where the video changes most."""

from __future__ import annotations

import cv2
import numpy as np

from vam.core import Video
from vam.sampling.base import Sampler


class MotionSampler(Sampler):
    """Select frames at equal shares of cumulative inter-frame change.

    One sequential decode pass computes a cheap motion score per frame
    (mean absolute difference of downscaled grayscale frames). Frames are then
    chosen so that each covers an equal share of total motion, which allocates
    more frames to fast-changing intervals and fewer to static ones.

    stride: score every stride-th frame to bound the cost on long videos.
    thumb:  downscale size used for scoring.
    """

    def __init__(self, num_frames: int, stride: int = 1, thumb: tuple[int, int] = (64, 36)):
        super().__init__(num_frames)
        self.stride = max(1, stride)
        self.thumb = thumb

    def motion_scores(self, video: Video) -> np.ndarray:
        scores = np.zeros(video.num_frames, dtype=np.float32)
        prev = None
        for idx, rgb in video.iter_frames():
            if idx % self.stride:
                continue
            g = cv2.cvtColor(cv2.resize(rgb, self.thumb, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
            g = g.astype(np.float32)
            if prev is not None:
                scores[idx] = float(np.abs(g - prev).mean())
            prev = g
        return scores

    def select(self, video: Video) -> list[int]:
        n = video.num_frames
        if n <= self.num_frames:
            return list(range(n))
        scores = self.motion_scores(video)
        cum = np.cumsum(scores + 1e-6)  # epsilon keeps static videos from collapsing onto one frame
        cum /= cum[-1]
        targets = (np.arange(self.num_frames) + 0.5) / self.num_frames
        idx = np.searchsorted(cum, targets)
        return sorted(set(int(min(i, n - 1)) for i in idx))
