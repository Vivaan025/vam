from __future__ import annotations

import numpy as np

from vam.core import Video
from vam.sampling.base import Sampler


class UniformSampler(Sampler):
    """Evenly spaced frames across the clip. The baseline every other sampler must beat."""

    def select(self, video: Video) -> list[int]:
        n = video.num_frames
        if n <= self.num_frames:
            return list(range(n))
        centers = (np.arange(self.num_frames) + 0.5) * n / self.num_frames
        return sorted(set(int(c) for c in centers))
