"""Samplers decide WHICH frames a model sees. This is the first research lever.

A sampler maps a Video and a frame budget to a list of frame indices. Keeping
this separate from decoding and from the model means every strategy (uniform,
motion-based, learned, adaptive) is benchmarked under identical conditions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vam.core import FrameBatch, Video


class Sampler(ABC):
    def __init__(self, num_frames: int):
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        self.num_frames = num_frames

    @abstractmethod
    def select(self, video: Video) -> list[int]:
        """Return sorted frame indices to decode and process."""

    def __call__(self, video: Video) -> FrameBatch:
        return video.read(self.select(video))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(num_frames={self.num_frames})"
