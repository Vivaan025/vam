"""The one interface every model in vam implements.

Models never decode video or choose frames. They receive a FrameBatch and
return an ActionOutput. This is what lets the runtime swap samplers, cache
features and stream without any model knowing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import torch

from vam.core import ActionOutput, FrameBatch, Video
from vam.sampling import Sampler, UniformSampler


class VideoActionModel(torch.nn.Module, ABC):
    """Base class. Subclasses set default_num_frames and implement forward_frames."""

    default_num_frames: int = 16

    @abstractmethod
    def forward_frames(self, batch: FrameBatch) -> ActionOutput:
        """Run the model on already selected frames."""

    def forward(self, batch: FrameBatch) -> ActionOutput:
        return self.forward_frames(batch)

    @torch.inference_mode()
    def __call__(self, video: Union[Video, FrameBatch], sampler: Optional[Sampler] = None) -> ActionOutput:
        if isinstance(video, Video):
            sampler = sampler or UniformSampler(self.default_num_frames)
            video = sampler(video)
        return self.forward_frames(video)
