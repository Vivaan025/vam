"""vam: a unified library and runtime for video-action models.

Design principle: PyTorch is the backend, not the competitor. vam owns the
abstractions above tensors (Video, FrameBatch, ActionOutput), the samplers that
decide which frames to process, the adapters that make existing models plug in,
and the runtime that makes long-horizon and streaming inference cheap.
"""

from vam.core import ActionOutput, FrameBatch, Video, load_video
from vam.sampling import MotionSampler, Sampler, UniformSampler
from vam.stream import Stream
from vam.runtime.profiler import Profiler, profile
from vam.runtime.cache import FeatureCache
from vam.models import get_model, list_models

__version__ = "0.1.0"

__all__ = [
    "Video",
    "FrameBatch",
    "ActionOutput",
    "load_video",
    "Sampler",
    "UniformSampler",
    "MotionSampler",
    "Stream",
    "Profiler",
    "profile",
    "FeatureCache",
    "get_model",
    "list_models",
]
