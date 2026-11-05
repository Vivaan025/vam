"""vam: a unified library and runtime for video-action models.

Design principle: PyTorch is the backend, not the competitor. vam owns the
abstractions above tensors (Video, FrameBatch, ActionOutput), the samplers that
decide which frames to process, the adapters that make existing models plug in,
and the runtime that makes long-horizon and streaming inference cheap.
"""

from vam.core import ActionOutput, FrameBatch, Video, load_video
from vam.errors import DecodeError, ScanError, VamError, VideoOpenError
from vam.models import get_model, list_models
from vam.runtime.cache import FeatureCache
from vam.runtime.profiler import Profiler, profile
from vam.sampling import KeyframeSampler, MotionSampler, Sampler, UniformSampler
from vam.stream import Stream

__version__ = "0.1.0"

__all__ = [
    "Video",
    "FrameBatch",
    "ActionOutput",
    "load_video",
    "VamError",
    "VideoOpenError",
    "DecodeError",
    "ScanError",
    "Sampler",
    "UniformSampler",
    "MotionSampler",
    "KeyframeSampler",
    "Stream",
    "Profiler",
    "profile",
    "FeatureCache",
    "get_model",
    "list_models",
]
