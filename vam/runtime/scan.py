"""Keyframe scan: the cheap first look at a long video.

Compressed video stores only keyframes (I-frames) whole; every other frame is a
delta. Telling the decoder to skip non-key frames means no inter-frame work at all,
so a 3-minute H.264 file yields its ~60 keyframes in ~100 ms, a fifth of what one
random-access read of 16 frames costs (each such read seeks to a keyframe and decodes
forward). Measured on THUMOS14: 96 ms/video for the scan vs ~500 ms random access.

Keyframes land every ~5 s, and annotated actions are ~3 s long, so a keyframe is
*inside* the action only ~36% of the time but *within 4 s* of it 98% of the time.
The scan therefore locates regions; a second stage decodes a short window forward
from the chosen keyframes (vam.sampling.keyframe).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Keyframes:
    times: np.ndarray       # (K,) seconds
    indices: np.ndarray     # (K,) frame index in the stream
    thumbs: np.ndarray      # (K, H, W, 3) uint8 RGB
    fps: float
    duration: float
    num_frames: int

    def __len__(self) -> int:
        return int(self.times.shape[0])


def scan_keyframes(path: str, thumb: tuple[int, int] = (112, 112), max_keyframes: int | None = None) -> Keyframes:
    """Decode only keyframes and return low-res thumbnails with their timestamps.

    thumb: (width, height) of the thumbnails. Conversion dominates scan time, so keep it small.
    """
    import av  # optional dependency, imported lazily

    with av.open(path) as c:
        s = c.streams.video[0]
        s.codec_context.skip_frame = "NONKEY"
        s.thread_type = "AUTO"
        tb = float(s.time_base)
        fps = float(s.average_rate) if s.average_rate else 30.0
        duration = float(s.duration * tb) if s.duration else 0.0
        num_frames = int(s.frames) if s.frames else int(round(duration * fps))
        times, thumbs = [], []
        for f in c.decode(s):
            if f.pts is None:
                continue
            times.append(f.pts * tb)
            thumbs.append(f.to_ndarray(format="rgb24", width=thumb[0], height=thumb[1]))
            if max_keyframes and len(times) >= max_keyframes:
                break
    if not times:
        raise ValueError(f"no keyframes decoded from {path}")
    t = np.asarray(times, dtype=np.float64)
    if duration <= 0:
        duration = float(t[-1])
    idx = np.clip(np.round(t * fps).astype(np.int64), 0, max(num_frames - 1, 0))
    return Keyframes(t, idx, np.stack(thumbs), fps, duration, num_frames)
