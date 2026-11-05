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


def _scan_range(path: str, t0: float, t1: float, thumb: tuple[int, int], stride: int) -> list[tuple[float, np.ndarray]]:
    """Decode keyframes with t0 <= t < t1 (seconds). One container per worker, single codec thread."""
    import av

    out: list[tuple[float, np.ndarray]] = []
    with av.open(path) as c:
        s = c.streams.video[0]
        s.codec_context.skip_frame = "NONKEY"
        s.codec_context.thread_count = 1  # several workers; ffmpeg's own pool would thrash
        tb = float(s.time_base)
        if t0 > 0:
            c.seek(int(t0 / tb), stream=s, backward=True, any_frame=False)
        k = 0
        done = False
        # Feed the decoder keyframe packets only. skip_frame="NONKEY" alone is not enough: libvpx (VP9) and
        # some HEVC builds ignore it and decode every frame (011: 34 s per video on VP9 instead of ~0.4 s).
        # The demuxer flags keyframes from the container index, so non-key packets are dropped unread.
        for pkt in c.demux(s):
            if pkt.dts is not None and not pkt.is_keyframe:
                continue
            for f in pkt.decode():
                if f.pts is None:
                    continue
                t = f.pts * tb
                if t < t0:
                    continue
                if t >= t1:
                    done = True
                    break
                if k % stride == 0:
                    out.append((t, f.to_ndarray(format="rgb24", width=thumb[0], height=thumb[1])))
                k += 1
            if done:
                break
    return out


_POOL = None
_POOL_SIZE = 0


def _process_pool(workers: int):
    """A persistent process pool (fork on Linux). Created lazily; grown if more workers are asked for."""
    global _POOL, _POOL_SIZE
    if _POOL is None or _POOL_SIZE < workers:
        import atexit
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        if _POOL is not None:
            _POOL.shutdown(wait=False)
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
        _POOL = ProcessPoolExecutor(workers, mp_context=ctx)
        _POOL_SIZE = workers
        atexit.register(_POOL.shutdown, wait=False)
    return _POOL


def shutdown_pools() -> None:
    """Stop the scan's worker processes (also registered with atexit). Safe to call twice."""
    global _POOL, _POOL_SIZE
    if _POOL is not None:
        _POOL.shutdown(wait=True)
        _POOL, _POOL_SIZE = None, 0


def default_scan_workers() -> int:
    import os

    return max(1, min(8, (os.cpu_count() or 2) - 1))


def default_scan_backend() -> str:
    """Processes where they can fork (Linux/macOS): workers inherit the loaded modules for free.
    Threads elsewhere: spawned workers would each re-import torch/cv2, which costs seconds and memory."""
    import multiprocessing as mp

    return "process" if "fork" in mp.get_all_start_methods() else "thread"


def scan_keyframes(path: str, thumb: tuple[int, int] = (112, 112), max_keyframes: int | None = None,
                   workers: int | None = None, stride: int = 1, backend: str | None = None) -> Keyframes:
    """Decode only keyframes and return low-res thumbnails with their timestamps.

    thumb:   (width, height) of the thumbnails.
    workers: the timeline is split into this many ranges scanned in parallel (default: cpu_count-1,
             at most 8). Software keyframe decode costs ~1.5 ms each and the per-frame Python work holds
             the GIL, so threads stop scaling at ~4 workers while processes scale to 8 (experiment 005:
             242 -> 44 ms on a 98-keyframe video; 6x throughput across videos).
    stride:  keep every stride-th keyframe (within each worker's range). Cheaper scan, coarser scan.
    backend: "process" (default where fork is available), "thread" (default elsewhere), or "serial".
    """
    import av

    from vam.errors import ScanError, VideoOpenError

    try:
        with av.open(path) as c:
            if not c.streams.video:
                raise VideoOpenError(path, "no video stream")
            s = c.streams.video[0]
            tb = float(s.time_base)
            fps = float(s.average_rate) if s.average_rate else 30.0
            duration = float(s.duration * tb) if s.duration else 0.0
            num_frames = int(s.frames) if s.frames else int(round(duration * fps))
    except VideoOpenError:
        raise
    except Exception as e:  # av.error.* (FileNotFoundError, InvalidDataError, ...)
        raise VideoOpenError(path, type(e).__name__) from e

    workers = default_scan_workers() if workers is None else max(1, int(workers))
    backend = backend or default_scan_backend()
    if duration <= 0 or workers == 1 or backend == "serial":
        parts = [_scan_range(path, 0.0, float("inf"), thumb, stride)]
    else:
        edges = np.linspace(0.0, duration + 1.0, workers + 1)
        jobs = [(path, float(edges[i]), float(edges[i + 1]), thumb, stride) for i in range(workers)]
        try:
            if backend == "process":
                parts = list(_process_pool(workers).map(_scan_range_job, jobs))
            elif backend == "thread":
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(workers) as ex:
                    parts = list(ex.map(_scan_range_job, jobs))
            else:
                raise ValueError(f"unknown backend {backend!r}")
        except ValueError:
            raise
        except Exception as e:  # a worker hit a corrupt packet
            raise ScanError(path, type(e).__name__) from e
    items = sorted((x for part in parts for x in part), key=lambda x: x[0])
    if max_keyframes:
        items = items[:max_keyframes]
    if not items:
        raise ScanError(path, "no keyframes decoded")
    t = np.asarray([x[0] for x in items], dtype=np.float64)
    thumbs = np.stack([x[1] for x in items])
    if duration <= 0:
        duration = float(t[-1])
    idx = np.clip(np.round(t * fps).astype(np.int64), 0, max(num_frames - 1, 0))
    return Keyframes(t, idx, thumbs, fps, duration, num_frames)


def _scan_range_job(job):
    return _scan_range(*job)
