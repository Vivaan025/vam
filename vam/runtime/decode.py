"""Windowed decode: read short frame ranges after chosen keyframes, one worker per window.

Experiment 006 measured the CPU decode path on 320x180 H.264: OpenCV charges a flat 35-60 ms per
seek, whatever the target, plus ~0.3 ms per decoded frame (PyAV is slower still, ~0.9 ms per frame
of per-packet overhead). A 5 s window is therefore ~95 ms, and a sampler that needs four of them
spends ~400 ms serially. The windows are independent, so they are decoded in parallel on the same
persistent process pool the keyframe scan uses (experiment 007).

Each window job returns the frames it was asked to keep (full resolution, optionally downscaled so
the shortest edge is `keep_edge`) and, for refinement, a thumbnail of every `step`-th frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from vam.runtime.scan import _process_pool, default_scan_backend, default_scan_workers


@dataclass
class Window:
    start: int                    # first frame index (a keyframe)
    end: int                      # exclusive
    want: Optional[list[int]] = None   # frames to keep at full res; None -> every `step`-th frame
    step: int = 1                 # thumbnail / candidate stride when want is None


@dataclass
class WindowFrames:
    indices: np.ndarray           # (M,) frame indices kept
    frames: np.ndarray            # (M, H, W, 3) uint8 RGB, shortest edge <= keep_edge
    thumbs: Optional[np.ndarray]  # (M, th, tw, 3) uint8 RGB or None


_LOCAL = None


def _capture(path: str):
    """One cached VideoCapture per worker (per process, and per thread when the pool is threads;
    a VideoCapture is not safe to share between threads). Opening costs a few ms."""
    global _LOCAL
    import threading

    import cv2

    if _LOCAL is None:
        _LOCAL = threading.local()
    cap = getattr(_LOCAL, "cap", None)
    if cap is None or getattr(_LOCAL, "path", None) != path or not cap.isOpened():
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            from vam.errors import VideoOpenError

            raise VideoOpenError(path)
        _LOCAL.cap, _LOCAL.path = cap, path
    return cap


def _downscale(frame: np.ndarray, keep_edge: int, cv2) -> np.ndarray:
    h, w = frame.shape[:2]
    e = min(h, w)
    if keep_edge and e > keep_edge:
        r = keep_edge / e
        frame = cv2.resize(frame, (max(1, round(w * r)), max(1, round(h * r))), interpolation=cv2.INTER_AREA)
    return frame


def read_window(path: str, win: Window, thumb: Optional[tuple[int, int]], keep_edge: int) -> WindowFrames:
    """Seek once to `win.start` and read forward. Runs inside a worker."""
    import cv2

    cap = _capture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, win.start)
    want = None if win.want is None else set(int(i) for i in win.want)
    last = win.end if want is None else min(win.end, max(want) + 1)
    idx = win.start
    kept_idx: list[int] = []
    kept: list[np.ndarray] = []
    thumbs: list[np.ndarray] = []
    while idx < last:
        take = (idx in want) if want is not None else ((idx - win.start) % win.step == 0)
        if not take:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if thumb is not None:
            thumbs.append(cv2.resize(rgb, thumb, interpolation=cv2.INTER_AREA))
        kept.append(_downscale(rgb, keep_edge, cv2))
        kept_idx.append(idx)
        idx += 1
    if not kept:
        return WindowFrames(np.zeros(0, np.int64), np.zeros((0, 1, 1, 3), np.uint8), None)
    return WindowFrames(np.asarray(kept_idx, np.int64), np.stack(kept), np.stack(thumbs) if thumb is not None else None)


def _job(args):
    return read_window(*args)


def read_windows(path: str, windows: Sequence[Window], thumb: Optional[tuple[int, int]] = None,
                 keep_edge: int = 224, workers: int | None = None, backend: str | None = None) -> list[WindowFrames]:
    """Decode several windows of one file, in parallel. Results are in the order of `windows`.

    backend: "process" (default where fork exists; shares the scan's pool), "thread" (OpenCV releases
             the GIL while decoding, so threads also scale here, unlike the PyAV scan), or "serial".
    """
    jobs = [(path, w, thumb, keep_edge) for w in windows]
    if not jobs:
        return []
    workers = min(len(jobs), default_scan_workers() if workers is None else max(1, int(workers)))
    backend = backend or default_scan_backend()
    if workers == 1 or backend == "serial":
        return [_job(j) for j in jobs]
    if backend == "process":
        return list(_process_pool(workers).map(_job, jobs))
    if backend == "thread":
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(_job, jobs))
    raise ValueError(f"unknown backend {backend!r}")
