"""Experiment 013: bad and degenerate inputs raise documented errors or give the uniform answer."""

from pathlib import Path

import numpy as np
import pytest

import vam
from vam.errors import DecodeError, ScanError, VamError, VideoOpenError
from vam.runtime.scan import scan_keyframes, shutdown_pools
from vam.runtime.scheduler import ACCURACY, Config, CostModel, Features, Scheduler, features_for
from vam.sampling import UniformSampler
from vam.sampling.keyframe import KeyframeSampler, OracleKeyframeScorer


def _clip(path: str, n: int, fps: int = 10, gop: int = 20, w: int = 96, h: int = 64) -> str:
    import av
    import cv2

    with av.open(path, "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.options = {"g": str(gop), "keyint_min": str(gop), "sc_threshold": "0", "preset": "ultrafast"}
        for i in range(n):
            f = np.zeros((h, w, 3), np.uint8)
            cv2.rectangle(f, ((i * 3) % (w - 10), 10), ((i * 3) % (w - 10) + 10, 30), (0, 200, 60), -1)
            for pk in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    return path


# ---- H1: documented errors, no hang -------------------------------------------------------

def test_missing_file_raises_video_open_error(tmp_path):
    with pytest.raises(VideoOpenError) as e:
        vam.load_video(str(tmp_path / "nope.mp4"))
    assert "nope.mp4" in str(e.value) and isinstance(e.value, VamError) and isinstance(e.value, IOError)


def test_empty_and_garbage_files_raise(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"not a video at all" * 300)
    for p in (empty, garbage):
        with pytest.raises(VideoOpenError):
            vam.load_video(str(p))
        with pytest.raises(VamError):
            scan_keyframes(str(p), thumb=(16, 12), workers=2, backend="thread")


def test_truncated_file_raises_or_degrades(tmp_path):
    good = _clip(str(tmp_path / "good.mp4"), 60)
    data = Path(good).read_bytes()
    trunc = tmp_path / "trunc.mp4"
    trunc.write_bytes(data[: len(data) // 4])  # the moov atom (faststart off) is at the end: gone
    with pytest.raises(VamError):
        v = vam.load_video(str(trunc))
        v.read([0, 1])
    with pytest.raises(VamError):
        scan_keyframes(str(trunc), thumb=(16, 12), workers=2, backend="thread")


def test_png_renamed_mp4_is_a_one_frame_video(tmp_path):
    """ffmpeg and OpenCV both read a still image as a one-frame stream; that is a degenerate video, not an
    error, and the sampler answers with its single frame."""
    import cv2

    p = str(tmp_path / "image.mp4")
    cv2.imwrite(str(tmp_path / "image.png"), np.zeros((32, 32, 3), np.uint8))
    Path(p).write_bytes((tmp_path / "image.png").read_bytes())
    kf = scan_keyframes(p, thumb=(16, 12), workers=2, backend="thread")
    assert len(kf) == 1
    v = vam.load_video(p)
    assert v.num_frames == 1  # OpenCV reports 0 frames for a still image; load_video counts by grabbing
    s = KeyframeSampler(4, _oracle(), refine=False)
    assert s.select(v) == [0] and s.last_fallback == "short-clip"
    v.close()


def test_process_scan_after_threads_does_not_hang(tmp_path):
    """013: a fork()ed worker created after thread pools exist can deadlock; the pool uses forkserver and
    the scan has a timeout, so this returns within seconds either way."""
    import multiprocessing as mp
    import time
    from concurrent.futures import ThreadPoolExecutor

    if "forkserver" not in mp.get_all_start_methods():
        pytest.skip("no forkserver on this platform")
    p = _clip(str(tmp_path / "pool.mp4"), 60, gop=10)
    with ThreadPoolExecutor(4) as ex:  # threads exist before the pool is created
        list(ex.map(lambda i: scan_keyframes(p, thumb=(16, 12), workers=2, backend="thread"), range(4)))
    t0 = time.time()
    kf = scan_keyframes(p, thumb=(16, 12), workers=4, backend="process")
    assert len(kf) >= 5 and time.time() - t0 < 60
    shutdown_pools()


def test_decode_error_for_impossible_indices(tmp_path):
    v = vam.load_video(_clip(str(tmp_path / "c.mp4"), 30))
    with pytest.raises(DecodeError):
        v.read([10_000, 20_000])
    v.close()


def test_shutdown_pools_is_idempotent():
    shutdown_pools()
    shutdown_pools()


# ---- H2: degenerate inputs give the uniform answer -------------------------------------

def _oracle(seg=((1.0, 2.0),), w=2.0):
    return OracleKeyframeScorer(list(seg), w)


def test_single_keyframe_clip_falls_back_to_uniform(tmp_path):
    p = _clip(str(tmp_path / "one.mp4"), 30, gop=1000)  # 3 s, one keyframe
    v = vam.load_video(p)
    s = KeyframeSampler(4, _oracle(), window_s=2.0, refine=False)
    idx = s.select(v)
    assert s.last_fallback == "single-keyframe"
    assert idx == UniformSampler(4).select(v)
    assert s.read(v, idx).frames.shape[0] == 4
    v.close()


def test_clip_shorter_than_budget_returns_every_frame(tmp_path):
    p = _clip(str(tmp_path / "tiny.mp4"), 6)
    v = vam.load_video(p)
    s = KeyframeSampler(16, _oracle(), refine=False)
    idx = s.select(v)
    assert s.last_fallback == "short-clip" and idx == list(range(v.num_frames))
    v.close()


def test_flat_scores_fall_back_to_uniform(tmp_path):
    p = _clip(str(tmp_path / "flat.mp4"), 120, gop=10)  # 12 s, 12 keyframes
    v = vam.load_video(p)
    s = KeyframeSampler(4, lambda kf: np.full(len(kf), 0.37), window_s=2.0, refine=False)
    idx = s.select(v)
    assert s.last_fallback == "flat-scores" and idx == UniformSampler(4).select(v)
    s2 = KeyframeSampler(4, lambda kf: np.full(len(kf), 0.37), window_s=2.0, refine=False, fallback="none")
    assert s2.select(v) and s2.last_fallback is None
    v.close()


def test_informative_scores_do_not_fall_back(tmp_path):
    p = _clip(str(tmp_path / "ok.mp4"), 120, gop=10)
    v = vam.load_video(p)
    s = KeyframeSampler(4, _oracle(seg=((8.0, 10.0),)), window_s=2.0, refine=False)
    idx = s.select(v)
    assert s.last_fallback is None and len(idx) == 4
    assert sum(i >= 60 for i in idx) >= 2  # the top-scored window (8-10 s) gets its share; equal alloc fills the other
    v.close()


def test_scorer_exception_propagates(tmp_path):
    p = _clip(str(tmp_path / "boom.mp4"), 60, gop=10)
    v = vam.load_video(p)

    def bad(kf):
        raise RuntimeError("scorer exploded")

    with pytest.raises(RuntimeError, match="exploded"):
        KeyframeSampler(4, bad, refine=False).select(v)
    v.close()


# ---- H3: scheduler uniform fallback ----------------------------------------------------

def _model():
    def ms(f):
        return 30 + 0.4 * f.K + 0.3 * f.Lmax + 0.25 * f.C + 0.5 * f.F + 10 * f.R
    rows = []
    for K in (10, 40, 90):
        for cfg in (Config(4, False), Config(8, True, 6), Config(16, True, 3), Config(16, False)):
            f = features_for(cfg, K, [150, 150, 120, 90])
            rows.append((f, ms(f)))
    return CostModel.fit(rows)


def test_scheduler_uses_uniform_only_when_scan_is_flat():
    sched = Scheduler(10_000, _model())
    p = sched.post_plan(30, [150] * 4, flat=False)
    assert not p.config.uniform and ACCURACY[p.config] == max(a for c, a in ACCURACY.items() if not c.uniform)
    p = sched.post_plan(30, [150] * 4, flat=True)
    assert p.config.uniform and p.config.frames == 16
    tight = Scheduler(1, _model()).post_plan(30, [150] * 4, flat=True)
    assert tight.config.uniform and tight.alternatives == 0


def test_planner_returns_uniform_for_flat_scores(tmp_path):
    p = _clip(str(tmp_path / "plan.mp4"), 120, gop=10)
    v = vam.load_video(p)
    sched = Scheduler(10_000, _model())
    s = KeyframeSampler(4, lambda kf: np.ones(len(kf)), window_s=2.0, planner=sched.planner())
    idx = s.select(v)
    assert s.last_fallback == "planner-uniform" and sched.last_plan.config.uniform
    assert idx == UniformSampler(sched.last_plan.config.frames).select(v)
    s = KeyframeSampler(4, _oracle(seg=((8.0, 10.0),)), window_s=2.0, planner=sched.planner())
    s.select(v)
    assert s.last_fallback is None and not sched.last_plan.config.uniform
    v.close()


def test_scan_is_flat_rule():
    assert Scheduler.scan_is_flat([0.5])
    assert Scheduler.scan_is_flat([0.5, 0.5, 0.5])
    assert not Scheduler.scan_is_flat([0.1, 0.9, 0.2])


def test_high_window_coverage_falls_back_to_uniform(tmp_path):
    """012/013: on a 6 s clip two 5 s windows cover the whole video; selection has nothing to choose."""
    p = _clip(str(tmp_path / "short.mp4"), 60, gop=10)  # 6 s, keyframes every 1 s
    v = vam.load_video(p)
    s = KeyframeSampler(4, _oracle(seg=((2.0, 3.0),)), window_s=5.0, min_window_s=5.0, window="fixed", refine=False)
    idx = s.select(v)
    assert s.last_fallback == "high-coverage" and idx == UniformSampler(4).select(v)
    s = KeyframeSampler(4, _oracle(seg=((2.0, 3.0),)), window_s=1.0, window="fixed", refine=False)
    s.select(v)
    assert s.last_fallback is None  # two 1 s windows on 6 s: 33 % coverage, selection is meaningful
    sched = Scheduler(10_000, _model())
    s = KeyframeSampler(4, _oracle(seg=((2.0, 3.0),)), window_s=5.0, min_window_s=5.0, window="fixed", planner=sched.planner())
    s.select(v)
    assert s.last_fallback == "planner-uniform" and sched.last_plan.config.uniform
    v.close()
