import cv2
import numpy as np
import pytest

import vam
from vam.runtime.scan import scan_keyframes
from vam.sampling.keyframe import KeyframeSampler, OracleKeyframeScorer, uniform_scores


def test_scan_finds_keyframes_in_order(h264_video):
    path, fps, n = h264_video
    kf = scan_keyframes(path, thumb=(32, 24))
    assert 5 <= len(kf) <= 40
    assert np.all(np.diff(kf.times) > 0)
    assert kf.thumbs.shape[1:] == (24, 32, 3)
    assert abs(kf.fps - fps) < 0.5
    assert kf.indices.max() < n


def test_oracle_scorer_prefers_action_window(h264_video):
    path, fps, n = h264_video
    kf = scan_keyframes(path, thumb=(32, 24))
    s = OracleKeyframeScorer([(20.0, 30.0)], window_s=2.0)(kf)
    best = kf.times[int(np.argmax(s))]
    assert 18.0 <= best <= 30.0
    assert s[kf.times < 17.5].max() == 0.0


def test_keyframe_sampler_lands_in_action(h264_video):
    path, fps, n = h264_video
    v = vam.load_video(path)
    sampler = KeyframeSampler(8, OracleKeyframeScorer([(20.0, 30.0)], 2.0), window_s=2.0, segments=2)
    idx = sampler.select(v)
    assert len(idx) == 8
    assert all(i >= 180 for i in idx), idx  # all within the moving part (allowing keyframe lead-in)
    b = v.read(idx)
    assert len(b) == 8
    v.close()


def test_uniform_scores_fall_back_to_nms(h264_video):
    path, fps, n = h264_video
    v = vam.load_video(path)
    idx = KeyframeSampler(4, uniform_scores, window_s=2.0, segments=2).select(v)
    assert len(idx) == 4 and idx == sorted(idx)
    v.close()


def test_sequential_read_matches_seek_read(h264_video):
    path, fps, n = h264_video
    v = vam.load_video(path)
    idx = [100, 101, 103, 110, 250, 251]
    a = v.read(idx, seq_threshold=64)
    b = v.read(idx, seq_threshold=0)
    assert a.indices.tolist() == b.indices.tolist() == idx
    assert (a.frames == b.frames).all()
    v.close()


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_parallel_scan_matches_single(h264_video, backend):
    import multiprocessing as mp

    if backend == "process" and "fork" not in mp.get_all_start_methods():
        pytest.skip("process backend needs fork (workers would re-import torch/cv2 under spawn)")
    path, fps, n = h264_video
    a = scan_keyframes(path, thumb=(32, 24), workers=1, backend="serial")
    b = scan_keyframes(path, thumb=(32, 24), workers=4, backend=backend)
    assert np.allclose(a.times, b.times)
    assert (a.thumbs == b.thumbs).all()


def test_keyframe_stride_halves(h264_video):
    path, fps, n = h264_video
    a = scan_keyframes(path, thumb=(32, 24), workers=1)
    b = scan_keyframes(path, thumb=(32, 24), workers=1, stride=2)
    assert len(b) == (len(a) + 1) // 2
    assert np.allclose(b.times, a.times[::2])


def _prefer_time(t_star):
    """Scorer that peaks at the keyframe nearest t_star (to steer stage 2 in tests)."""
    def f(kf):
        return -np.abs(kf.times - t_star)
    return f


def test_gap_window_stops_at_next_keyframe(h264_video):
    path, fps, n = h264_video
    v = vam.load_video(path)
    s = KeyframeSampler(4, uniform_scores, window_s=5.0, segments=2, window="gap", min_window_s=0.5)
    s.select(v)
    kf = s.last_keyframes
    for i, (start, end) in zip(s.last_chosen, s.last_windows):
        if i + 1 < len(kf):
            assert end <= int(kf.indices[i + 1]) + 1, (start, end, kf.indices[i + 1])
        assert end - start >= 0.5 * fps - 1
    v.close()


def test_softmax_allocation_sums_to_budget(h264_video):
    path, fps, n = h264_video
    v = vam.load_video(path)
    for budget in (4, 8, 16):
        s = KeyframeSampler(budget, OracleKeyframeScorer([(20.0, 30.0)], 2.0), window_s=2.0, segments=4,
                            alloc="softmax", temperature=0.1)
        idx = s.select(v)
        per = s.allocate(s.last_chosen, s.last_scores)
        assert sum(per.values()) == budget
        assert len(idx) == budget
        # the best keyframe (inside the action) gets at least as many frames as any other
        best = max(s.last_chosen, key=lambda i: s.last_scores[i])
        assert per[best] == max(per.values())
    v.close()


def test_refinement_moves_frames_into_action(h264_video):
    path, fps, n = h264_video
    v = vam.load_video(path)
    # stage 2 deliberately picks the keyframe at ~18 s; the action starts at 20 s inside its 4 s window
    fine = OracleKeyframeScorer([(20.0, 30.0)], 0.1)
    base = KeyframeSampler(4, _prefer_time(18.0), window_s=4.0, segments=1, window="fixed")
    ref = KeyframeSampler(4, _prefer_time(18.0), window_s=4.0, segments=1, window="fixed", refine=fine, refine_stride=2)
    a, b = base.select(v), ref.select(v)
    assert abs(v.read(a).indices[0].item() - 180) <= 20
    assert any(i < 200 for i in a), a          # even placement wastes frames before the action
    assert all(i >= 200 for i in b), b         # refinement puts all four inside it
    assert len(b) == 4 and ref.last_refine_ms > 0
    # frames handed over from the refinement cache equal a fresh decode
    cached, fresh = ref.read(v, b), v.read(b)
    assert cached.indices.tolist() == fresh.indices.tolist()
    assert (cached.frames == fresh.frames).all()
    v.close()


def test_iter_frames_step_matches_read(h264_video):
    path, fps, n = h264_video
    v = vam.load_video(path)
    got = list(v.iter_frames(100, 112, step=3))
    assert [i for i, _ in got] == [100, 103, 106, 109]
    ref = v.read([100, 103, 106, 109])
    assert all((f == ref.frames[k].numpy()).all() for k, (_, f) in enumerate(got))
    v.close()


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_read_windows_matches_video_read(h264_video, backend):
    import multiprocessing as mp

    from vam.runtime.decode import Window, read_windows

    if backend == "process" and "fork" not in mp.get_all_start_methods():
        pytest.skip("process backend needs fork")
    path, fps, n = h264_video
    v = vam.load_video(path)
    wins = [Window(40, 80, [41, 55, 79]), Window(200, 240, None, 7), Window(280, 300, [280, 299])]
    parts = read_windows(path, wins, thumb=(16, 12), keep_edge=0, workers=3, backend=backend)
    assert parts[0].indices.tolist() == [41, 55, 79]
    assert parts[1].indices.tolist() == list(range(200, 240, 7))
    assert parts[2].indices.tolist() == [280, 299]
    assert parts[1].thumbs.shape == (len(parts[1].indices), 12, 16, 3)
    for wf in parts:
        ref = v.read(wf.indices.tolist())
        assert (wf.frames == ref.frames.numpy()).all()
    v.close()


def test_read_windows_keep_edge_downscales(h264_video):
    from vam.runtime.decode import Window, read_windows

    path, fps, n = h264_video
    (wf,) = read_windows(path, [Window(10, 20, [10, 15])], keep_edge=60, backend="serial")
    assert wf.frames.shape[1:] == (60, 80, 3)  # 120x160 -> shortest edge 60


def test_parallel_read_equals_serial_read(h264_video):
    path, fps, n = h264_video
    v = vam.load_video(path)
    scorer = OracleKeyframeScorer([(20.0, 30.0)], 2.0)
    a = KeyframeSampler(8, scorer, window_s=2.0, segments=2, decode_backend="serial")
    b = KeyframeSampler(8, scorer, window_s=2.0, segments=2, decode_backend="thread")
    ia, ib = a.select(v), b.select(v)
    assert ia == ib
    fa, fb = a.read(v, ia), b.read(v, ib)
    assert fa.indices.tolist() == fb.indices.tolist() == ia
    assert (fa.frames == fb.frames).all()
    v.close()
