import cv2
import numpy as np
import pytest

import vam
from vam.runtime.scan import scan_keyframes
from vam.sampling.keyframe import KeyframeSampler, OracleKeyframeScorer, uniform_scores


@pytest.fixture(scope="session")
def h264_video(tmp_path_factory):
    """30 s @ 10 fps: static for 20 s, then a moving square for 10 s. Written with PyAV so the
    GOP structure is real (keyframe every 2 s)."""
    import av

    path = str(tmp_path_factory.mktemp("kf") / "clip.mp4")
    fps, n, w, h = 10, 300, 160, 120
    with av.open(path, "w") as c:
        s = c.add_stream("mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.codec_context.gop_size = 20
        for i in range(n):
            f = np.zeros((h, w, 3), np.uint8)
            x = 20 if i < 200 else 20 + (i - 200)
            cv2.rectangle(f, (x, 40), (x + 30, 70), (0, 220, 60), -1)
            for pk in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    return path, fps, n


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
