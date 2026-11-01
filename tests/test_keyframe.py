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
