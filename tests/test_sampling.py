import vam


def test_uniform_covers_clip(synthetic_video):
    path, n, _ = synthetic_video
    v = vam.load_video(path)
    idx = vam.UniformSampler(8).select(v)
    assert len(idx) == 8
    assert idx == sorted(idx)
    assert idx[0] < n // 8 and idx[-1] > n - n // 8
    v.close()


def test_uniform_short_video_returns_all(synthetic_video):
    path, n, _ = synthetic_video
    v = vam.load_video(path)
    assert vam.UniformSampler(1000).select(v) == list(range(n))
    v.close()


def test_motion_sampler_favours_moving_half(synthetic_video):
    path, n, _ = synthetic_video
    v = vam.load_video(path)
    idx = vam.MotionSampler(8).select(v)
    assert len(idx) == 8
    moving = sum(1 for i in idx if i >= 30)
    assert moving >= 6, f"expected most frames in moving half, got {idx}"
    v.close()


def test_sampler_call_returns_batch(synthetic_video):
    path, _, _ = synthetic_video
    v = vam.load_video(path)
    b = vam.UniformSampler(4)(v)
    assert len(b) == 4
    v.close()
