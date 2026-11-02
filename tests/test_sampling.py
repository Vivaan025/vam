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


def test_gap_window_targets_and_features():
    import numpy as np
    import torch

    from vam.runtime.scan import Keyframes
    from vam.sampling.learned import LearnedScorer, ScorerHead, build_features, gap_window_targets

    times = np.array([0.0, 5.0, 10.0, 12.0, 20.0])
    t = gap_window_targets(times, [(6.0, 11.0)], min_window_s=1.0, window_s=5.0)
    assert t.tolist() == [0.0, 0.8, 0.5, 0.0, 0.0]  # 5-10 covers 4 s of 5; 10-12 covers 1 of 2; others none
    emb = torch.arange(20, dtype=torch.float32).view(5, 4)
    gaps = torch.as_tensor(np.append(np.diff(times), 5.0))
    x = build_features(emb, gaps, None, context=1)
    assert x.shape == (5, 13)
    assert (x[0, :4] == 0).all() and (x[0, 4:8] == emb[0]).all() and (x[0, 8:12] == emb[1]).all()
    assert (x[-1, 8:12] == 0).all()

    class FakeEmbedder:
        device = "cpu"

        def embed(self, thumbs):
            return torch.ones(len(thumbs), 4)

        def scores_from_embedding(self, e):
            return np.full(len(e), 0.5)

    head = ScorerHead(14, 8, "mlp")
    ckpt = {"config": {"kind": "mlp", "context": 1, "use_zeroshot": True, "in_dim": 14, "hidden": 8},
            "folds": {"v1": 0}, "heads": [head.state_dict(), head.state_dict()]}
    sc = LearnedScorer(FakeEmbedder(), ckpt, device="cpu")
    kf = Keyframes(times, np.arange(5), np.zeros((5, 8, 8, 3), np.uint8), 1.0, 20.0, 20)
    assert sc(kf).shape == (5,)
    assert len(sc.for_video("v1")._heads()) == 1 and len(sc.for_video("new")._heads()) == 2
