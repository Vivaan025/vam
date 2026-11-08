"""Experiment 019: coverage estimate from the scan, and the pre-scoring uniform route."""

import numpy as np
import pytest

import vam
from vam.runtime.scan import Keyframes, scan_keyframes
from vam.sampling import CoverageEstimator, KeyframeSampler, scan_features
from vam.sampling.coverage import FEATURE_NAMES


def _kf(k, seed=0, h=16, w=16):
    rng = np.random.RandomState(seed)
    thumbs = rng.randint(0, 255, (k, h, w, 3), dtype=np.uint8)
    times = np.linspace(0, 10, k)
    return Keyframes(times, (times * 10).astype(int), thumbs, 10.0, 10.0, 100)


def test_scan_features_fixed_length_and_finite():
    for k in (1, 2, 3, 40, 100):
        f = scan_features(_kf(k))
        assert f.shape == (len(FEATURE_NAMES),) and np.all(np.isfinite(f))
    # a static video (identical thumbnails) has zero change statistics and a full static run
    kf = _kf(20)
    kf.thumbs[:] = kf.thumbs[0]
    f = scan_features(kf)
    names = dict(zip(FEATURE_NAMES, f))
    assert names["diff_mean"] == 0 and names["pairwise_mean"] == 0 and names["longest_static_frac"] == 1.0


def test_estimator_fit_predict_roundtrip(tmp_path):
    rng = np.random.RandomState(1)
    X = rng.randn(200, len(FEATURE_NAMES))
    y = np.clip(0.5 + 0.3 * X[:, 0] - 0.2 * X[:, 3], 0, 1)
    est = CoverageEstimator.fit(X, y, alpha=0.1)
    pred = est.predict_features(X)
    assert pred.shape == (200,) and np.abs(pred - y).mean() < 0.05
    assert 0.0 <= est.predict(_kf(12)) <= 1.0
    est.save(tmp_path / "cov.npz")
    est2 = CoverageEstimator.load(tmp_path / "cov.npz")
    assert np.allclose(est2.predict_features(X), pred)


class _Const:
    def __init__(self, v):
        self.v = v

    def predict(self, kf):
        return self.v


def _scorer(kf):
    return np.linspace(0, 1, len(kf))


def test_keyframe_sampler_routes_before_scoring(h264_video):
    path, fps, n = h264_video
    calls = []

    def scorer(kf):
        calls.append(len(kf))
        return _scorer(kf)

    v = vam.load_video(path)
    try:
        s = KeyframeSampler(4, scorer, refine=None, coverage_estimator=_Const(0.9))
        idx = s.select(v)
        assert s.last_fallback == "predicted-coverage" and s.last_predicted_coverage == 0.9
        assert calls == [] and idx == s.uniform_indices(v)
        s = KeyframeSampler(4, scorer, refine=None, coverage_estimator=_Const(0.1))
        s.select(v)
        assert s.last_fallback is None and len(calls) == 1 and s.last_predicted_coverage == 0.1
        # fallback="none" ignores the estimator
        s = KeyframeSampler(4, scorer, refine=None, fallback="none", coverage_estimator=_Const(0.9))
        s.select(v)
        assert s.last_fallback is None and s.last_predicted_coverage is None
    finally:
        v.close()


def test_features_from_a_real_scan(h264_video):
    path, fps, n = h264_video
    kf = scan_keyframes(path, thumb=(32, 24))
    f = scan_features(kf)
    assert f.shape == (len(FEATURE_NAMES),) and np.all(np.isfinite(f))
    assert np.isclose(f[0], np.log1p(kf.duration))
