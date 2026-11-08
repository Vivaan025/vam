"""Experiment 019: coverage estimate from the scan, and the pre-scoring uniform route."""

import numpy as np
import pytest

import vam
from vam.runtime.scan import Keyframes, scan_keyframes
from vam.sampling import CoverageEstimator, KeyframeSampler, scan_features, score_features
from vam.sampling.coverage import FEATURE_NAMES, SCORE_FEATURE_NAMES


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


def test_score_features_and_scores_estimator_runs_after_scoring(h264_video, tmp_path):
    for sc in (np.zeros(0), np.zeros(1), np.linspace(0, 1, 30), np.array([5.0, -1.0, 0.2])):
        f = score_features(sc, np.arange(len(sc)) * 2.0, 60.0)
        assert f.shape == (len(SCORE_FEATURE_NAMES),) and np.all(np.isfinite(f))
    peaked = score_features(np.r_[np.zeros(29), 3.0], np.arange(30) * 2.0, 60.0)
    flat = score_features(np.zeros(30), np.arange(30) * 2.0, 60.0)
    names = dict(zip(SCORE_FEATURE_NAMES, peaked))
    assert names["entropy_softmax"] < dict(zip(SCORE_FEATURE_NAMES, flat))["entropy_softmax"]
    rng = np.random.RandomState(0)
    X = rng.randn(100, len(SCORE_FEATURE_NAMES))
    est = CoverageEstimator.fit(X, np.full(100, 0.9), kind="scores")
    est.save(tmp_path / "s.npz")
    est = CoverageEstimator.load(tmp_path / "s.npz")
    assert est.kind == "scores"
    path, fps, n = h264_video
    calls = []

    def scorer(kf):
        calls.append(len(kf))
        return np.linspace(0, 1, len(kf))

    v = vam.load_video(path)
    try:
        s = KeyframeSampler(4, scorer, refine=None, coverage_estimator=est)
        idx = s.select(v)
        assert len(calls) == 1  # scored first, then routed
        assert s.last_fallback == "predicted-coverage" and s.last_predicted_coverage is not None and s.last_scores is not None
        assert idx == s.uniform_indices(v)
    finally:
        v.close()
