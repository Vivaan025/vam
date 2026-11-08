"""Action-coverage estimate from the keyframe scan alone (experiment 019, roadmap R10).

Frame selection pays when the annotated action covers a small part of the video (THUMOS14) and does nothing
when it fills the clip (Charades, ActivityNet). The estimate here lets `KeyframeSampler` route a video to
uniform frames *before* the CLIP scorer runs, from statistics of the scan it already has: duration, keyframe
rate, and how much the keyframe thumbnails change along the video.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

FEATURE_NAMES = [
    "log_duration", "log_kf_per_s", "log_num_kf",
    "diff_mean", "diff_std", "diff_max", "diff_p90", "cut_rate",
    "dist_mean_mean", "dist_mean_std", "pairwise_mean", "brightness_std", "longest_static_frac",
]
DEFAULT_WEIGHTS = "coverage_ridge.npz"  # vam/weights/, written by scripts/train_coverage.py --release


def scan_features(kf) -> np.ndarray:
    """Fixed-length vector from a `Keyframes` (times, thumbs, duration). All thumbnail statistics are L1
    distances of uint8 RGB thumbnails scaled to [0, 1]; the vector is NaN-free for any K >= 1."""
    k = int(len(kf))
    duration = max(float(kf.duration), 1e-3)
    t = np.asarray(kf.thumbs, dtype=np.float32) / 255.0  # (K,H,W,3)
    flat = t.reshape(k, -1)
    if k >= 2:
        diffs = np.abs(flat[1:] - flat[:-1]).mean(axis=1)
        med = float(np.median(diffs))
        cut_rate = float(np.mean(diffs > 2.0 * max(med, 1e-6)))
        runs, cur = [], 1
        for d in diffs:
            if d < 0.5 * max(med, 1e-6):
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)
        longest_static = max(runs) / k
    else:
        diffs = np.zeros(1, dtype=np.float32)
        cut_rate, longest_static = 0.0, 1.0
    mean_thumb = flat.mean(axis=0, keepdims=True)
    dist_mean = np.abs(flat - mean_thumb).mean(axis=1)
    sub = flat if k <= 32 else flat[np.linspace(0, k - 1, 32).astype(int)]
    if len(sub) >= 2:
        pw = np.abs(sub[:, None, :] - sub[None, :, :]).mean(axis=2)
        pairwise = float(pw[np.triu_indices(len(sub), 1)].mean())
    else:
        pairwise = 0.0
    brightness = flat.mean(axis=1)
    return np.array([
        np.log1p(duration), np.log1p(k / duration), np.log1p(k),
        float(diffs.mean()), float(diffs.std()), float(diffs.max()), float(np.percentile(diffs, 90)), cut_rate,
        float(dist_mean.mean()), float(dist_mean.std()), pairwise, float(brightness.std()), float(longest_static),
    ], dtype=np.float64)


class CoverageEstimator:
    """Ridge regression on standardised scan features -> predicted action coverage in [0, 1]."""

    def __init__(self, weights: np.ndarray, bias: float, mean: np.ndarray, std: np.ndarray):
        self.w = np.asarray(weights, dtype=np.float64)
        self.b = float(bias)
        self.mu = np.asarray(mean, dtype=np.float64)
        self.sigma = np.where(np.asarray(std, dtype=np.float64) > 0, std, 1.0)

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> "CoverageEstimator":
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        mu, sigma = X.mean(axis=0), X.std(axis=0)
        sigma = np.where(sigma > 0, sigma, 1.0)
        Z = (X - mu) / sigma
        A = Z.T @ Z + alpha * np.eye(Z.shape[1])
        w = np.linalg.solve(A, Z.T @ (y - y.mean()))
        return cls(w, float(y.mean()), mu, sigma)

    def predict_features(self, X: np.ndarray) -> np.ndarray:
        Z = (np.atleast_2d(np.asarray(X, dtype=np.float64)) - self.mu) / self.sigma
        return np.clip(Z @ self.w + self.b, 0.0, 1.0)

    def predict(self, kf) -> float:
        return float(self.predict_features(scan_features(kf))[0])

    def save(self, path: str | os.PathLike) -> None:
        np.savez(path, w=self.w, b=self.b, mu=self.mu, sigma=self.sigma, names=np.array(FEATURE_NAMES))

    @classmethod
    def load(cls, path: str | os.PathLike) -> "CoverageEstimator":
        d = np.load(path, allow_pickle=False)
        return cls(d["w"], float(d["b"]), d["mu"], d["sigma"])


_DEFAULT: CoverageEstimator | None = None


def default_estimator() -> CoverageEstimator:
    """The packaged estimator (vam/weights/coverage_ridge.npz), loaded once."""
    global _DEFAULT
    if _DEFAULT is None:
        from importlib import resources

        p = resources.files("vam.weights") / DEFAULT_WEIGHTS
        if not Path(str(p)).exists():
            raise FileNotFoundError(f"packaged coverage weights missing: {p}; run scripts/train_coverage.py --release")
        _DEFAULT = CoverageEstimator.load(str(p))
    return _DEFAULT
