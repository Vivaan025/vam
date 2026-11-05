"""Millisecond-budget scheduler (experiment 010).

The runtime has a handful of knobs (frame budget, refinement, refinement stride) whose cost and accuracy we
have measured. A user has a latency budget. The scheduler:

1. fits a per-stage cost model on the target machine (`calibrate`, a few seconds; cached per GPU + CPU count),
2. before the scan picks the most accurate configuration whose predicted latency fits, assuming a typical
   video,
3. after the scan (keyframe count and window lengths now known) re-solves once with the real numbers and
   shrinks (refinement -> frames) or grows if there is slack.

Latency model, linear in the quantities a configuration determines:

    ms = w0 + w1 * K + w2 * Lmax + w3 * C + w4 * F + w5 * R

K = keyframes scanned, Lmax = longest decoded window in frames (windows run in parallel), C = refinement
candidates = sum of window lengths / stride (CLIP thumbnails), F = frames the model sees, R = 1 if refinement
runs (its fixed cost). The accuracy table is THUMOS14 (`docs/results/`); only its ordering matters here.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class Config:
    frames: int
    refine: bool
    stride: int = 6
    uniform: bool = False   # 013: evenly spaced frames, no scan information used

    @property
    def segments(self) -> int:
        return max(1, min(4, self.frames // 2))

    def __str__(self) -> str:
        if self.uniform:
            return f"uniform-{self.frames}"
        return f"{self.frames}f" + (f"+refine/{self.stride}" if self.refine else "")


# measured top-1 on THUMOS14 val (009 for the learned scorer; stride-3 deltas from 007). Ordering is what matters.
ACCURACY: dict[Config, float] = {
    Config(4, False): 0.855, Config(4, True, 6): 0.865, Config(4, True, 3): 0.875,
    Config(8, False): 0.905, Config(8, True, 6): 0.925, Config(8, True, 3): 0.930,
    Config(16, False): 0.935, Config(16, True, 6): 0.940, Config(16, True, 3): 0.940,
    # uniform rows (THUMOS14 val, 002/007). Only chosen when the scan carries no information (013): on
    # Charades (012) uniform-16 beat every keyframe configuration because every frame is action.
    Config(4, False, uniform=True): 0.720, Config(8, False, uniform=True): 0.795,
    Config(16, False, uniform=True): 0.880,
}
CONFIGS = sorted(ACCURACY, key=lambda c: (ACCURACY[c], -c.frames, c.refine, -c.stride))


@dataclass
class Features:
    K: int
    Lmax: int
    C: int
    F: int
    R: int

    def vec(self) -> np.ndarray:
        return np.array([1.0, self.K, self.Lmax, self.C, self.F, self.R], dtype=np.float64)


def features_for(cfg: Config, K: int, window_frames: list[int]) -> Features:
    if cfg.uniform:  # scan already paid for; uniform decode is seek-bound like a window per frame
        return Features(K, 0, 0, cfg.frames, 0)
    wins = window_frames[: cfg.segments] or [0]
    Lmax = max(wins)
    C = int(sum(wins) / cfg.stride) if cfg.refine else 0
    return Features(K, Lmax, C, cfg.frames, int(cfg.refine))


@dataclass
class CostModel:
    w: np.ndarray                       # (6,)
    resid_p75: float = 0.0              # safety margin added to predictions
    typical_K: float = 40.0
    typical_window: float = 150.0       # frames
    n_fit: int = 0
    machine: str = ""

    def predict(self, f: Features) -> float:
        return float(f.vec() @ self.w) + self.resid_p75

    @classmethod
    def fit(cls, rows: list[tuple[Features, float]], machine: str = "") -> "CostModel":
        """Non-negative least squares on (features -> ms). Repeated (features) rows are reduced to their
        median first, so a one-off stall in calibration does not become the intercept."""
        by_key: dict[tuple, list[float]] = {}
        feats: dict[tuple, Features] = {}
        for f, ms in rows:
            k = (f.K, f.Lmax, f.C, f.F, f.R)
            by_key.setdefault(k, []).append(ms)
            feats[k] = f
        X = np.stack([feats[k].vec() for k in by_key])
        y = np.array([float(np.median(v)) for v in by_key.values()])
        w = _nnls(X, y)
        resid = y - X @ w
        Ks = [f.K for f in feats.values()]
        Ls = [f.Lmax for f in feats.values() if f.Lmax > 0]
        margin = float(max(0.0, np.percentile(resid, 75))) if len(y) > 2 else 0.0
        return cls(w, margin, float(np.median(Ks)) if Ks else 40.0, float(np.median(Ls)) if Ls else 150.0,
                   len(rows), machine)

    def to_json(self) -> dict:
        return {"w": self.w.tolist(), "resid_p75": self.resid_p75, "typical_K": self.typical_K,
                "typical_window": self.typical_window, "n_fit": self.n_fit, "machine": self.machine}

    @classmethod
    def from_json(cls, d: dict) -> "CostModel":
        return cls(np.asarray(d["w"], dtype=np.float64), d["resid_p75"], d["typical_K"], d["typical_window"],
                   d.get("n_fit", 0), d.get("machine", ""))


def _nnls(X: np.ndarray, y: np.ndarray, iters: int = 5000) -> np.ndarray:
    """Small non-negative least squares by projected gradient (6 unknowns; no scipy on every box)."""
    scale = np.maximum(np.abs(X).max(axis=0), 1e-9)
    Xs = X / scale
    w = np.zeros(X.shape[1])
    lr = 1.0 / (np.linalg.norm(Xs, 2) ** 2 + 1e-9)
    for _ in range(iters):
        g = Xs.T @ (Xs @ w - y)
        w_new = np.maximum(w - lr * g, 0.0)
        if np.abs(w_new - w).max() < 1e-9:
            w = w_new
            break
        w = w_new
    return w / scale


def machine_key() -> str:
    gpu = ""
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return f"{gpu or 'cpu'}|{platform.processor() or platform.machine()}|cpus={os.cpu_count()}"


def _cache_path(key: str) -> Path:
    h = hashlib.sha1(key.encode()).hexdigest()[:10]
    return Path(os.environ.get("VAM_CACHE", Path.home() / ".cache" / "vam")) / f"costmodel_{h}.json"


def calibrate(videos: list, run: Callable[[Config, object], tuple[Features, float]],
              configs: Optional[list[Config]] = None, cache: bool = True, repeats: int = 2) -> CostModel:
    """Fit the cost model on the target machine.

    `run(cfg, video)` executes the full pipeline for one video under `cfg` and returns the realised features
    and wall ms. Three videos x three configurations (no refinement at 4, refine/6 at 8, refine/3 at 16) span
    the feature space; the first call warms up and is discarded.
    """
    key = machine_key()
    path = _cache_path(key)
    if cache and path.exists():
        try:
            m = CostModel.from_json(json.loads(path.read_text()))
            if m.machine == key:
                return m
        except Exception:
            pass
    configs = configs or [Config(4, False), Config(8, True, 6), Config(16, True, 3), Config(16, False)]
    rows: list[tuple[Features, float]] = []
    for cfg in configs:  # warm-up pass (model load, pools, CUDA context, first-touch of each path) is discarded
        run(cfg, videos[0])
    for _ in range(max(1, repeats)):
        for v in videos:
            for cfg in configs:
                rows.append(run(cfg, v))
    model = CostModel.fit(rows, key)
    if cache:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(model.to_json()))
        except OSError:
            pass
    return model


@dataclass
class Plan:
    config: Config
    predicted_ms: float
    stage: str                   # "pre" or "post"
    budget_ms: float
    alternatives: int = 0        # configs that fit at this stage


class Scheduler:
    """Pick the most accurate configuration that fits a latency budget, then re-plan after the scan."""

    def __init__(self, budget_ms: float, cost: CostModel, accuracy: dict[Config, float] | None = None,
                 margin: float = 1.0, min_config: Config = Config(4, False)):
        self.budget = float(budget_ms)
        self.cost = cost
        self.accuracy = accuracy or ACCURACY
        self.margin = margin
        self.min_config = min_config
        self.last_plan: Optional[Plan] = None
        self.plans: list[Plan] = []

    def _best(self, K: int, window_frames: list[int], stage: str, flat: bool = False) -> Plan:
        """Most accurate configuration that fits. With `flat` (the scan could not rank the keyframes) only
        uniform configurations compete; otherwise uniform ones are excluded (013)."""
        fits = []
        for cfg in self.accuracy:
            if cfg.uniform != flat:
                continue
            ms = self.cost.predict(features_for(cfg, K, window_frames)) * self.margin
            if ms <= self.budget:
                fits.append((self.accuracy[cfg], -cfg.frames, -cfg.stride, len(fits), cfg, ms))
        if fits:
            cfg, ms = max(fits)[-2:]
            return Plan(cfg, ms, stage, self.budget, len(fits))
        cfg = Config(self.min_config.frames, False, uniform=True) if flat else self.min_config
        return Plan(cfg, self.cost.predict(features_for(cfg, K, window_frames)) * self.margin, stage, self.budget, 0)

    def pre_plan(self, duration_s: float | None = None, fps: float = 30.0) -> Plan:
        K = int(self.cost.typical_K if duration_s is None else max(2, duration_s / 4.5))
        win = int(self.cost.typical_window)
        plan = self._best(K, [win] * 4, "pre")
        self.last_plan = plan
        return plan

    def post_plan(self, K: int, window_frames: list[int], flat: bool = False) -> Plan:
        plan = self._best(K, sorted(window_frames, reverse=True), "post", flat)
        self.last_plan = plan
        self.plans.append(plan)
        return plan

    @staticmethod
    def scan_is_flat(scores, flat_eps: float = 1e-3) -> bool:
        s = np.asarray(scores, dtype=np.float64)
        if len(s) < 2:
            return True
        return bool(np.std(s) < flat_eps * max(float(np.abs(s).mean()), 1e-12))

    def planner(self, flat_eps: float = 1e-3, max_coverage: float = 0.5):
        """Callback for KeyframeSampler(planner=...): receives (kf, scores, chosen, windows) after the scan and
        returns (num_frames, refine_on, refine_stride, uniform). The plan is a uniform configuration when the
        scan carries no information: one keyframe, scores the scorer cannot separate, or chosen windows that
        would cover more than `max_coverage` of the video, so there is nothing to select (013, from 012)."""
        def _plan(kf, scores, chosen, wins):
            coverage = sum(e - s for s, e in wins) / max(int(getattr(kf, "num_frames", 0)), 1)
            flat = self.scan_is_flat(scores, flat_eps) or coverage > max_coverage
            p = self.post_plan(len(kf), [e - s for s, e in wins], flat)
            return p.config.frames, p.config.refine, p.config.stride, p.config.uniform
        return _plan
