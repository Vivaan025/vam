"""Experiment 019: fit the coverage estimator on keyframe-scan features and report routing per dataset.

    python scripts/train_coverage.py --thumos14 data/thumos14 --activitynet data/activitynet --charades data/charades \
        --out results/cache/coverage_ridge.npz [--release]

Scans every video once per (dataset, split) and caches features + targets in results/cache/coverage_feats_<ds>_<split>.npz.
Train: THUMOS14 val, ActivityNet train, Charades train. Held out: THUMOS14 test, ActivityNet val, Charades test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vam.runtime.scan import scan_keyframes  # noqa: E402
from vam.sampling.coverage import FEATURE_NAMES, CoverageEstimator, scan_features  # noqa: E402


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def collect(name: str, split: str, videos, cache_dir: Path, limit: int | None):
    cache = cache_dir / f"coverage_feats_{name}_{split}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"], d["ms"], list(d["ids"])
    X, y, ms, ids = [], [], [], []
    t_all = time.time()
    for i, uv in enumerate(videos):
        if limit and i >= limit:
            break
        try:
            t0 = time.perf_counter()
            kf = scan_keyframes(uv.path, thumb=(112, 112))
            t1 = time.perf_counter()
            f = scan_features(kf)
            t2 = time.perf_counter()
        except Exception as e:  # noqa: BLE001
            print(f"skip {uv.video_id}: {type(e).__name__}: {e}", flush=True)
            continue
        X.append(f)
        y.append(uv.action_fraction(kf.duration))
        ms.append([(t1 - t0) * 1000, (t2 - t1) * 1000])
        ids.append(uv.video_id)
        if (i + 1) % 200 == 0:
            print(f"{name}/{split}: {i + 1} videos, {(time.time() - t_all) / (i + 1) * 1000:.0f} ms/video", flush=True)
    X, y, ms = np.array(X), np.array(y), np.array(ms)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, y=y, ms=ms, ids=np.array(ids))
    return X, y, ms, ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thumos14", default=None)
    ap.add_argument("--activitynet", default=None)
    ap.add_argument("--charades", default=None)
    ap.add_argument("--cache-dir", default="results/cache")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None, help="videos per split (smoke)")
    ap.add_argument("--out", default="results/cache/coverage_ridge.npz")
    ap.add_argument("--release", action="store_true", help="also write vam/weights/coverage_ridge.npz")
    ap.add_argument("--report", default="results/coverage_019.json")
    args = ap.parse_args()

    sets: dict[tuple[str, str], tuple] = {}
    cache_dir = Path(args.cache_dir)
    if args.thumos14:
        from vam.data import THUMOS14

        for split in ("val", "test"):
            sets[("thumos14", split)] = collect("thumos14", split, THUMOS14(args.thumos14, split).videos, cache_dir, args.limit)
    if args.activitynet:
        from vam.data import ActivityNet

        for split in ("train", "val"):
            sets[("activitynet", split)] = collect("activitynet", split, ActivityNet(args.activitynet, split).videos, cache_dir, args.limit)
    if args.charades:
        from vam.data import Charades

        for split in ("train", "test"):
            sets[("charades", split)] = collect("charades", split, Charades(args.charades, split).videos, cache_dir, args.limit)

    train_keys = [("thumos14", "val"), ("activitynet", "train"), ("charades", "train")]
    held_keys = [("thumos14", "test"), ("activitynet", "val"), ("charades", "test")]
    Xtr = np.concatenate([sets[k][0] for k in train_keys if k in sets])
    ytr = np.concatenate([sets[k][1] for k in train_keys if k in sets])
    est = CoverageEstimator.fit(Xtr, ytr, alpha=args.alpha)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    est.save(args.out)
    if args.release:
        rel = Path(__file__).resolve().parents[1] / "vam" / "weights" / "coverage_ridge.npz"
        est.save(rel)
        print("wrote", rel)

    report = {"train_n": int(len(ytr)), "alpha": args.alpha, "weights": dict(zip(FEATURE_NAMES, est.w.round(4).tolist())), "sets": {}}
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"train n={len(ytr)}  weights: " + ", ".join(f"{n}={w:+.3f}" for n, w in zip(FEATURE_NAMES, est.w)))
    for key, (X, y, ms, ids) in sets.items():
        pred = est.predict_features(X)
        row = {"n": int(len(y)), "coverage_median": float(np.median(y)), "pred_median": float(np.median(pred)),
               "spearman": spearman(pred, y), "mae": float(np.abs(pred - y).mean()),
               "scan_ms_p50": float(np.median(ms[:, 0])), "feat_ms_p50": float(np.median(ms[:, 1])),
               "route_uniform": {str(t): float(np.mean(pred > t)) for t in thresholds},
               "true_high": {str(t): float(np.mean(y > t)) for t in thresholds}}
        report["sets"]["/".join(key)] = row
        tag = "held" if key in held_keys else "train"
        print(f"{key[0]:12s} {key[1]:5s} {tag:5s} n={row['n']:5d} cov_med={row['coverage_median']:.2f} pred_med={row['pred_median']:.2f} "
              f"rho={row['spearman']:.3f} mae={row['mae']:.3f} feat_ms={row['feat_ms_p50']:.1f}  route->uniform@" +
              " ".join(f"{t}:{row['route_uniform'][str(t)]:.2f}" for t in thresholds), flush=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.report, "w"), indent=1)


if __name__ == "__main__":
    main()
