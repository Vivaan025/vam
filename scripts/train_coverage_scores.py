"""Experiment 019, second estimator: coverage from the learned scorer's keyframe scores.

Reuses the CLIP keyframe caches of scripts/train_scorer.py (times + embeddings per video) and computes the
packaged class-free scorer's scores from them; videos without a cache are scanned and embedded here and cached
in the same format. Targets and durations come from the scan-feature caches of scripts/train_coverage.py.

    python scripts/train_coverage_scores.py --thumos14 data/thumos14 --activitynet data/activitynet --charades data/charades \
        --out results/cache/coverage_scores.npz [--release]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vam.runtime.scan import scan_keyframes  # noqa: E402
from vam.sampling.coverage import SCORE_FEATURE_NAMES, CoverageEstimator, score_features  # noqa: E402
from vam.sampling.learned import default_scorer  # noqa: E402

EXISTING = {("thumos14", "val"): "scorer_feats.pt", ("activitynet", "train"): "scorer_feats_anet.pt",
            ("charades", "test"): "scorer_feats_charades.pt"}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def embed_cache(name, split, videos, cache_dir: Path, scorer, limit):
    path = cache_dir / EXISTING.get((name, split), f"scorer_feats_{name}_{split}.pt")
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)["feats"]
    feats, t0 = {}, time.time()
    for i, uv in enumerate(videos):
        if limit and i >= limit:
            break
        try:
            kf = scan_keyframes(uv.path, thumb=(112, 112))
            e = scorer.embedder.embed(kf.thumbs)
        except Exception as ex:  # noqa: BLE001
            print(f"skip {uv.video_id}: {type(ex).__name__}: {ex}", flush=True)
            continue
        feats[uv.video_id] = {"times": torch.as_tensor(kf.times, dtype=torch.float32), "emb": e.cpu(),
                              "duration": float(kf.duration)}
        if (i + 1) % 200 == 0:
            print(f"{name}/{split}: embedded {i + 1} ({(time.time() - t0) / (i + 1) * 1000:.0f} ms/video)", flush=True)
    torch.save({"feats": feats}, path)
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thumos14", default=None)
    ap.add_argument("--activitynet", default=None)
    ap.add_argument("--charades", default=None)
    ap.add_argument("--cache-dir", default="results/cache")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/cache/coverage_scores.npz")
    ap.add_argument("--release", action="store_true")
    ap.add_argument("--report", default="results/coverage_scores_019.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scorer = default_scorer(device=device)
    cache_dir = Path(args.cache_dir)

    specs = []
    if args.thumos14:
        from vam.data import THUMOS14

        specs += [("thumos14", s, lambda s=s: THUMOS14(args.thumos14, s).videos) for s in ("val", "test")]
    if args.activitynet:
        from vam.data import ActivityNet

        specs += [("activitynet", s, lambda s=s: ActivityNet(args.activitynet, s).videos) for s in ("train", "val")]
    if args.charades:
        from vam.data import Charades

        specs += [("charades", s, lambda s=s: Charades(args.charades, s).videos) for s in ("train", "test")]

    sets = {}
    for name, split, loader in specs:
        videos = loader()
        feats = embed_cache(name, split, videos, cache_dir, scorer, args.limit)
        cov = np.load(cache_dir / f"coverage_feats_{name}_{split}.npz")
        ids = list(cov["ids"])
        dur = {i: float(np.expm1(x[0])) for i, x in zip(ids, cov["X"])}
        tgt = dict(zip(ids, cov["y"]))
        X, y, ms, kept = [], [], [], []
        for vid, f in feats.items():
            if vid not in tgt:
                continue
            t0 = time.perf_counter()
            sc = scorer.scores_from_embedding(f["emb"], f["times"].numpy())
            d = f.get("duration", dur[vid])
            X.append(score_features(sc, f["times"].numpy(), d))
            ms.append((time.perf_counter() - t0) * 1000)
            y.append(tgt[vid])
            kept.append(vid)
        sets[(name, split)] = (np.array(X), np.array(y), np.array(ms), kept)
        print(f"{name}/{split}: {len(kept)} videos with scores + targets", flush=True)

    train_keys = [("thumos14", "val"), ("activitynet", "train"), ("charades", "train")]
    held_keys = [("thumos14", "test"), ("activitynet", "val"), ("charades", "test")]
    Xtr = np.concatenate([sets[k][0] for k in train_keys if k in sets])
    ytr = np.concatenate([sets[k][1] for k in train_keys if k in sets])
    est = CoverageEstimator.fit(Xtr, ytr, alpha=args.alpha, kind="scores")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    est.save(args.out)
    if args.release:
        rel = Path(__file__).resolve().parents[1] / "vam" / "weights" / "coverage_scores.npz"
        est.save(rel)
        print("wrote", rel)
    report = {"train_n": int(len(ytr)), "alpha": args.alpha, "kind": "scores",
              "weights": dict(zip(SCORE_FEATURE_NAMES, est.w.round(4).tolist())), "sets": {}}
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"train n={len(ytr)}  weights: " + ", ".join(f"{n}={w:+.3f}" for n, w in zip(SCORE_FEATURE_NAMES, est.w)))
    for key, (X, y, ms, ids) in sets.items():
        pred = est.predict_features(X)
        row = {"n": int(len(y)), "coverage_median": float(np.median(y)), "pred_median": float(np.median(pred)),
               "spearman": spearman(pred, y), "mae": float(np.abs(pred - y).mean()), "score_ms_p50": float(np.median(ms)),
               "route_uniform": {str(t): float(np.mean(pred > t)) for t in thresholds},
               "true_high": {str(t): float(np.mean(y > t)) for t in thresholds}}
        report["sets"]["/".join(key)] = row
        tag = "held" if key in held_keys else "train"
        print(f"{key[0]:12s} {key[1]:5s} {tag:5s} n={row['n']:5d} cov_med={row['coverage_median']:.2f} pred_med={row['pred_median']:.2f} "
              f"rho={row['spearman']:.3f} mae={row['mae']:.3f} score_ms={row['score_ms_p50']:.1f}  route->uniform@" +
              " ".join(f"{t}:{row['route_uniform'][str(t)]:.2f}" for t in thresholds), flush=True)
    json.dump(report, open(args.report, "w"), indent=1)


if __name__ == "__main__":
    main()
