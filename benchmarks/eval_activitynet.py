"""Experiment 017: ActivityNet-1.3 / Mini-Kinetics under the FrameExit protocol, one classifier, many samplers.

Arms (``--arms``):
  frameexit      FrameExit's own 10-frame order and early exit (H1: reproduces 0.7612 mAP / 0.7331 top-1)
  fe-uniform-10  the same 10 frames, no exit, head for 10 frames (FrameExit's upper bound)
  uniform-K      vam UniformSampler(K) -> head for K frames
  keyframe-K     vam KeyframeSampler(K) with the packaged scorer (or --scorer-head), refinement as in 007
  budget-B       vam BudgetedSampler(B ms), cost model calibrated on this machine

GMACs follow FrameExit's accounting (4.12 per ResNet-50 frame) plus 4.4 per CLIP ViT-B/32 image the sampler
scores (keyframes and refinement candidates). Measured milliseconds are select / decode / infer per video.

    VAM_FRAMEEXIT_DIR=data/tools/checkpoints python benchmarks/eval_activitynet.py --root data/activitynet \
        --arms frameexit fe-uniform-10 uniform-4 uniform-6 uniform-8 uniform-10 keyframe-4 keyframe-6 keyframe-8 budget-100 budget-150 budget-200
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import vam
from vam.core import FrameBatch
from vam.data import ActivityNet, MiniKinetics
from vam.models.frameexit import GMACS_PER_FRAME, FrameExitModel
from vam.sampling import KeyframeSampler, UniformSampler

CLIP_GMACS = 4.4  # ViT-B/32 image tower at 224, same FLOPs-as-MACs convention FrameExit uses for ResNet-50


def frameexit_indices(video, n: int = 10, fps_extract: float = 1.0) -> list[int]:
    """FrameExit's uniform positions over frames extracted at `fps_extract` (1 fps ActivityNet, 5 fps Mini-K)."""
    n_ext = max(1, int(video.duration * fps_extract))
    pos = np.clip(np.linspace(0, n_ext, n), 0, n_ext - 1).astype(int)
    return [min(video.num_frames - 1, int(round(p / fps_extract * video.fps))) for p in pos]


def average_precision(scores: np.ndarray, y: np.ndarray) -> float:
    """mAP as FrameExit computes it: per-class AP over videos, mean over classes with positives."""
    aps = []
    for c in range(scores.shape[1]):
        pos = y[:, c] > 0
        if pos.sum() == 0:
            continue
        order = np.argsort(-scores[:, c])
        hits = pos[order]
        prec = np.cumsum(hits) / (np.arange(len(hits)) + 1)
        aps.append(float((prec * hits).sum() / hits.sum()))
    return float(np.mean(aps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="activitynet", choices=["activitynet", "minikinetics"])
    ap.add_argument("--root", default="data/activitynet")
    ap.add_argument("--split", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--arms", nargs="+", default=["frameexit", "fe-uniform-10", "uniform-4", "uniform-6", "uniform-8", "uniform-10",
                                                  "keyframe-4", "keyframe-6", "keyframe-8"])
    ap.add_argument("--scorer-head", default=None, help="LearnedScorer weights; default = packaged THUMOS scorer")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else f"cpu ({platform.processor() or 'unknown'})"
    if args.dataset == "activitynet":
        ds = ActivityNet(args.root, args.split or "validation")
        fps_extract = 1.0
    else:
        ds = MiniKinetics(args.root, args.split or "val")
        fps_extract = 5.0
    if args.limit:
        ds.videos = ds.videos[: args.limit]
    model = FrameExitModel(args.dataset, checkpoint=args.checkpoint, device=device)
    model.labels = ds.classes
    print(gpu, "|", ds, "| checkpoint", getattr(model, "checkpoint", None), flush=True)

    scorer = None
    if args.scorer_head:
        from vam.sampling import ClipScorer, LearnedScorer

        scorer = LearnedScorer.load(args.scorer_head, ClipScorer((), device=device))
    else:
        from vam.sampling.learned import default_scorer

        scorer = default_scorer(device=device)

    def make(arm: str):
        if arm in ("frameexit", "fe-uniform-10"):
            return None
        kind, val = arm.rsplit("-", 1)
        if kind == "uniform":
            return UniformSampler(int(val))
        if kind == "keyframe":
            k = int(val)
            return KeyframeSampler(k, scorer, refine=None if args.no_refine else scorer)
        if kind == "budget":
            return vam.BudgetedSampler(float(val), model, calibrate_on=[uv.path for uv in ds.videos[:3]], scorer=scorer)
        raise ValueError(arm)

    out = Path(args.out or f"results/{args.dataset}_017.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    n_cls = len(ds.classes)
    with out.open("a") as f:
        for arm in args.arms:
            sampler = make(arm)
            scores, ys, ms_tot, gmacs, used, n_fb = [], [], [], [], [], 0
            t_start = time.time()
            for uv in ds:
                try:
                    v = uv.video()
                except Exception as e:  # noqa: BLE001
                    print(f"skip {uv.video_id}: {e}", flush=True)
                    continue
                p = vam.Profiler()
                try:
                    if sampler is None:
                        with p.span("select"):
                            idx = frameexit_indices(v, 10, fps_extract)
                        with p.span("decode"):
                            batch: FrameBatch = v.read(idx)
                        with p.span("infer"):
                            if arm == "frameexit":
                                logits, k_used = model.exit_forward(batch)
                            else:
                                logits, k_used = model(batch).logits, len(idx)
                        extra = 0.0
                    else:
                        with p.span("select"):
                            idx = sampler.select(v)
                        with p.span("decode"):
                            batch = sampler.read(v, idx)
                        with p.span("infer"):
                            logits = model(batch).logits
                        k_used = len(idx)
                        ks = sampler.last if isinstance(sampler, vam.BudgetedSampler) else sampler
                        n_scored = len(getattr(ks, "last_keyframes", None) or []) if isinstance(ks, KeyframeSampler) else 0
                        fine = getattr(ks, "last_fine", None) if isinstance(ks, KeyframeSampler) else None
                        n_scored += len(fine[0]) if fine and fine[0] is not None else 0
                        extra = CLIP_GMACS * n_scored
                        n_fb += getattr(ks, "last_fallback", None) is not None
                    if device == "cuda":
                        torch.cuda.synchronize()
                except Exception as e:  # noqa: BLE001
                    print(f"skip {uv.video_id}: {type(e).__name__}: {e}", flush=True)
                    continue
                finally:
                    duration, fps = v.duration, v.fps
                    v.close()
                d = p.as_dict()
                ms = {k: d[k]["wall_ms"] for k in ("select", "decode", "infer")}
                y = np.zeros(n_cls)
                y[ds.labels(uv)] = 1
                prob = torch.softmax(logits, 0).numpy()
                scores.append(prob)
                ys.append(y)
                ms_tot.append(sum(ms.values()))
                g = GMACS_PER_FRAME * k_used + extra
                gmacs.append(g)
                used.append(k_used)
                plan = getattr(sampler, "last_plan", None)
                f.write(json.dumps({
                    "ts": time.time(), "host": platform.node(), "gpu": gpu, "dataset": args.dataset, "split": ds.split,
                    "arm": arm, "video": uv.video_id, "duration_s": duration, "fps": fps, "labels": ds.labels(uv),
                    "pred": int(prob.argmax()), "hit1": bool(y[int(prob.argmax())] > 0), "frames_used": k_used,
                    "gmacs": g, "config": str(plan.config) if plan else None, "predicted_ms": plan.predicted_ms if plan else None,
                    "fallback": getattr(sampler.last if isinstance(sampler, vam.BudgetedSampler) else sampler, "last_fallback", None),
                    **{f"{k}_ms": v_ for k, v_ in ms.items()}, "total_ms": sum(ms.values()),
                    "scores": [round(float(x), 5) for x in prob],
                }) + "\n")
            f.flush()
            S, Y = np.array(scores), np.array(ys)
            if len(S) == 0:
                print(f"{arm}: no videos", flush=True)
                continue
            mAP = average_precision(S, Y)
            hit1 = float(np.mean(Y[np.arange(len(S)), S.argmax(1)] > 0))
            a = np.array(ms_tot)
            print(f"{arm:14s} n={len(S):5d}  mAP={mAP:.4f}  hit@1={hit1:.4f}  frames={np.mean(used):.2f}  GMACs={np.mean(gmacs):.1f}  "
                  f"p50={np.median(a):.0f} p90={np.percentile(a, 90):.0f} ms  fallback={n_fb}  ({(time.time() - t_start) / len(S) * 1000:.0f} ms/video wall)", flush=True)
            f.write(json.dumps({"summary": True, "arm": arm, "dataset": args.dataset, "n": len(S), "mAP": mAP, "hit1": hit1,
                                "frames": float(np.mean(used)), "gmacs": float(np.mean(gmacs)), "p50_ms": float(np.median(a)),
                                "p90_ms": float(np.percentile(a, 90)), "fallback": n_fb, "gpu": gpu, "scorer_head": args.scorer_head}) + "\n")
            f.flush()
    vam.runtime.shutdown_pools()


if __name__ == "__main__":
    main()
