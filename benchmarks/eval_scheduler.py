"""Experiment 010: does a millisecond-budget scheduler match the best static configuration at every budget?

Per budget it calibrates a cost model on the target machine (3 videos, cached), then for every THUMOS14 val
video lets the Scheduler pick the configuration before and after the scan, and logs the chosen configuration,
predicted and realised milliseconds, and correctness.

    python benchmarks/eval_scheduler.py --head results/cache/<head>.pt --scorer-head results/cache/scorer_context_nozs.pt \
        --budgets 100 125 150 200 300
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
from vam.data import THUMOS14, UCF101
from vam.models.probe import HFFeatureExtractor, LinearProbe
from vam.runtime.scheduler import Config, Features, Scheduler, calibrate, features_for
from vam.sampling import ClipScorer, KeyframeSampler, LearnedScorer

BACKBONES = {
    "videomae-base-kinetics": "MCG-NJU/videomae-base-finetuned-kinetics",
    "videomae-small-kinetics": "MCG-NJU/videomae-small-finetuned-kinetics",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/thumos14")
    ap.add_argument("--split", default="val", choices=["val", "test"], help="THUMOS14 split (011: test is held out)")
    ap.add_argument("--ucf-root", default="data/ucf101")
    ap.add_argument("--backbone", default="videomae-base-kinetics", choices=sorted(BACKBONES))
    ap.add_argument("--head", required=True)
    ap.add_argument("--scorer-head", required=True)
    ap.add_argument("--budgets", type=float, nargs="+", default=[100, 125, 150, 200, 300])
    ap.add_argument("--margin", type=float, default=1.0, help="multiply predictions by this before comparing to the budget")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--calib-videos", type=int, default=3)
    ap.add_argument("--calib-repeats", type=int, default=2)
    ap.add_argument("--no-cache", action="store_true", help="re-fit the cost model instead of using the cached one")
    ap.add_argument("--out", default="results/thumos14_scheduler.jsonl")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else f"cpu ({platform.processor() or 'unknown'})"
    ucf = UCF101(args.ucf_root, "train", 1, limit=1)
    ds = THUMOS14(args.root, args.split)
    if args.limit:
        ds.videos = ds.videos[: args.limit]
    sub_idx = torch.tensor([ucf.class_to_idx[c] for c in ds.classes])
    print(gpu, "|", args.backbone, "|", ds, flush=True)

    clip = ClipScorer(ds.classes, device=device)
    learned = LearnedScorer.load(args.scorer_head, clip)
    extractor = HFFeatureExtractor(BACKBONES[args.backbone], device=device)
    probe = LinearProbe(extractor, ucf.num_classes, ucf.classes).load_head(args.head)

    def make_sampler(cfg: Config, video_id: str, planner=None) -> KeyframeSampler:
        sc = learned.for_video(video_id)
        # With a planner the post-scan plan may switch refinement on, so the sampler always carries the scorer
        # and the planner toggles it (011 erratum: before this, a grow-into-refinement was logged but never ran).
        return KeyframeSampler(cfg.frames, sc, segments=cfg.segments, alloc="softmax", temperature=0.2,
                               refine=sc if (cfg.refine or planner is not None) else None,
                               refine_stride=cfg.stride, planner=planner)

    def run_once(sampler: KeyframeSampler, v):
        p = vam.Profiler()
        with p.span("select"):
            idx = sampler.select(v)
        with p.span("decode"):
            batch = sampler.read(v, idx)
        with p.span("infer"):
            o = probe(batch)
        d = p.as_dict()
        ms = {k: d[k]["wall_ms"] for k in ("select", "decode", "infer")}
        return idx, o, ms

    def calib_run(cfg: Config, uv):
        v = uv.video()
        try:
            s = make_sampler(cfg, uv.video_id)
            idx, o, ms = run_once(s, v)
            wins = [e - st for st, e in s.last_windows]
            f = features_for(cfg, len(s.last_keyframes), wins)
            return f, sum(ms.values())
        finally:
            v.close()

    t0 = time.time()
    cost = calibrate(ds.videos[: args.calib_videos], calib_run, cache=not args.no_cache, repeats=args.calib_repeats)
    print(f"cost model ({cost.n_fit} rows, {time.time() - t0:.1f} s): w={np.round(cost.w, 3).tolist()} "
          f"p75 resid={cost.resid_p75:.1f} typical K={cost.typical_K:.0f} window={cost.typical_window:.0f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for budget in args.budgets:
            sched = Scheduler(budget, cost, margin=args.margin)
            correct = n = within = 0
            real, errs, chosen_cfgs = [], [], []
            t_start = time.time()
            for uv in ds:
                try:
                    v = uv.video()
                except OSError as e:
                    print(f"skip {uv.video_id}: {e}", flush=True)
                    continue
                pre = sched.pre_plan(v.duration, v.fps)
                sampler = make_sampler(pre.config, uv.video_id, planner=sched.planner())
                try:
                    idx, o, ms = run_once(sampler, v)
                except Exception as e:
                    print(f"skip {uv.video_id}: {e}", flush=True)
                    continue
                finally:
                    duration, fps = v.duration, v.fps
                    v.close()
                post = sched.last_plan
                total = sum(ms.values())
                logits20 = o.logits[sub_idx]
                pred_cls = ds.classes[int(logits20.argmax())]
                ok = pred_cls in uv.classes
                hit = float(np.mean([uv.frame_in_action(i / fps) for i in idx])) if idx else 0.0
                correct += ok
                n += 1
                within += total <= budget
                real.append(total)
                errs.append(total - post.predicted_ms)
                chosen_cfgs.append(str(post.config))
                f.write(json.dumps({
                    "ts": time.time(), "host": platform.node(), "gpu": gpu, "backbone": args.backbone, "split": args.split, "root": args.root,
                    "sampler": f"sched@{budget:.0f}", "budget": int(budget), "video": uv.video_id,
                    "duration_s": duration, "labels": sorted(uv.classes), "pred": pred_cls, "correct": bool(ok),
                    "hit_rate": hit, "pre_config": str(pre.config), "config": str(post.config),
                    "frames": post.config.frames, "refine": post.config.refine, "stride": post.config.stride,
                    "predicted_ms": post.predicted_ms, "alternatives": post.alternatives,
                    **{f"{k}_ms": v_ for k, v_ in ms.items()}, "total_ms": total,
                }) + "\n")
            f.flush()
            real_a = np.array(real)
            cfg_counts = {c: chosen_cfgs.count(c) for c in sorted(set(chosen_cfgs))}
            print(f"budget={budget:4.0f}  top1={correct / max(n, 1):.3f}  within={within / max(n, 1):.2f}  "
                  f"p50={np.median(real_a):.0f} p90={np.percentile(real_a, 90):.0f} ms  "
                  f"pred err mean={np.mean(errs):+.0f} mae={np.mean(np.abs(errs)):.0f}  configs={cfg_counts}  "
                  f"({(time.time() - t_start) / max(n, 1) * 1000:.0f} ms/video wall)", flush=True)


if __name__ == "__main__":
    main()
