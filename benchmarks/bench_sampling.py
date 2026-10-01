"""Phase 1 benchmark: profile decode, sampling and inference per sampler and frame budget.

Usage:
    python benchmarks/bench_sampling.py path/to/video.mp4 --model videomae-base-kinetics \
        --budgets 4 8 16 --out results.jsonl

Each run appends one JSON line with GPU name, sampler, budget, timings, peak memory and
top-1 prediction, so runs from different machines (RTX 3050 vs RTX 5090) can be compared.
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import torch

import vam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="videomae-base-kinetics")
    ap.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results.jsonl")
    args = ap.parse_args()

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    model = vam.get_model(args.model)
    video = vam.load_video(args.video)
    print(f"{gpu} | {args.model} | {video.num_frames} frames @ {video.fps:.1f} fps")

    samplers = {
        "uniform": vam.UniformSampler,
        "motion": vam.MotionSampler,
    }

    with open(args.out, "a") as f:
        for budget in args.budgets:
            for name, cls in samplers.items():
                sampler = cls(budget)
                for r in range(args.repeats):
                    p = vam.Profiler()
                    with p.span("select"):
                        idx = sampler.select(video)
                    with p.span("decode"):
                        batch = video.read(idx)
                    with p.span("infer"):
                        out = model(batch)
                    top = out.top_k(1)[0]
                    row = {
                        "ts": time.time(),
                        "host": platform.node(),
                        "gpu": gpu,
                        "model": args.model,
                        "video": args.video,
                        "sampler": name,
                        "budget": budget,
                        "repeat": r,
                        "top1": top[0],
                        "p_top1": top[1],
                        **{f"{k}_ms": v["wall_ms"] for k, v in p.as_dict().items()},
                        "infer_peak_mb": p.as_dict()["infer"]["peak_mb"],
                    }
                    f.write(json.dumps(row) + "\n")
                    if r == args.repeats - 1:
                        total = sum(v["wall_ms"] for v in p.as_dict().values())
                        print(f"{name:8s} budget={budget:3d}  {total:8.1f} ms  {top[0]} ({top[1]:.2f})")
    video.close()


if __name__ == "__main__":
    main()
