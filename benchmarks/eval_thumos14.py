"""Long-video benchmark: does frame selection matter when the action is a sliver of the timeline?

Setup
  * THUMOS14 validation videos (untrimmed, minutes long, 20 classes).
  * Head: the linear probe fitted on UCF101 (benchmarks/eval_ucf101.py), restricted to the 20
    THUMOS classes. No training on THUMOS at all.
  * A video counts as correct if the argmax over the 20 classes is one of its annotated classes.

Samplers
  * uniform, motion  : the real contenders.
  * oracle           : uniform *inside annotated segments* — an upper bound on what any
                       selection policy could achieve. The uniform→oracle gap is the headroom.

Per video and config we log accuracy, per-stage timings, action fraction, and the share of
selected frames that fall inside an action segment ("hit rate").

    python benchmarks/eval_thumos14.py --head results/cache/<head>.pt --budgets 4 8 16 32
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
from vam.data import THUMOS14, UCF101, UntrimmedVideo
from vam.models.probe import HFFeatureExtractor, LinearProbe
from vam.sampling import MotionSampler, Sampler, UniformSampler

BACKBONES = {
    "videomae-base-kinetics": "MCG-NJU/videomae-base-finetuned-kinetics",
    "videomae-small-kinetics": "MCG-NJU/videomae-small-finetuned-kinetics",
}


class OracleSampler(Sampler):
    """Evenly spaced frames restricted to annotated action segments. Needs ground truth."""

    def __init__(self, num_frames: int, uv: UntrimmedVideo):
        super().__init__(num_frames)
        self.uv = uv

    def select(self, video):
        fps = video.fps
        ranges = []
        for s in self.uv.segments:
            a, b = int(s.start * fps), min(int(s.end * fps), video.num_frames - 1)
            if b > a:
                ranges.append((a, b))
        if not ranges:
            return UniformSampler(self.num_frames).select(video)
        lengths = np.array([b - a for a, b in ranges], dtype=float)
        total = lengths.sum()
        targets = (np.arange(self.num_frames) + 0.5) / self.num_frames * total
        bounds = np.cumsum(lengths)
        out = []
        for t in targets:
            k = int(np.searchsorted(bounds, t, side="right"))
            k = min(k, len(ranges) - 1)
            off = t - (bounds[k] - lengths[k])
            out.append(int(ranges[k][0] + off))
        return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/thumos14")
    ap.add_argument("--ucf-root", default="data/ucf101")
    ap.add_argument("--backbone", default="videomae-base-kinetics", choices=sorted(BACKBONES))
    ap.add_argument("--head", required=True, help="linear probe head .pt from eval_ucf101.py")
    ap.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--samplers", nargs="+", default=["uniform", "motion", "oracle"])
    ap.add_argument("--motion-stride", type=int, default=4, help="score every k-th frame for motion")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/thumos14.jsonl")
    args = ap.parse_args()

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    ucf = UCF101(args.ucf_root, "train", 1, limit=1)  # only for the class list
    ds = THUMOS14(args.root, "val")
    if args.limit:
        ds.videos = ds.videos[: args.limit]
    missing = [c for c in ds.classes if c not in ucf.class_to_idx]
    if missing:
        raise SystemExit(f"THUMOS classes not in UCF101: {missing}")
    sub_idx = torch.tensor([ucf.class_to_idx[c] for c in ds.classes])
    print(gpu, "|", args.backbone, "|", ds, flush=True)

    extractor = HFFeatureExtractor(BACKBONES[args.backbone])
    probe = LinearProbe(extractor, ucf.num_classes, ucf.classes).load_head(args.head)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for budget in args.budgets:
            for sname in args.samplers:
                correct = n = 0
                hit_sum = 0.0
                t_start = time.time()
                for uv in ds:
                    try:
                        v = uv.video()
                    except OSError as e:
                        print(f"skip {uv.video_id}: {e}", flush=True)
                        continue
                    if sname == "uniform":
                        sampler = UniformSampler(budget)
                    elif sname == "motion":
                        sampler = MotionSampler(budget, stride=args.motion_stride)
                    else:
                        sampler = OracleSampler(budget, uv)
                    p = vam.Profiler()
                    try:
                        with p.span("select"):
                            idx = sampler.select(v)
                        with p.span("decode"):
                            batch = v.read(idx)
                        with p.span("infer"):
                            o = probe(batch)
                    except Exception as e:
                        print(f"skip {uv.video_id}: {e}", flush=True)
                        continue
                    finally:
                        duration, fps, nf = v.duration, v.fps, v.num_frames
                        v.close()
                    logits20 = o.logits[sub_idx]
                    pred_cls = ds.classes[int(logits20.argmax())]
                    ok = pred_cls in uv.classes
                    hit = float(np.mean([uv.frame_in_action(i / fps) for i in idx])) if idx else 0.0
                    correct += ok
                    hit_sum += hit
                    n += 1
                    f.write(json.dumps({
                        "ts": time.time(), "host": platform.node(), "gpu": gpu, "backbone": args.backbone,
                        "sampler": sname, "budget": budget, "video": uv.video_id, "duration_s": duration,
                        "num_frames": nf, "action_fraction": uv.action_fraction(duration),
                        "labels": sorted(uv.classes), "pred": pred_cls, "correct": bool(ok), "hit_rate": hit,
                        **{f"{k}_ms": s["wall_ms"] for k, s in p.as_dict().items()},
                        "infer_peak_mb": p.as_dict()["infer"]["peak_mb"],
                    }) + "\n")
                f.flush()
                print(f"{sname:8s} budget={budget:3d}  top1={correct / max(n, 1):.3f}  "
                      f"hit={hit_sum / max(n, 1):.2f}  n={n}  "
                      f"{(time.time() - t_start) / max(n, 1) * 1000:.0f} ms/video", flush=True)


if __name__ == "__main__":
    main()
