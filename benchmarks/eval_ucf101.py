"""Phase 1 accuracy benchmark on UCF101: accuracy vs compute for each sampler and frame budget.

Step 1 (once per backbone): extract uniform-16 features on the train split, fit a linear head.
Step 2: evaluate the fixed head on the test split under every (sampler, budget), logging per-clip
        prediction, correctness and per-stage timings to JSONL.

Usage:
    python benchmarks/eval_ucf101.py --root data/ucf101 --backbone videomae-base-kinetics \
        --train-limit 2000 --test-limit 1000 --budgets 2 4 8 16 --out results/ucf101.jsonl
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

import vam
from vam.data import UCF101
from vam.models.probe import HFFeatureExtractor, LinearProbe
from vam.sampling import MotionSampler, UniformSampler

BACKBONES = {
    "videomae-base-kinetics": "MCG-NJU/videomae-base-finetuned-kinetics",
    "videomae-small-kinetics": "MCG-NJU/videomae-small-finetuned-kinetics",
    "timesformer-base-k400": "facebook/timesformer-base-finetuned-k400",
}


def extract_train_features(extractor, ds, cache: Path):
    if cache.exists():
        d = torch.load(cache)
        return d["x"], d["y"]
    xs, ys = [], []
    t0 = time.time()
    sampler = UniformSampler(extractor.default_num_frames)
    for i, clip in enumerate(ds):
        v = clip.video()
        try:
            xs.append(extractor.features(sampler(v)))
            ys.append(clip.label)
        except Exception as e:  # corrupt clip: skip but record
            print(f"skip {clip.path}: {e}")
        finally:
            v.close()
        if (i + 1) % 200 == 0:
            print(f"  features {i + 1}/{len(ds)}  {time.time() - t0:.0f}s")
    x, y = torch.stack(xs).cpu(), torch.tensor(ys)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "y": y}, cache)
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ucf101")
    ap.add_argument("--backbone", default="videomae-base-kinetics", choices=sorted(BACKBONES))
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--test-limit", type=int, default=None)
    ap.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--samplers", nargs="+", default=["uniform", "motion"])
    ap.add_argument("--out", default="results/ucf101.jsonl")
    ap.add_argument("--cache-dir", default="results/cache")
    args = ap.parse_args()

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    train = UCF101(args.root, "train", args.fold, limit=args.train_limit)
    test = UCF101(args.root, "test", args.fold, limit=args.test_limit)
    print(gpu, "|", args.backbone, "|", train, "|", test)

    extractor = HFFeatureExtractor(BACKBONES[args.backbone])
    probe = LinearProbe(extractor, train.num_classes, train.classes)

    cache_dir = Path(args.cache_dir)
    head_path = cache_dir / f"{args.backbone}_fold{args.fold}_n{len(train)}_head.pt"
    if head_path.exists():
        probe.load_head(head_path)
        print("loaded head", head_path)
    else:
        feat_path = cache_dir / f"{args.backbone}_fold{args.fold}_n{len(train)}_train_feats.pt"
        x, y = extract_train_features(extractor, train, feat_path)
        head = LinearProbe.fit_head(x, y, train.num_classes, device=str(extractor._dev))
        probe.head.load_state_dict(head.state_dict())
        train_acc = (probe.head(x.to(extractor._dev)).argmax(1).cpu() == y).float().mean().item()
        print(f"fitted head on {len(y)} clips, train acc {train_acc:.3f}")
        probe.save_head(head_path)

    sampler_cls = {"uniform": UniformSampler, "motion": MotionSampler}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for budget in args.budgets:
            for sname in args.samplers:
                sampler = sampler_cls[sname](budget)
                correct = 0
                n = 0
                t_start = time.time()
                for clip in test:
                    v = clip.video()
                    p = vam.Profiler()
                    try:
                        with p.span("select"):
                            idx = sampler.select(v)
                        with p.span("decode"):
                            batch = v.read(idx)
                        with p.span("infer"):
                            o = probe(batch)
                    except Exception as e:
                        print(f"skip {clip.path}: {e}")
                        continue
                    finally:
                        v.close()
                    pred = int(o.logits.argmax())
                    ok = pred == clip.label
                    correct += ok
                    n += 1
                    f.write(json.dumps({
                        "ts": time.time(), "host": platform.node(), "gpu": gpu,
                        "backbone": args.backbone, "fold": args.fold, "sampler": sname,
                        "budget": budget, "clip": clip.path, "label": clip.label, "pred": pred,
                        "correct": bool(ok), "num_frames": v.num_frames, "fps": v.fps,
                        **{f"{k}_ms": s["wall_ms"] for k, s in p.as_dict().items()},
                        "infer_peak_mb": p.as_dict()["infer"]["peak_mb"],
                    }) + "\n")
                f.flush()
                print(f"{sname:8s} budget={budget:3d}  top1={correct / max(n, 1):.3f}  "
                      f"n={n}  {(time.time() - t_start) / max(n, 1) * 1000:.1f} ms/clip")


if __name__ == "__main__":
    main()
