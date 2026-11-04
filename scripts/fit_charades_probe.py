"""Experiment 012: fit a linear probe on frozen VideoMAE features for Charades (157 classes, multi-label).

    python scripts/fit_charades_probe.py --root data/charades --out results/cache/videomae-base-kinetics_charades_head.pt

Features: one per video, uniform 16 frames over the whole video (the same recipe as the UCF101 probe in
benchmarks/eval_ucf101.py). Head: one-vs-rest logistic regression (BCE) with the input normalisation folded
into the layer, saved in the LinearProbe.save_head format so the benchmarks load it with --head. Prints
top-1-in-labels (the decision metric, as on THUMOS14) and mAP on the test split from cached features.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from vam.data import Charades
from vam.models.probe import HFFeatureExtractor
from vam.sampling import UniformSampler

BACKBONES = {
    "videomae-base-kinetics": "MCG-NJU/videomae-base-finetuned-kinetics",
    "videomae-small-kinetics": "MCG-NJU/videomae-small-finetuned-kinetics",
}


def extract(extractor, ds, cache: Path, limit: int | None = None):
    if cache.exists():
        d = torch.load(cache)
        x = d["x"]
        if x.dim() == 1:  # extractor.features returns one (dim,) vector per video; torch.cat flattened an older cache
            x = x.view(len(d["ids"]), -1)
        return x, d["y"], d["ids"]
    sampler = UniformSampler(extractor.default_num_frames)
    xs, ys, ids = [], [], []
    vids = ds.videos[:limit] if limit else ds.videos
    t0 = time.time()
    for i, uv in enumerate(vids):
        try:
            v = uv.video()
        except OSError as e:
            print(f"skip {uv.video_id}: {e}", flush=True)
            continue
        try:
            xs.append(extractor.features(sampler(v)).float().cpu().reshape(-1))
        except Exception as e:
            print(f"skip {uv.video_id}: {e}", flush=True)
            continue
        finally:
            v.close()
        y = torch.zeros(len(ds.classes))
        for c in uv.classes:
            y[ds.class_to_idx[c]] = 1.0
        ys.append(y)
        ids.append(uv.video_id)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(vids)}  ({(time.time() - t0) / (i + 1) * 1000:.0f} ms/video)", flush=True)
    x, y = torch.stack(xs), torch.stack(ys)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "y": y, "ids": ids}, cache)
    return x, y, ids


def fit_multilabel(x: torch.Tensor, y: torch.Tensor, epochs: int, lr: float, wd: float, device: str) -> torch.nn.Linear:
    x, y = x.to(device), y.to(device)
    mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True) + 1e-6
    head = torch.nn.Linear(x.shape[1], y.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    xn = (x - mean) / std
    pos = y.mean(0).clamp(1e-3, 1 - 1e-3)
    pos_weight = ((1 - pos) / pos).sqrt()  # soften the class imbalance (median class is in ~1 % of videos)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(xn), y, pos_weight=pos_weight)
        loss.backward()
        opt.step()
    with torch.no_grad():
        head.weight.div_(std)
        head.bias.sub_((head.weight * mean).sum(1))
    return head


def mean_ap(scores: np.ndarray, y: np.ndarray) -> float:
    aps = []
    for c in range(y.shape[1]):
        if y[:, c].sum() == 0:
            continue
        order = np.argsort(-scores[:, c])
        hits = y[order, c]
        prec = np.cumsum(hits) / (np.arange(len(hits)) + 1)
        aps.append(float((prec * hits).sum() / hits.sum()))
    return float(np.mean(aps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/charades")
    ap.add_argument("--backbone", default="videomae-base-kinetics", choices=sorted(BACKBONES))
    ap.add_argument("--cache", default="results/cache/charades_feats")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-test", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train, test = Charades(args.root, "train"), Charades(args.root, "test")
    print(train, test, flush=True)
    extractor = HFFeatureExtractor(BACKBONES[args.backbone], device=device)
    cache = Path(args.cache)
    xtr, ytr, _ = extract(extractor, train, cache.with_name(cache.name + f"_{args.backbone}_train.pt"), args.limit_train)
    xte, yte, _ = extract(extractor, test, cache.with_name(cache.name + f"_{args.backbone}_test.pt"), args.limit_test)
    print(f"features: train {tuple(xtr.shape)} test {tuple(xte.shape)}", flush=True)
    head = fit_multilabel(xtr, ytr, args.epochs, args.lr, args.wd, device)
    with torch.no_grad():
        s = head(xte.to(device)).cpu().numpy()
    yn = yte.numpy()
    top1 = float(np.mean([yn[i, int(np.argmax(s[i]))] > 0 for i in range(len(yn))]))
    print(f"test: top-1-in-labels {top1:.3f}  mAP {mean_ap(s, yn):.3f}  (uniform-16 features, {len(yn)} videos)", flush=True)
    out = Path(args.out or f"results/cache/{args.backbone}_charades_head.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": {k: v.cpu() for k, v in head.state_dict().items()}, "labels": train.classes}, out)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
