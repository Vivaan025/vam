"""Experiment 009: train the learned keyframe scorer with 5-fold cross-validation by video.

Step 1 caches per-video features (keyframe times, CLIP embeddings, zero-shot scores, oracle gap-window
targets); step 2 trains a head per fold and reports scorer-level metrics on the held-out fold; the checkpoint
stores all fold heads plus the video -> fold map so the benchmark scores each video with an unseen head.

    python scripts/train_scorer.py --cache results/cache/scorer_feats.pt                       # step 1 (+2 with defaults)
    python scripts/train_scorer.py --cache ... --kind context --context 2 --out results/cache/scorer_context.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from vam.data import THUMOS14
from vam.runtime.scan import scan_keyframes
from vam.sampling import ClipScorer
from vam.sampling.learned import ScorerHead, build_features, gap_window_targets


def cache_features(root: str, out: Path, device: str, thumb: int) -> dict:
    ds = THUMOS14(root, "val")
    clip = ClipScorer(ds.classes, device=device)
    feats = {}
    t0 = time.time()
    for n, uv in enumerate(ds):
        kf = scan_keyframes(uv.path, thumb=(thumb, thumb))
        e = clip.embed(kf.thumbs)
        feats[uv.video_id] = {
            "times": torch.as_tensor(kf.times, dtype=torch.float32),
            "emb": e.cpu(),
            "zs": torch.as_tensor(clip.scores_from_embedding(e), dtype=torch.float32),
            "target": torch.as_tensor(gap_window_targets(kf.times, [(s.start, s.end) for s in uv.segments]),
                                      dtype=torch.float32),
            "classes": sorted(uv.classes),
        }
        if n % 20 == 0:
            print(f"cached {n + 1}/{len(ds)} ({time.time() - t0:.0f} s)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"classes": ds.classes, "thumb": thumb, "feats": feats}, out)
    return feats


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def top_k_precision(pred: np.ndarray, target: np.ndarray, k: int = 2) -> float:
    """Fraction of the k best-predicted keyframes whose oracle score is > 0 (window overlaps action)."""
    if len(pred) == 0:
        return float("nan")
    top = np.argsort(-pred, kind="stable")[:k]
    return float((target[top] > 0).mean())


def train_fold(train_x, train_y, train_vid, val_x, val_y, val_vid, kind, hidden, device, epochs, lr, wd, seed):
    torch.manual_seed(seed)
    head = ScorerHead(train_x.shape[1], hidden, kind).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    x, y = train_x.to(device), train_y.to(device)
    vid = torch.as_tensor(train_vid, device=device)
    best, best_state, best_ep = -1e9, None, 0
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(x), device=device)
        for i in range(0, len(x), 512):
            b = perm[i:i + 512]
            pred = head(x[b])
            mse = ((pred - y[b]) ** 2).mean()
            # pairwise ranking within the same video: higher target -> higher score (margin 0.1)
            same = vid[b][:, None] == vid[b][None, :]
            dy = y[b][:, None] - y[b][None, :]
            dp = pred[:, None] - pred[None, :]
            mask = same & (dy > 0.05)
            rank = torch.relu(0.1 - dp)[mask].mean() if mask.any() else pred.sum() * 0
            loss = mse + 0.5 * rank
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 5 == 4 or ep == epochs - 1:
            m = evaluate(head, val_x, val_y, val_vid, device)
            if m["spearman"] > best:
                best, best_ep = m["spearman"], ep
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, best_ep


@torch.no_grad()
def evaluate(head, x, y, vid, device) -> dict:
    head.eval()
    pred = head(x.to(device)).cpu().numpy()
    y = y.numpy()
    vid = np.asarray(vid)
    sp, p2, p4, chosen = [], [], [], []
    for v in np.unique(vid):
        m = vid == v
        sp.append(spearman(pred[m], y[m]))
        p2.append(top_k_precision(pred[m], y[m], 2))
        p4.append(top_k_precision(pred[m], y[m], 4))
        chosen.append(y[m][np.argsort(-pred[m], kind="stable")[:2]].mean())
    return {"spearman": float(np.nanmean(sp)), "top2_prec": float(np.nanmean(p2)), "top4_prec": float(np.nanmean(p4)),
            "chosen_overlap": float(np.mean(chosen))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/thumos14")
    ap.add_argument("--cache", default="results/cache/scorer_feats.pt")
    ap.add_argument("--thumb", type=int, default=112)
    ap.add_argument("--kind", default="mlp", choices=["ridge", "mlp", "context"])
    ap.add_argument("--context", type=int, default=2, help="neighbours each side (used when --kind context)")
    ap.add_argument("--no-zeroshot", action="store_true", help="drop the zero-shot score input (class-free scorer)")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cache = Path(args.cache)
    if cache.exists():
        feats = torch.load(cache, map_location="cpu", weights_only=False)["feats"]
        print(f"loaded {len(feats)} videos from {cache}")
    else:
        feats = cache_features(args.root, cache, device, args.thumb)

    context = args.context if args.kind == "context" else 0
    use_zs = not args.no_zeroshot
    vids = sorted(feats)
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(vids))
    folds = {vids[i]: int(k % args.folds) for k, i in enumerate(order)}

    xs, ys, vs = {}, {}, {}
    for v in vids:
        f = feats[v]
        gaps = torch.cat([f["times"][1:] - f["times"][:-1], torch.tensor([5.0])])
        xs[v] = build_features(f["emb"], gaps, f["zs"] if use_zs else None, context)
        ys[v] = f["target"]
        vs[v] = v
    in_dim = next(iter(xs.values())).shape[1]

    # zero-shot baseline, same metrics
    pred_zs = {v: feats[v]["zs"].numpy() for v in vids}
    sp = np.nanmean([spearman(pred_zs[v], ys[v].numpy()) for v in vids])
    p2 = np.nanmean([top_k_precision(pred_zs[v], ys[v].numpy(), 2) for v in vids])
    ch = np.mean([ys[v].numpy()[np.argsort(-pred_zs[v], kind="stable")[:2]].mean() for v in vids])
    print(f"zero-shot CLIP   spearman={sp:.3f}  top2_prec={p2:.3f}  chosen_overlap={ch:.3f}")
    orc = np.mean([np.sort(ys[v].numpy())[::-1][:2].mean() for v in vids])
    print(f"oracle           chosen_overlap={orc:.3f}  (ceiling of the top-2 windows)")

    heads, per_fold = [], []
    for k in range(args.folds):
        tr = [v for v in vids if folds[v] != k]
        va = [v for v in vids if folds[v] == k]
        tx = torch.cat([xs[v] for v in tr]); ty = torch.cat([ys[v] for v in tr])
        tv = np.concatenate([[i] * len(ys[v]) for i, v in enumerate(tr)])
        vx = torch.cat([xs[v] for v in va]); vy = torch.cat([ys[v] for v in va])
        vv = np.concatenate([[v] * len(ys[v]) for v in va])
        head, ep = train_fold(tx, ty, tv, vx, vy, vv, args.kind, args.hidden, device, args.epochs, args.lr, args.wd,
                              args.seed + k)
        m = evaluate(head, vx, vy, vv, device)
        per_fold.append(m)
        heads.append({kk: v.cpu() for kk, v in head.state_dict().items()})
        print(f"fold {k}: best epoch {ep:3d}  spearman={m['spearman']:.3f}  top2_prec={m['top2_prec']:.3f}  "
              f"top4_prec={m['top4_prec']:.3f}  chosen_overlap={m['chosen_overlap']:.3f}", flush=True)
    mean = {k: float(np.mean([m[k] for m in per_fold])) for k in per_fold[0]}
    std = {k: float(np.std([m[k] for m in per_fold])) for k in per_fold[0]}
    print(f"{args.kind:8s} zs={use_zs} ctx={context}  " + "  ".join(f"{k}={mean[k]:.3f}±{std[k]:.3f}" for k in mean))

    out = Path(args.out or f"results/cache/scorer_{args.kind}{'' if use_zs else '_nozs'}.pt")
    torch.save({"config": {"kind": args.kind, "context": context, "use_zeroshot": use_zs, "in_dim": in_dim,
                           "hidden": args.hidden}, "folds": folds, "heads": heads, "cv": per_fold}, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
