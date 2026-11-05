"""Experiment 014: a policy (vision-language-action model) through the same runtime as recognition models.

Runs an `ActionPolicy` over `vam.Stream(window=1, hop=k)` on one or more clips, logs per-step latency and the
7-D action, and, when the clip comes from a LeRobot-format dataset (parquet next to the video), compares the
predicted actions with the recorded ones: per-dimension Pearson correlation, sign agreement of the translation
deltas, and gripper open/close agreement. Also calibrates the scheduler's cost model with the policy as the
model, to show the cost model is model-agnostic.

    python benchmarks/eval_policy.py --model openvla-7b --unnorm-key bridge_orig \\
        --clip data/bridge/videos/chunk-000/observation.images.image_0/episode_000000.mp4 \\
        --clip data/charades/Charades_v1_480/<id>.mp4 --instruction "put small spoon from basket to tray"
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import time
from pathlib import Path

import numpy as np
import torch

import vam
from vam.models import get_model


def lerobot_truth(clip: Path):
    """If `clip` is <root>/videos/chunk-XXX/<camera>/episode_N.mp4, return (actions (T,7), task) from the parquet."""
    m = re.search(r"videos[\\/](chunk-\d+)[\\/][^\\/]+[\\/](episode_\d+)\.mp4$", str(clip))
    if not m:
        return None, None
    root = Path(str(clip)[: m.start()])
    pq_path = root / "data" / m.group(1) / f"{m.group(2)}.parquet"
    if not pq_path.exists():
        return None, None
    import pyarrow.parquet as pq

    t = pq.read_table(str(pq_path))
    actions = np.stack(t.column("action").to_pylist()).astype(np.float32)
    task = None
    tasks = root / "meta" / "tasks.jsonl"
    if tasks.exists():
        ti = int(t.column("task_index")[0].as_py())
        for line in tasks.read_text().splitlines():
            d = json.loads(line)
            if d.get("task_index") == ti:
                task = d["task"]
    return actions, task


def agreement(pred: np.ndarray, truth: np.ndarray) -> dict:
    n = min(len(pred), len(truth))
    p, t = pred[:n], truth[:n]
    out = {"n": n}
    corr = []
    for d in range(min(p.shape[1], t.shape[1])):
        if p[:, d].std() > 0 and t[:, d].std() > 0:
            corr.append(float(np.corrcoef(p[:, d], t[:, d])[0, 1]))
        else:
            corr.append(float("nan"))
    out["corr_per_dim"] = corr
    out["sign_agree_xyz"] = float(np.mean(np.sign(p[:, :3]) == np.sign(t[:, :3])))
    g_p = (p[:, -1] > 0.5).astype(int)
    g_t = (t[:, -1] > 0.5).astype(int)
    out["gripper_agree"] = float(np.mean(g_p == g_t))
    out["gripper_flips_pred"] = int(np.abs(np.diff(g_p)).sum())
    out["gripper_flips_true"] = int(np.abs(np.diff(g_t)).sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openvla-7b")
    ap.add_argument("--unnorm-key", default="bridge_orig")
    ap.add_argument("--clip", action="append", required=True)
    ap.add_argument("--instruction", default=None, help="default: the LeRobot task of the clip, else 'pick up the object'")
    ap.add_argument("--hop", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/policy.jsonl")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    kw = {"device": device}
    if args.model == "openvla-7b":
        kw["unnorm_key"] = args.unnorm_key
    t0 = time.time()
    policy = get_model(args.model, **kw)
    print(f"{gpu} | {args.model} loaded in {time.time() - t0:.0f} s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for clip in args.clip:
            clip_p = Path(clip)
            truth, task = lerobot_truth(clip_p)
            instruction = args.instruction or task or "pick up the object"
            policy.instruction = instruction
            preds, lat = [], []
            for k, batch in enumerate(vam.Stream(str(clip_p), window=1, hop=args.hop)):
                if k >= args.max_steps:
                    break
                t1 = time.perf_counter()
                o = policy(batch)
                if device == "cuda":
                    torch.cuda.synchronize()
                lat.append((time.perf_counter() - t1) * 1000)
                preds.append(o.actions[0].numpy())
            preds = np.stack(preds)
            rec = {
                "ts": time.time(), "host": platform.node(), "gpu": gpu, "model": args.model, "clip": str(clip_p),
                "instruction": instruction, "steps": len(preds), "hop": args.hop,
                "ms_per_step_p50": float(np.median(lat)), "ms_per_step_p90": float(np.percentile(lat, 90)),
                "actions_mean": preds.mean(0).round(4).tolist(), "actions_std": preds.std(0).round(4).tolist(),
                "gripper_flips_pred": int(np.abs(np.diff((preds[:, -1] > 0.5).astype(int))).sum()),
                "peak_mb": float(torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else 0.0,
            }
            if truth is not None:
                rec["agreement"] = agreement(preds, truth[:: args.hop])
            f.write(json.dumps(rec) + "\n")
            f.flush()
            msg = (f"{clip_p.name:28s} steps={len(preds):3d}  {rec['ms_per_step_p50']:.0f} ms/step (p90 {rec['ms_per_step_p90']:.0f})  "
                   f"instruction={instruction!r}")
            if truth is not None:
                a = rec["agreement"]
                msg += (f"\n    vs recorded: corr xyz={np.round(a['corr_per_dim'][:3], 2).tolist()} "
                        f"sign-agree xyz={a['sign_agree_xyz']:.2f} gripper-agree={a['gripper_agree']:.2f} "
                        f"flips pred/true={a['gripper_flips_pred']}/{a['gripper_flips_true']}")
            print(msg, flush=True)

    # the cost model does not care what the model is
    from vam.runtime.scheduler import Config, Features, calibrate

    def run(cfg: Config, clip):
        v = vam.load_video(str(clip))
        try:
            t1 = time.perf_counter()
            batch = vam.UniformSampler(cfg.frames)(v)
            policy(batch)
            if device == "cuda":
                torch.cuda.synchronize()
            return Features(0, 0, 0, cfg.frames, 0), (time.perf_counter() - t1) * 1000
        finally:
            v.close()

    cm = calibrate([Path(c) for c in args.clip[:1]], run, configs=[Config(1, False), Config(2, False), Config(4, False)],
                   cache=False, repeats=2)
    print(f"cost model with the policy as the model: {cm.w[0]:.0f} ms + {cm.w[4]:.0f} ms per frame (p75 resid {cm.resid_p75:.0f})",
          flush=True)


if __name__ == "__main__":
    main()
