"""Experiment 020: Video-MME multiple choice with one video-LLM, frames chosen by different samplers.

Arms (``--arms``): uniform-N | keyframe-N (packaged class-free scorer, refinement on) | kfnr-N (no refinement)
| hybrid-N (N/2 uniform anchors + N/2 keyframe frames, union) | budget-B (BudgetedSampler, B ms).
One sampling per video, one generation per question.

    python benchmarks/eval_videomme.py --root data/videomme --arms uniform-8 keyframe-8 uniform-16 keyframe-16 --out results/videomme_020.jsonl
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import vam
from vam.data.videomme import VideoMME
from vam.models.videollm import VideoLLM
from vam.sampling import KeyframeSampler, Sampler, UniformSampler


class HybridSampler(Sampler):
    """020 H5: evenly spaced anchors keep the whole video in view, keyframe picks add the moments."""

    def __init__(self, num_frames: int, scorer):
        super().__init__(num_frames)
        self.uniform = UniformSampler(max(1, num_frames // 2))
        self.keyframe = KeyframeSampler(max(1, num_frames - num_frames // 2), scorer, refine=None)
        self.last_fallback = None

    def select(self, video):
        u = self.uniform.select(video)
        k = self.keyframe.select(video)
        self.last_fallback = self.keyframe.last_fallback
        return sorted(set(u) | set(k))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/videomme")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--max-edge", type=int, default=448)
    ap.add_argument("--arms", nargs="+", default=["uniform-8", "keyframe-8", "uniform-16", "keyframe-16"])
    ap.add_argument("--scorer-head", default=None)
    ap.add_argument("--buckets", nargs="+", default=["short", "medium", "long"])
    ap.add_argument("--limit", type=int, default=None, help="videos per bucket")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/videomme_020.jsonl")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    ds = VideoMME(args.root)
    videos = []
    for b in args.buckets:
        vs = [v for v in ds.videos if v.duration == b]
        videos += vs[: args.limit] if args.limit else vs
    llm = VideoLLM(args.model, device=device, max_edge=args.max_edge)
    print(gpu, "|", ds, "| running", len(videos), "videos |", args.model, flush=True)

    if args.scorer_head:
        from vam.sampling import ClipScorer, LearnedScorer

        scorer = LearnedScorer.load(args.scorer_head, ClipScorer((), device=device))
    else:
        from vam.sampling.learned import default_scorer

        scorer = default_scorer(device=device)

    def make(arm: str):
        kind, val = arm.rsplit("-", 1)
        if kind == "uniform":
            return UniformSampler(int(val))
        if kind == "keyframe":
            return KeyframeSampler(int(val), scorer, refine=scorer)
        if kind == "kfnr":
            return KeyframeSampler(int(val), scorer, refine=None)
        if kind == "hybrid":
            return HybridSampler(int(val), scorer)
        if kind == "budget":
            return vam.BudgetedSampler(float(val), None, calibrate_on=[v.path for v in videos[:3]], scorer=scorer)
        raise ValueError(arm)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for arm in args.arms:
            sampler = make(arm)
            correct = defaultdict(list)
            ms = defaultdict(list)
            n_fb = 0
            t_arm = time.time()
            for mv in videos:
                try:
                    v = vam.load_video(mv.path)
                except Exception as e:  # noqa: BLE001
                    print(f"skip {mv.youtube_id}: {e}", flush=True)
                    continue
                p = vam.Profiler()
                try:
                    with p.span("select"):
                        idx = sampler.select(v)
                    with p.span("decode"):
                        batch = sampler.read(v, idx)
                    fps, nf, dur = v.fps, v.num_frames, v.duration
                except Exception as e:  # noqa: BLE001
                    print(f"skip {mv.youtube_id}: {type(e).__name__}: {e}", flush=True)
                    v.close()
                    continue
                finally:
                    v.close()
                ks = sampler.last if isinstance(sampler, vam.BudgetedSampler) else sampler
                fb = getattr(ks, "last_fallback", None)
                n_fb += fb is not None
                d = p.as_dict()
                sel_ms, dec_ms = d["select"]["wall_ms"], d["decode"]["wall_ms"]
                for q in mv.questions:
                    t0 = time.perf_counter()
                    try:
                        pred = llm.answer(batch, q.question, q.options, fps=fps, num_frames=nf, duration=dur)
                    except Exception as e:  # noqa: BLE001
                        print(f"skip {q.question_id}: {type(e).__name__}: {e}", flush=True)
                        continue
                    if device == "cuda":
                        torch.cuda.synchronize()
                    gen_ms = (time.perf_counter() - t0) * 1000
                    ok = pred == q.answer
                    correct[mv.duration].append(ok)
                    correct[("task", q.task_type)].append(ok)
                    ms["select"].append(sel_ms)
                    ms["decode"].append(dec_ms)
                    ms["generate"].append(gen_ms)
                    f.write(json.dumps({
                        "ts": time.time(), "host": platform.node(), "gpu": gpu, "model": args.model, "arm": arm,
                        "video_id": mv.video_id, "youtube_id": mv.youtube_id, "bucket": mv.duration, "domain": mv.domain,
                        "question_id": q.question_id, "task_type": q.task_type, "answer": q.answer, "pred": pred,
                        "correct": ok, "frames": len(idx), "fallback": fb, "select_ms": sel_ms, "decode_ms": dec_ms,
                        "generate_ms": gen_ms, "prompt_tokens": llm.last_tokens, "reply": llm.last_text,
                    }) + "\n")
                f.flush()
            summ = {b: float(np.mean(correct[b])) for b in args.buckets if correct[b]}
            allc = [x for b in args.buckets for x in correct[b]]
            tasks = {k[1]: float(np.mean(vv)) for k, vv in correct.items() if isinstance(k, tuple)}
            line = (f"{arm:12s} n={len(allc):5d} acc={np.mean(allc) if allc else float('nan'):.3f}  " +
                    " ".join(f"{b}={summ[b]:.3f}" for b in summ) +
                    f"  select p50={np.median(ms['select']):.0f} decode p50={np.median(ms['decode']):.0f} gen p50={np.median(ms['generate']):.0f} ms"
                    f"  fallback={n_fb}  ({(time.time() - t_arm) / max(len(allc), 1) * 1000:.0f} ms/question wall)")
            print(line, flush=True)
            f.write(json.dumps({"summary": True, "arm": arm, "model": args.model, "n": len(allc), "acc": float(np.mean(allc)) if allc else None,
                                "buckets": summ, "tasks": tasks, "select_p50": float(np.median(ms["select"])) if ms["select"] else None,
                                "decode_p50": float(np.median(ms["decode"])) if ms["decode"] else None,
                                "generate_p50": float(np.median(ms["generate"])) if ms["generate"] else None,
                                "fallback": n_fb, "gpu": gpu}) + "\n")
            f.flush()
    vam.runtime.shutdown_pools()


if __name__ == "__main__":
    main()
