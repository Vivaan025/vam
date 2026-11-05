"""A sampler that takes a latency budget in milliseconds (016; the scheduler of 010 behind one class).

    sampler = vam.BudgetedSampler(150, model, calibrate_on=[clip_a, clip_b, clip_c])
    out = model(video, sampler=sampler)       # frames, refinement and stride chosen for this video
    sampler.last_plan                          # what it chose and why

Calibration runs the full pipeline (keyframe scan, window decode, the model) on the given clips under a few
configurations, fits the linear cost model and caches it per machine (`~/.cache/vam/`). Per video the scheduler
picks the most accurate configuration whose predicted cost fits before the scan, and re-plans once after it.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence, Union

from vam.core import FrameBatch, Video, load_video
from vam.runtime.scheduler import Config, CostModel, Features, Plan, Scheduler, calibrate, features_for
from vam.sampling.base import Sampler
from vam.sampling.keyframe import KeyframeSampler


class BudgetedSampler(Sampler):
    def __init__(self, budget_ms: float, model=None, calibrate_on: Optional[Sequence[Union[str, Video]]] = None,
                 scorer=None, cost: Optional[CostModel] = None, margin: float = 1.0, cache: bool = True,
                 repeats: int = 2, accuracy: Optional[dict] = None, **keyframe_kwargs):
        super().__init__(1)  # the real frame count is decided per video
        if scorer is None:
            from vam.sampling.learned import default_scorer

            scorer = default_scorer()
        self.scorer = scorer
        self.model = model
        self.kw = keyframe_kwargs
        if cost is None:
            if not calibrate_on:
                raise ValueError("BudgetedSampler needs `cost` (a fitted CostModel) or `calibrate_on` (a few clips)")
            cost = calibrate(list(calibrate_on), self._calib_run, cache=cache, repeats=repeats)
        self.cost = cost
        self.scheduler = Scheduler(budget_ms, cost, accuracy=accuracy, margin=margin)
        self.last: Optional[KeyframeSampler] = None
        self.last_plan: Optional[Plan] = None

    def _keyframe_sampler(self, cfg: Config, planner=None) -> KeyframeSampler:
        kw = dict(segments=cfg.segments, alloc="softmax", temperature=0.2,
                  refine=self.scorer if (cfg.refine or planner is not None) else None,
                  refine_stride=cfg.stride, planner=planner)
        kw.update(self.kw)
        return KeyframeSampler(cfg.frames, self.scorer, **kw)

    def _calib_run(self, cfg: Config, clip) -> tuple[Features, float]:
        v = load_video(clip) if isinstance(clip, str) else clip
        try:
            t0 = time.perf_counter()
            ks = self._keyframe_sampler(cfg)
            idx = ks.select(v)
            batch = ks.read(v, idx)
            if self.model is not None:
                self.model(batch)
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except Exception:
                    pass
            ms = (time.perf_counter() - t0) * 1000.0
            K = len(ks.last_keyframes.times) if ks.last_keyframes is not None else 0
            wins = [e - s for s, e in ks.last_windows]
            return features_for(cfg, K, wins), ms
        finally:
            if isinstance(clip, str):
                v.close()

    def select(self, video: Video) -> list[int]:
        pre = self.scheduler.pre_plan(video.duration, video.fps)
        self.last = self._keyframe_sampler(pre.config, planner=self.scheduler.planner())
        idx = self.last.select(video)
        self.last_plan = self.scheduler.last_plan or pre
        self.num_frames = len(idx)
        return idx

    def read(self, video: Video, indices: list[int]) -> FrameBatch:
        return self.last.read(video, indices) if self.last is not None else video.read(indices)

    def __repr__(self) -> str:
        return f"BudgetedSampler(budget_ms={self.scheduler.budget:.0f}, cost={self.cost.machine or 'given'})"
