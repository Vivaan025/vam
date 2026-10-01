"""Measure what actually costs time and memory. Every claim in the paper comes from here."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class Span:
    name: str
    wall_ms: float
    peak_mem_mb: float


@dataclass
class Profiler:
    spans: list[Span] = field(default_factory=list)

    @contextmanager
    def span(self, name: str):
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0
            mem = torch.cuda.max_memory_allocated() / 2**20 if cuda else 0.0
            self.spans.append(Span(name, ms, mem))

    def report(self) -> str:
        if not self.spans:
            return "(no spans)"
        w = max(len(s.name) for s in self.spans)
        lines = [f"{'stage':<{w}}  {'wall_ms':>9}  {'peak_mb':>8}"]
        for s in self.spans:
            lines.append(f"{s.name:<{w}}  {s.wall_ms:9.1f}  {s.peak_mem_mb:8.1f}")
        total = sum(s.wall_ms for s in self.spans)
        lines.append(f"{'total':<{w}}  {total:9.1f}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {s.name: {"wall_ms": s.wall_ms, "peak_mb": s.peak_mem_mb} for s in self.spans}


@contextmanager
def profile(name: str = "run"):
    """One-off profiling: with vam.profile("infer") as p: ... then print(p.report())."""
    p = Profiler()
    with p.span(name):
        yield p
