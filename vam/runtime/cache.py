"""Feature cache: reuse per-frame features when the scene barely changed.

v0.1 is a correct, simple LRU keyed on frame index with a memory budget.
Research direction: similarity-keyed reuse, GPU-resident vs CPU-offloaded tiers,
and eviction policies tuned to action-prediction error rather than recency.
"""

from __future__ import annotations

from collections import OrderedDict

import torch


class FeatureCache:
    def __init__(self, budget_mb: float = 256.0):
        self.budget_bytes = int(budget_mb * 2**20)
        self._store: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    def get(self, key: int) -> torch.Tensor | None:
        t = self._store.get(key)
        if t is None:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return t

    def put(self, key: int, value: torch.Tensor) -> None:
        if key in self._store:
            self._bytes -= self._nbytes(self._store[key])
        self._store[key] = value
        self._store.move_to_end(key)
        self._bytes += self._nbytes(value)
        while self._bytes > self.budget_bytes and self._store:
            _, old = self._store.popitem(last=False)
            self._bytes -= self._nbytes(old)

    def __len__(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0

    def stats(self) -> dict[str, float]:
        return {"entries": len(self), "bytes": self._bytes, "hit_rate": self.hit_rate}
