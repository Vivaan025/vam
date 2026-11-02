"""Learned keyframe scorer (experiment 009): a small head on frozen CLIP image embeddings, trained to
predict the oracle window-overlap score of a keyframe from its embedding and its neighbours'.

Needs no class names at inference. The same head scores refinement candidates (they are thumbnails too).
Trained with scripts/train_scorer.py; heads are stored per cross-validation fold together with the fold
assignment, so a benchmark video is always scored by a head that never saw it. Unknown videos get the
average of all folds' heads.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn

from vam.runtime.scan import Keyframes


def gap_window_targets(times: np.ndarray, segments: list[tuple[float, float]],
                       min_window_s: float = 1.0, window_s: float = 5.0) -> np.ndarray:
    """Oracle score of each keyframe under the gap-window rule: overlap of [t, t + clip(gap, min, max))
    with annotated action, as a fraction of the window."""
    t = np.asarray(times, dtype=np.float64)
    gaps = np.append(np.diff(t), window_s)
    spans = np.clip(gaps, min_window_s, window_s)
    out = np.zeros(len(t))
    for i, (a, span) in enumerate(zip(t, spans)):
        b = a + span
        out[i] = sum(max(0.0, min(b, e) - max(a, st)) for st, e in segments) / span
    return out


def build_features(emb: torch.Tensor, gaps: torch.Tensor, zs: Optional[torch.Tensor], context: int) -> torch.Tensor:
    """(K, D) embeddings -> (K, F): own embedding, +-context neighbours (zero padded), gap, [zero-shot score]."""
    k, d = emb.shape
    parts = []
    for off in range(-context, context + 1):
        if off == 0:
            parts.append(emb)
        else:
            shifted = torch.zeros_like(emb)
            if off < 0:
                shifted[-off:] = emb[:k + off] if k + off > 0 else emb[:0]
            else:
                shifted[:k - off] = emb[off:] if k - off > 0 else emb[:0]
            parts.append(shifted)
    parts.append((gaps.float() / 5.0).clamp(0, 2).unsqueeze(1))
    if zs is not None:
        parts.append(zs.float().unsqueeze(1))
    return torch.cat(parts, dim=1)


class ScorerHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, kind: str = "mlp"):
        super().__init__()
        self.kind = kind
        if kind == "ridge":
            self.net = nn.Linear(in_dim, 1)
        else:
            self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LearnedScorer:
    """scorer(kf) -> (K,) scores. `embedder` is a ClipScorer (its `embed` and, optionally, zero-shot scores)."""

    def __init__(self, embedder, checkpoint: dict, device: str | None = None):
        self.embedder = embedder
        self.device = device or embedder.device
        cfg = checkpoint["config"]
        self.context = int(cfg["context"])
        self.use_zs = bool(cfg["use_zeroshot"])
        self.folds: dict[str, int] = checkpoint.get("folds", {})
        self.heads: list[ScorerHead] = []
        for sd in checkpoint["heads"]:
            h = ScorerHead(int(cfg["in_dim"]), int(cfg.get("hidden", 256)), cfg["kind"]).to(self.device)
            h.load_state_dict(sd)
            h.eval()
            self.heads.append(h)
        self.video_id: Optional[str] = None

    @classmethod
    def load(cls, path: str, embedder, device: str | None = None) -> "LearnedScorer":
        return cls(embedder, torch.load(path, map_location="cpu", weights_only=False), device)

    def for_video(self, video_id: Optional[str]) -> "LearnedScorer":
        """Select the fold head that did not train on this video (benchmark use)."""
        self.video_id = video_id
        return self

    def _heads(self) -> list[ScorerHead]:
        if self.video_id is not None and self.video_id in self.folds:
            return [self.heads[self.folds[self.video_id]]]
        return self.heads

    @torch.inference_mode()
    def __call__(self, kf: Keyframes) -> np.ndarray:
        e = self.embedder.embed(kf.thumbs)
        gaps = torch.as_tensor(np.append(np.diff(kf.times), 5.0), device=e.device)
        zs = torch.as_tensor(self.embedder.scores_from_embedding(e), device=e.device) if self.use_zs else None
        x = build_features(e, gaps, zs, self.context)
        out = torch.stack([h(x) for h in self._heads()]).mean(0)
        return out.float().cpu().numpy()
