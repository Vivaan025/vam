"""Two-stage keyframe sampler: scan keyframes cheaply, score them, decode windows near the best.

Stage 1  scan_keyframes()  -> thumbnails + timestamps (~100 ms for a 3-minute file)
Stage 2  scorer(thumbs)    -> one score per keyframe (semantic: CLIP, oracle, learned ...)
Stage 3  pick `segments` keyframes by score with temporal non-max suppression, then place the
         frame budget evenly inside a window of `window_s` seconds *after* each chosen keyframe.
         Decoding forward from a keyframe needs no seek, so stage 3 costs one short sequential
         read per segment instead of one seek per frame.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from vam.core import Video
from vam.runtime.scan import Keyframes, scan_keyframes
from vam.sampling.base import Sampler

Scorer = Callable[[Keyframes], np.ndarray]


class KeyframeSampler(Sampler):
    def __init__(self, num_frames: int, scorer: Scorer, window_s: float = 5.0,
                 segments: Optional[int] = None, thumb: tuple[int, int] = (112, 112)):
        super().__init__(num_frames)
        self.scorer = scorer
        self.window_s = window_s
        self.segments = segments or max(1, min(4, num_frames // 2))
        self.thumb = thumb
        self.last_keyframes: Optional[Keyframes] = None
        self.last_scores: Optional[np.ndarray] = None
        self.last_chosen: list[int] = []

    def pick_keyframes(self, kf: Keyframes, scores: np.ndarray) -> list[int]:
        """Greedy top-k with temporal NMS: no two chosen keyframes within window_s."""
        order = np.argsort(-scores, kind="stable")
        chosen: list[int] = []
        for i in order:
            t = kf.times[i]
            if all(abs(t - kf.times[j]) >= self.window_s for j in chosen):
                chosen.append(int(i))
            if len(chosen) >= self.segments:
                break
        return sorted(chosen, key=lambda i: kf.times[i])

    def select(self, video: Video) -> list[int]:
        kf = scan_keyframes(video.path, thumb=self.thumb)
        scores = np.asarray(self.scorer(kf), dtype=np.float64)
        if scores.shape != (len(kf),):
            raise ValueError(f"scorer returned {scores.shape}, expected ({len(kf)},)")
        chosen = self.pick_keyframes(kf, scores)
        self.last_keyframes, self.last_scores, self.last_chosen = kf, scores, chosen

        # split the budget across chosen segments, remainder to the best-scored ones
        m = len(chosen)
        base, extra = divmod(self.num_frames, m)
        by_score = sorted(chosen, key=lambda i: -scores[i])
        per = {i: base + (1 if r < extra else 0) for r, i in enumerate(by_score)}

        fps = video.fps
        n = video.num_frames
        out: list[int] = []
        for i in chosen:
            k = per[i]
            if k <= 0:
                continue
            start = int(kf.indices[i])
            span = max(1, int(round(self.window_s * fps)))
            end = min(start + span, n)
            offs = (np.arange(k) + 0.5) / k * (end - start)
            out.extend(int(start + o) for o in offs)
        return sorted(set(min(i, n - 1) for i in out))


# ---- scorers -------------------------------------------------------------------------------

def uniform_scores(kf: Keyframes) -> np.ndarray:
    """Baseline scorer: all keyframes equal, so NMS picks the first ones. For ablation only."""
    return np.zeros(len(kf))


class OracleKeyframeScorer:
    """Scores a keyframe by how much of the window after it overlaps annotated action.

    Not a usable method: it isolates the *runtime* (scan + windowed decode) from the scorer.
    If this reaches near-oracle accuracy, the pipeline works and only the scorer is left.
    """

    def __init__(self, segments: list[tuple[float, float]], window_s: float = 5.0):
        self.segments = segments
        self.window_s = window_s

    def __call__(self, kf: Keyframes) -> np.ndarray:
        s = np.zeros(len(kf))
        for i, t in enumerate(kf.times):
            a, b = t, t + self.window_s
            s[i] = sum(max(0.0, min(b, e) - max(a, st)) for st, e in self.segments) / self.window_s
        return s


class ClipScorer:
    """Zero-shot CLIP: how strongly a keyframe matches any target class versus background prompts.

    score = 1 - P(background) under a softmax over class prompts + background prompts.
    ~5 ms per keyframe on a GPU with ViT-B/32; text embeddings are computed once.
    """

    BACKGROUND = ["a crowd of spectators", "a scoreboard", "an empty stadium", "a television studio",
                  "a blank screen", "a logo", "people talking", "a commercial break"]

    def __init__(self, class_names: list[str], model: str = "openai/clip-vit-base-patch32",
                 device: str | None = None, template: str = "a photo of {}"):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model).eval().to(self.device)
        self.proc = CLIPProcessor.from_pretrained(model)
        names = [self._human(c) for c in class_names]
        prompts = [template.format(n) for n in names] + [template.format(b) for b in self.BACKGROUND]
        self.n_cls = len(names)
        with torch.inference_mode():
            t = self.proc(text=prompts, return_tensors="pt", padding=True).to(self.device)
            # text_model + projection is stable across transformers 4.x and 5.x
            # (get_text_features returns a tensor in 4.x and an output object in 5.x)
            pooled = self.model.text_model(**t).pooler_output
            e = self.model.text_projection(pooled)
            self.text = e / e.norm(dim=-1, keepdim=True)

    @staticmethod
    def _human(name: str) -> str:
        import re
        return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).lower()

    def __call__(self, kf: Keyframes) -> np.ndarray:
        import torch

        with torch.inference_mode():
            px = self.proc(images=list(kf.thumbs), return_tensors="pt")["pixel_values"].to(self.device)
            pooled = self.model.vision_model(pixel_values=px).pooler_output
            e = self.model.visual_projection(pooled)
            e = e / e.norm(dim=-1, keepdim=True)
            logits = 100.0 * e @ self.text.T
            p = logits.softmax(dim=-1)
            return (1.0 - p[:, self.n_cls:].sum(dim=-1)).float().cpu().numpy()
