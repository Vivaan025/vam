"""Two-stage keyframe sampler: scan keyframes cheaply, score them, decode windows near the best.

Stage 1  scan_keyframes()  -> thumbnails + timestamps (~100 ms for a 3-minute file)
Stage 2  scorer(thumbs)    -> one score per keyframe (semantic: CLIP, oracle, learned ...)
Stage 3  pick `segments` keyframes by score with temporal non-max suppression, then place the
         frame budget inside a window *after* each chosen keyframe.
         Decoding forward from a keyframe needs no seek, so stage 3 costs one short sequential
         read per segment instead of one seek per frame.

Stage 3 has three knobs (experiment 006):
  window   "gap" (default): up to the next keyframe, clipped to [min_window_s, window_s].
           "fixed": `window_s` seconds after the keyframe.
  alloc    "equal": budget split evenly, remainder to the best scores.
           "softmax": proportional to softmax(score / temperature).
  refine   None: frames spaced evenly inside the window.
           a scorer: the window is decoded once at thumbnail resolution every `refine_stride`
           frames, scored, and the frames go to the best-scoring moments. The frames decoded for
           refinement are kept, so `read()` hands them over without a second decode.
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
                 segments: Optional[int] = None, thumb: tuple[int, int] = (112, 112),
                 scan_workers: int | None = None, keyframe_stride: int = 1, scan_backend: str | None = None,
                 window: str = "gap", min_window_s: float = 1.0,
                 alloc: str = "equal", temperature: float = 0.2,
                 refine: Optional[Scorer] = None, refine_stride: int = 3, keep_edge: int = 224,
                 decode_backend: str | None = None, decode_workers: int | None = None):
        super().__init__(num_frames)
        if window not in ("fixed", "gap"):
            raise ValueError(f"window must be 'fixed' or 'gap', got {window!r}")
        if alloc not in ("equal", "softmax"):
            raise ValueError(f"alloc must be 'equal' or 'softmax', got {alloc!r}")
        self.scorer = scorer
        self.window_s = window_s
        self.segments = segments or max(1, min(4, num_frames // 2))
        self.thumb = thumb
        self.scan_workers = scan_workers
        self.keyframe_stride = keyframe_stride
        self.scan_backend = scan_backend
        self.window = window
        self.min_window_s = min_window_s
        self.alloc = alloc
        self.temperature = temperature
        self.refine = refine
        self.refine_stride = max(1, int(refine_stride))
        self.keep_edge = keep_edge
        self.decode_backend = decode_backend  # "process" | "thread" | "serial"; None -> platform default
        self.decode_workers = decode_workers
        self.last_keyframes: Optional[Keyframes] = None
        self.last_scores: Optional[np.ndarray] = None
        self.last_chosen: list[int] = []
        self.last_windows: list[tuple[int, int]] = []
        self.last_refine_ms: float = 0.0
        self._cache: dict[int, np.ndarray] = {}

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

    def allocate(self, chosen: list[int], scores: np.ndarray) -> dict[int, int]:
        """Frames per chosen keyframe, summing to the budget."""
        n = self.num_frames
        if self.alloc == "equal" or len(chosen) == 1:
            base, extra = divmod(n, len(chosen))
            by_score = sorted(chosen, key=lambda i: -scores[i])
            return {i: base + (1 if r < extra else 0) for r, i in enumerate(by_score)}
        z = np.array([scores[i] for i in chosen], dtype=np.float64) / max(self.temperature, 1e-6)
        p = np.exp(z - z.max())
        p /= p.sum()
        k = np.floor(p * n).astype(int)
        for i in np.argsort(-(p * n - k), kind="stable")[: n - int(k.sum())]:
            k[i] += 1
        return {i: int(c) for i, c in zip(chosen, k)}

    def windows(self, kf: Keyframes, chosen: list[int], video: Video) -> list[tuple[int, int]]:
        """[start, end) frame range decoded after each chosen keyframe."""
        fps, n = video.fps, video.num_frames
        out = []
        for i in chosen:
            start = int(kf.indices[i])
            span_s = self.window_s
            if self.window == "gap" and i + 1 < len(kf):
                gap = float(kf.times[i + 1] - kf.times[i])
                span_s = min(self.window_s, max(self.min_window_s, gap))
            end = min(start + max(1, int(round(span_s * fps))), n)
            out.append((start, end))
        return out

    def select(self, video: Video) -> list[int]:
        kf = scan_keyframes(video.path, thumb=self.thumb, workers=self.scan_workers, stride=self.keyframe_stride,
                            backend=self.scan_backend)
        scores = np.asarray(self.scorer(kf), dtype=np.float64)
        if scores.shape != (len(kf),):
            raise ValueError(f"scorer returned {scores.shape}, expected ({len(kf)},)")
        chosen = self.pick_keyframes(kf, scores)
        per = self.allocate(chosen, scores)
        wins = self.windows(kf, chosen, video)
        self.last_keyframes, self.last_scores, self.last_chosen, self.last_windows = kf, scores, chosen, wins
        self._cache = {}
        self.last_refine_ms = 0.0

        n = video.num_frames
        out: list[int] = []
        if self.refine is None:
            for i, (start, end) in zip(chosen, wins):
                k = per[i]
                if k <= 0:
                    continue
                offs = (np.arange(k) + 0.5) / k * (end - start)
                out.extend(int(start + o) for o in offs)
            return sorted(set(min(i, n - 1) for i in out))

        # coarse-to-fine: decode each window once at stride (in parallel, experiment 007), score the
        # thumbnails, place the window's frames at the peaks
        import time

        from vam.runtime.decode import Window, read_windows

        t0 = time.perf_counter()
        jobs = [(i, Window(start, end, None, self.refine_stride)) for i, (start, end) in zip(chosen, wins) if per[i] > 0]
        parts = read_windows(video.path, [w for _, w in jobs], thumb=self.thumb, keep_edge=self.keep_edge,
                             workers=self.decode_workers, backend=self.decode_backend)
        cand_idx: list[int] = []
        cand_thumb: list[np.ndarray] = []
        groups: list[tuple[int, int, int]] = []  # (chosen keyframe, first candidate, n candidates)
        for (i, _), wf in zip(jobs, parts):
            first = len(cand_idx)
            for j, idx in enumerate(wf.indices.tolist()):
                self._cache[idx] = wf.frames[j]
                cand_idx.append(idx)
                cand_thumb.append(wf.thumbs[j])
            groups.append((i, first, len(cand_idx) - first))
        if not cand_idx:
            return []
        fps = video.fps
        cands = Keyframes(times=np.asarray(cand_idx, dtype=np.float64) / fps, indices=np.asarray(cand_idx),
                          thumbs=np.stack(cand_thumb), fps=fps, duration=video.duration, num_frames=n)
        fine = np.asarray(self.refine(cands), dtype=np.float64)
        self.last_fine = (cands, fine)
        for i, first, m in groups:
            k = min(per[i], m)
            s = fine[first:first + m]
            spacing = max(1, m // (2 * k))  # keep the k picks from piling onto one peak
            picked: list[int] = []
            for j in np.argsort(-s, kind="stable"):
                if all(abs(int(j) - q) >= spacing for q in picked):
                    picked.append(int(j))
                if len(picked) >= k:
                    break
            out.extend(cand_idx[first + j] for j in picked)
        self.last_refine_ms = (time.perf_counter() - t0) * 1000.0
        return sorted(set(out))

    def read(self, video: Video, indices: list[int]):
        """Hand over frames kept by the refinement pass; otherwise decode the windows in parallel,
        each worker reading only up to its last wanted frame (experiment 007)."""
        import torch

        from vam.core import FrameBatch

        idx = sorted(set(int(i) for i in indices))
        if not idx:
            raise ValueError("no frames requested")
        if not all(i in self._cache for i in idx):
            if self.decode_backend == "serial" or not self.last_windows:
                return video.read(idx)
            from vam.runtime.decode import Window, read_windows

            jobs = []
            for start, end in self.last_windows:
                want = [i for i in idx if start <= i < end]
                if want:
                    jobs.append(Window(start, end, want))
            covered = {i for w in jobs for i in w.want}
            if covered != set(idx):
                return video.read(idx)
            for wf in read_windows(video.path, jobs, thumb=None, keep_edge=self.keep_edge,
                                   workers=self.decode_workers, backend=self.decode_backend):
                for j, i in enumerate(wf.indices.tolist()):
                    self._cache[i] = wf.frames[j]
            idx = [i for i in idx if i in self._cache]
            if not idx:
                raise ValueError("no frames could be decoded for the requested indices")
        arr = torch.from_numpy(np.stack([self._cache[i] for i in idx]))
        idx_t = torch.tensor(idx, dtype=torch.long)
        return FrameBatch(arr, idx_t.float() / float(video.fps), idx_t)


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
                 device: str | None = None, template: str = "a photo of {}", use_hf_processor: bool = False,
                 half: bool = False):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        from vam.runtime.preprocess import Preprocessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model).eval().to(self.device)
        self.half = half and self.device != "cpu"
        if self.half:
            self.model = self.model.half()
        self.proc = CLIPProcessor.from_pretrained(model)
        self.use_hf_processor = use_hf_processor
        self.pre = Preprocessor.from_hf(self.proc.image_processor, device=self.device)
        names = [self._human(c) for c in class_names]
        prompts = [template.format(n) for n in names] + [template.format(b) for b in self.BACKGROUND]
        self.n_cls = len(names)
        with torch.inference_mode():
            t = self.proc(text=prompts, return_tensors="pt", padding=True).to(self.device)
            # text_model + projection is stable across transformers 4.x and 5.x
            # (get_text_features returns a tensor in 4.x and an output object in 5.x)
            pooled = self.model.text_model(**t).pooler_output
            e = self.model.text_projection(pooled).float()
            self.text = e / e.norm(dim=-1, keepdim=True)

    @staticmethod
    def _human(name: str) -> str:
        import re
        return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).lower()

    def __call__(self, kf: Keyframes) -> np.ndarray:
        import torch

        with torch.inference_mode():
            if self.use_hf_processor:
                px = self.proc(images=list(kf.thumbs), return_tensors="pt")["pixel_values"].to(self.device)
            else:
                px = self.pre(kf.thumbs)
            if self.half:
                px = px.half()
            pooled = self.model.vision_model(pixel_values=px).pooler_output
            e = self.model.visual_projection(pooled).float()
            e = e / e.norm(dim=-1, keepdim=True)
            logits = 100.0 * e @ self.text.T
            p = logits.softmax(dim=-1)
            return (1.0 - p[:, self.n_cls:].sum(dim=-1)).float().cpu().numpy()
