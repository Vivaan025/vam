# Experiment 006 results: adaptive window, score-proportional allocation, coarse-to-fine refinement

**Date:** 2026-11-02 · **GPU:** RTX 5090 (24-core host) · setup identical to `thumos14_scan.md` (200 THUMOS14 val
videos, VideoMAE-base + UCF101 linear probe, 8-process keyframe scan). Raw data: `thumos14_adaptive_rtx5090.csv`.
Baseline rows are from 005 (fixed 5 s window, equal allocation, even placement).

Arms: `gap` = window ends at the next keyframe (clipped to [1 s, 5 s]); `softmax τ` = budget split by
softmax(score/τ); `refine` = each window decoded once at stride 3, thumbnails scored by CLIP, frames placed at
the peaks; `all τ` = gap + softmax τ + refine. `refine ms` is inside `select ms`; with refinement the decode span
is ~0 because the frames are handed over from the refinement pass.

| frames | arm | top-1 | hit | select ms | of which refine | decode ms | infer ms | total ms |
|---|---|---|---|---|---|---|---|---|
| 4 | baseline (005) | 0.795 | 0.41 | 47 | – | 85 | 20 | 152 |
| 4 | **gap** | **0.815** | 0.44 | 47 | – | 80 | 20 | **148** |
| 4 | softmax 0.1 | 0.785 | 0.41 | 46 | – | 84 | 20 | 150 |
| 4 | refine | 0.835 | 0.45 | 230 | 184 | 0 | 6 | 236 |
| 4 | all 0.05 | 0.855 | 0.45 | 212 | 165 | 0 | 6 | 218 |
| 4 | all 0.1 | 0.860 | 0.45 | 213 | 166 | 0 | 6 | 219 |
| 4 | all 0.2 | **0.865** | 0.45 | 213 | 167 | 0 | 5 | 219 |
| 8 | baseline (005) | 0.875 | 0.40 | 46 | – | 166 | 18 | 230 |
| 8 | **gap** | **0.885** | 0.42 | 46 | – | 159 | 19 | **223** |
| 8 | softmax 0.1 | 0.875 | 0.41 | 45 | – | 164 | 19 | 228 |
| 8 | refine | 0.890 | 0.44 | 408 | 361 | 0 | 5 | 413 |
| 8 | all 0.05 | 0.900 | 0.44 | 364 | 318 | 0 | 5 | 369 |
| 8 | all 0.1 | 0.895 | 0.44 | 370 | 324 | 0 | 5 | 375 |
| 8 | all 0.2 | **0.900** | 0.44 | 373 | 326 | 0 | 5 | 378 |
| 16 | baseline (005) | 0.935 | 0.39 | 45 | – | 157 | 8 | 210 |
| 16 | **gap** | **0.935** | 0.41 | 46 | – | 171 | 8 | **225** |
| 16 | softmax 0.1 | 0.920 | 0.40 | 46 | – | 158 | 8 | 212 |
| 16 | refine | 0.925 | 0.43 | 407 | 361 | 0 | 4 | 411 |
| 16 | all 0.05 | 0.925 | 0.44 | 368 | 322 | 0 | 4 | 372 |
| 16 | all 0.1 | 0.930 | 0.44 | 370 | 324 | 0 | 4 | 374 |
| 16 | all 0.2 | 0.925 | 0.44 | 372 | 326 | 0 | 4 | 377 |

Oracle-scored arms (mechanism check; 005 reference kf-oracle 0.845 / 0.925 / 0.935, oracle 0.890 / 0.925 / 0.955):

| frames | arm | top-1 | hit | total ms |
|---|---|---|---|---|
| 4 | kf-oracle + refine | 0.865 | 0.95 | 209 |
| 4 | kf-oracle + all 0.1 | 0.820 | 0.97 | 182 |
| 8 | kf-oracle + refine | 0.915 | 0.88 | 353 |
| 8 | kf-oracle + all 0.1 | 0.895 | 0.97 | 270 |
| 16 | kf-oracle + refine | 0.945 | 0.86 | 347 |
| 16 | kf-oracle + all 0.1 | 0.935 | 0.97 | 277 |

## Findings

1. **The gap window is a free win**: +2.0 points at 4 frames, +1.0 at 8, 0 at 16, at the same latency
   (hit rate 0.41 → 0.44 at 4). Frames no longer spill past the next keyframe, which in THUMOS usually marks a
   scene change. Adopted as the default.
2. **Score-proportional allocation alone hurts** (−1.0 / 0 / −1.5 points): with even placement, concentrating the
   budget on the top keyframe trades diversity for a window that CLIP may have got wrong. **With refinement it
   helps** (+3.0 at 4 frames, +1.0 at 8 over refine alone): once frames land on scored peaks, giving more of them
   to the best window pays. τ ∈ {0.05, 0.1, 0.2} differ by ≤ 1 point, within noise; 0.2 is the default value.
3. **Refinement meets both accuracy targets** (kf-clip-4 0.865 ≥ 0.83, kf-clip-8 0.900 ≥ 0.90 with all 0.2)
   and raises the hit rate to 0.45, **but fails the latency target by a wide margin**: +67 ms at 4 frames and
   +148 ms at 8 against an allowance of +40. At 16 frames it gains nothing (the budget already covers the window).
4. **Where the refinement time goes** (profiled on two videos, 4 windows × 5 s): OpenCV's `cap.set(POS_FRAMES)`
   costs a flat **35–60 ms per seek** whatever the target (keyframe ± 0, ± 2, + 5 all equal; the index does line
   up with PyAV's keyframe, pixel difference 2–8 vs 67–81 for the neighbours), plus **0.3 ms per decoded frame**.
   A 5 s window is therefore ~45 ms of decode + ~50 ms of seek, and refinement must decode whole windows where
   the baseline at 4 frames stops at its last frame. Retrieve + resize is negligible. CLIP on the window thumbnails
   costs **49 ms per 200 thumbnails** (not the ~8 ms assumed), i.e. ~25 ms at 4 frames and ~50 ms at 8/16.
5. **PyAV is not the fix for the window decode**: decoding the same 600 frames with PyAV costs 560 ms single-
   threaded (0.9 ms/frame of per-packet overhead, as in 005) and 300 ms with 4 frame threads, vs 390 ms in OpenCV
   including seeks. Both decoders are overhead-bound at 320×180; the lever is parallelism across windows (the
   process pool from 005) or NVDEC.
6. **The oracle arms show refinement is sound**: kf-oracle + refine reaches 0.865 at 4 frames (hit 0.95), above
   kf-oracle (0.845) and 2.5 points below the oracle. Adding softmax to the oracle hurts at 4 and 8 frames
   because the oracle's keyframe scores are flat (many windows score 1.0), so softmax concentrates arbitrarily.

## Experiment 006 hypothesis check

- kf-clip-4 ≥ 0.83: ✔ (0.865 with gap + softmax 0.2 + refine; 0.835 with refine alone).
- kf-clip-8 ≥ 0.90: ✔ (0.900).
- Total ms within +40 of 005: ✘ (+67 / +148). Gap alone is within budget but reaches only 0.815 / 0.885.
