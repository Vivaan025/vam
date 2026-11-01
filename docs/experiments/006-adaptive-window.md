# 006 · Adaptive window and coarse-to-fine refinement

**Date:** 2026-11-01 · **Status:** planned · **Results:** *(to fill)*

## Motivation (from 003 and 005)
With the scan at ~46 ms, the gap to the oracle is now the scorer's and the window's:

| frames | kf-clip | kf-oracle | oracle |
|---|---|---|---|
| 4 | 0.795 | 0.845 | 0.890 |
| 8 | 0.875 | 0.925 | 0.925 |
| 16 | 0.935 | 0.935 | 0.955 |

Two separate losses. (a) kf-clip → kf-oracle (5 points at 4 and 8 frames) is *which keyframes* are
chosen: the CLIP scorer. (b) kf-oracle → oracle (4.5 points at 4 frames, 2 at 16) is *where inside the
window* the frames land: the fixed 5 s window after a keyframe places frames evenly, so a 3 s action in a
5 s window gets ~60% hits, and a 0.5 s action is stepped over entirely (seen in 003 on video 52). Both
losses shrink with budget, which is why 16 frames is already near the ceiling.

This experiment attacks (b) and part of (a) without training anything.

## Hypothesis
Three changes, each cheap, together lift kf-clip-4 from 0.795 to ≥ 0.83 and kf-clip-8 from 0.875 to
≥ 0.90 at ≤ +40 ms per video:

1. **Gap-based window.** Use the interval to the *next keyframe* (capped at `max_window_s`) as the
   window instead of a fixed 5 s, since the action that made this keyframe score high lies before the
   next keyframe. Expected: fewer frames wasted past the action.
2. **Score-proportional allocation.** Split the budget across chosen keyframes in proportion to
   softmax(score / τ) instead of nearly equally, so a confident peak gets most of the frames.
3. **Coarse-to-fine refinement.** For each chosen window, decode it once sequentially at thumbnail
   resolution (reading forward from the keyframe costs ~0.3 ms/frame; a 5 s window at stride 3 is
   ~50 thumbnails), score those thumbnails with CLIP (~8 ms, batched), and place the window's frames at
   the highest-scoring moments instead of evenly. This is what finds a 0.5 s action inside a 5 s window.
   Expected cost: ~20–40 ms per video (decode + one CLIP batch); the full-resolution frames are then
   read at exactly the chosen indices (clustered → sequential reads, as now).

## Plan
1. Implement in `KeyframeSampler`: `window="gap"|float`, `alloc="softmax"|"equal"` with `temperature`,
   `refine=None|scorer` with `refine_stride`. Keep the current behaviour as the defaults for A/B.
2. Unit tests on the synthetic clip: gap window never crosses the next keyframe; softmax allocation
   sums to the budget; refinement with the oracle scorer lands frames inside the moving segment.
3. Ablation on the server, kf-clip at 4/8/16, each change alone and all together, plus kf-oracle with
   refinement (to see how far (b) closes). Same JSONL schema; `refine_ms` logged inside `select`.
4. Fill results; decide which become defaults.

## Metrics
Top-1, hit rate, select / decode / infer / total ms. Success: kf-clip-4 ≥ 0.83 and kf-clip-8 ≥ 0.90 with
total ms within +40 of 005 (152 / 230).

## Risks
- Refinement doubles the number of CLIP calls per video; batched it is ~8 ms, but decoding the window
  thumbnails is CPU work on the critical path. If it exceeds 40 ms, run it only for the top-1 window.
- Score-proportional allocation can collapse onto one wrong peak at 4 frames; the temperature controls
  this and will be swept (τ ∈ {0.05, 0.1, 0.2}).

## Results
*(to fill)*

## Decision
*(to fill)*
