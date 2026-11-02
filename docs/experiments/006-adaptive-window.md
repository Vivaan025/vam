# 006 · Adaptive window and coarse-to-fine refinement

**Date:** 2026-11-01 · **Status:** done · **Results:** [`docs/results/thumos14_adaptive.md`](../results/thumos14_adaptive.md)

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

## Changes made
- `KeyframeSampler(window="gap"|"fixed", min_window_s, alloc="equal"|"softmax", temperature, refine=scorer,
  refine_stride, keep_edge)`. Refinement decodes each window once (`Video.iter_frames(step)`), scores the
  thumbnails, places frames at the peaks with a minimum spacing of `window/(2k)` candidates, and keeps the
  decoded frames so the new `Sampler.read(video, indices)` hook hands them over without a second decode.
- `eval_thumos14.py --kf-window --kf-alloc --kf-temp --kf-refine --kf-refine-stride --tag`; `refine_ms` logged.
- Four tests: gap window stops at the next keyframe, softmax allocation sums to the budget, refinement with an
  oracle moves all frames into a short action, cached frames equal a fresh decode.

## Results
(200 videos, kf-clip, 4 / 8 / 16 frames; full tables in the results file.)
- **Gap window**: 0.815 / 0.885 / 0.935 at 148 / 223 / 225 ms. +2 / +1 / 0 points, same latency.
- **Softmax alone**: 0.785 / 0.875 / 0.920. Slightly worse; the top window is not reliable enough to bet on.
- **Refine alone**: 0.835 / 0.890 / 0.925 at 236 / 413 / 411 ms.
- **Gap + softmax 0.2 + refine**: **0.865 / 0.900 / 0.925** at 219 / 378 / 377 ms. τ ∈ {0.05, 0.1, 0.2}
  within 1 point of each other.
- **kf-oracle + refine**: 0.865 / 0.915 / 0.945 (oracle 0.890 / 0.925 / 0.955): the mechanism is sound.
- Hypothesis check: accuracy targets met (0.865 ≥ 0.83, 0.900 ≥ 0.90); latency target missed (+67 / +148 ms vs
  +40). Profiling: OpenCV seek is a flat 35–60 ms per window and decode 0.3 ms/frame, so a full 5 s window
  costs ~95 ms; CLIP on the window thumbnails is 49 ms per 200 thumbnails. PyAV is slower still (0.9 ms/frame).

## Decision
- `window="gap"` becomes the default (free). Softmax allocation stays off by default (helps only with
  refinement); default τ = 0.2.
- Refinement stays opt-in (`refine=scorer`). It is the right accuracy mechanism and is kept in the API, but it
  cannot be the default until the window decode is cheap. That is experiment 007 (windows decoded in parallel
  on the process pool, as the scan is), with NVDEC as the fallback (008). The learned scorer moves to 009.
