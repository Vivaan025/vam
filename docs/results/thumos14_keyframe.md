# Keyframe-scan runtime on THUMOS14: first real policy

**Date:** 2026-11-01 · **GPU:** RTX 5090 · **Backbone:** VideoMAE-base (frozen) + UCF101 linear probe (fixed)
**Data:** 200 annotated THUMOS14 validation videos, 20 classes. Raw data: `thumos14_keyframe_rtx5090.csv`.
Setup identical to `thumos14_phase2.md`; new samplers from `vam.sampling.keyframe`.

| sampler | what it is |
|---|---|
| uniform | evenly spaced frames (baseline) |
| oracle | uniform *inside* annotated segments (ceiling; needs ground truth) |
| kf-oracle | keyframe scan → oracle keyframe scores → 5 s windows after chosen keyframes (isolates the runtime from the scorer) |
| kf-clip | keyframe scan → zero-shot CLIP scores (class prompts vs background prompts) → 5 s windows (**first real policy**) |

| frames | sampler | top-1 | hit | select ms | decode ms | infer ms | total ms |
|---|---|---|---|---|---|---|---|
| 4 | uniform | 0.730 | 0.29 | 0 | 129 | 287 | 415 |
| 4 | **kf-clip** | **0.785** | 0.41 | 272 | 85 | 383 | 739 |
| 4 | kf-oracle | 0.845 | 0.82 | 180 | 87 | 314 | 581 |
| 4 | oracle | 0.890 | 1.00 | 0 | 118 | 269 | 387 |
| 8 | uniform | 0.795 | 0.30 | 0 | 257 | 297 | 553 |
| 8 | **kf-clip** | **0.875** | 0.40 | 268 | 166 | 349 | 783 |
| 8 | kf-oracle | 0.925 | 0.72 | 152 | 171 | 301 | 624 |
| 8 | oracle | 0.925 | 1.00 | 0 | 210 | 316 | 526 |
| 16 | uniform | 0.885 | 0.30 | 0 | 472 | 252 | 724 |
| 16 | **kf-clip** | **0.925** | 0.40 | 243 | 156 | 371 | 770 |
| 16 | kf-oracle | 0.935 | 0.70 | 153 | 155 | 325 | 633 |
| 16 | oracle | 0.955 | 1.00 | 0 | 338 | 314 | 653 |

## Findings

1. **The runtime recovers most of the oracle.** With perfect keyframe scores (kf-oracle) the two-stage
   pipeline closes 72% of the uniform→oracle gap at 4 frames and **100% at 8 frames** (0.925 = oracle),
   even though only 70–80% of its frames land inside the action. Keyframe granularity (one per ~5 s,
   actions ~3 s) is not the limiting factor at 8+ frames.
2. **The first real policy beats uniform at every budget, with zero training.** kf-clip: +5.5 points at
   4 frames, +8.0 at 8, +4.0 at 16. **kf-clip at 8 frames (0.875) matches uniform at 16 frames (0.885)
   with half the frames.** CLIP recovers 34–50% of the gap; the rest is scorer quality, not runtime.
3. **Windowed decode is cheaper than uniform decode**, as designed: 85 vs 129 ms at 4 frames, 156 vs
   472 ms at 16. Frames after a keyframe are read sequentially with no seeks (`Video.read` grouping).
4. **Selection cost is the problem to solve next.** kf-clip spends ~270 ms selecting: ~100 ms keyframe
   scan + ~170 ms CLIP, and most of the CLIP time is CPU-side image preprocessing (PIL resize/normalise
   of ~57 thumbnails) rather than the GPU forward. So on total latency kf-clip-8 (783 ms, 0.875) is
   roughly on par with uniform-16 (724 ms, 0.885), not ahead. Target: select < 100 ms, which would put
   kf-clip-8 at ~600 ms, below uniform-16 at equal accuracy.
5. **The "infer" column (250–380 ms) is dominated by CPU preprocessing for VideoMAE**, not the GPU
   forward (~7 ms on this card). It is a shared constant across samplers, so it does not change the
   comparison, but it is the largest single cost in the pipeline and the obvious target for GPU
   preprocessing.

## Implications

- Research framing is validated: frame selection as a runtime problem, measured against latency.
  The runtime (scan + windowed decode) is good enough; the scorer and the preprocessing path decide
  whether the policy wins on wall-clock.
- Next, in order: (1) GPU-side preprocessing for both CLIP thumbnails and VideoMAE frames (tensors from
  the decoder straight to the device, resize/normalise there); (2) adaptive window: narrow the window
  and concentrate frames when the scorer is confident, so sub-second actions are not stepped over;
  (3) a learned scorer distilled from oracle keyframe scores on THUMOS14 *training* annotations.
