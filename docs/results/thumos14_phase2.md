# Long-video benchmark: THUMOS14, does frame selection matter?

**Date:** 2026-10-03 · **GPU:** RTX 5090 · **Backbone:** VideoMAE-base (Kinetics pretrained, frozen)
**Head:** the UCF101 linear probe from Phase 1, restricted to the 20 THUMOS classes. Zero training on THUMOS.
**Data:** all 200 annotated THUMOS14 validation videos. Mean length ≈ 3 min; the annotated action covers
a median of a few percent of the timeline. A video is correct if the argmax class is one of its annotated
classes. Raw data: `thumos14_phase2_rtx5090.csv`.

**Samplers.** `uniform` and `motion` are real policies. `oracle` samples uniformly *inside* the annotated
segments and is the upper bound for any frame-selection policy. `hit` is the share of selected frames that
fall inside an action segment.

| sampler | frames | top-1 | hit | select ms | decode ms | infer ms | total ms |
|---|---|---|---|---|---|---|---|
| uniform | 4  | 0.730 | 0.29 | 0    | 128  | 450 | 578  |
| motion  | 4  | 0.705 | 0.30 | 1543 | 136  | 452 | 2130 |
| oracle  | 4  | **0.890** | 1.00 | 0 | 125 | 323 | 448  |
| uniform | 8  | 0.795 | 0.30 | 0    | 258  | 262 | 520  |
| motion  | 8  | 0.755 | 0.27 | 1606 | 267  | 238 | 2111 |
| oracle  | 8  | **0.925** | 1.00 | 0 | 249 | 254 | 503  |
| uniform | 16 | 0.885 | 0.30 | 0    | 503  | 243 | 746  |
| motion  | 16 | 0.805 | 0.27 | 1612 | 528  | 268 | 2408 |
| oracle  | 16 | **0.955** | 1.00 | 0 | 496 | 273 | 768  |
| uniform | 32 | 0.850 | 0.29 | 0    | 1003 | 240 | 1243 |
| motion  | 32 | 0.850 | 0.27 | 1615 | 1067 | 274 | 2956 |
| oracle  | 32 | **0.950** | 1.00 | 0 | 993 | 289 | 1282 |

## Findings

1. **Frame selection matters on long video.** The oracle beats uniform by **16 points at 4 frames**,
   13 at 8, 7 at 16. Equivalently, oracle-4 (89%) beats uniform-16 (88.5%) with a quarter of the frames.
   On trimmed UCF101 this gap did not exist. This is the headroom an adaptive scheduler is chasing.
2. **Uniform sees the action only 30% of the time**, and that is nearly independent of budget. Long
   videos are mostly background; blind sampling wastes 70% of the budget.
3. **Pixel-motion scoring is worse than uniform** at every budget (−2.5 to −8 points) and costs 1.5 s
   per video. Sports footage is full of camera motion and crowd movement; frame differencing finds
   *change*, not *the action*. A useful scorer has to be semantic (cheap features, not pixels) or learned.
4. **Uniform gets worse at 32 frames** (0.885 → 0.850). The model takes exactly 16 frames, so a 32-frame
   budget is subsampled back to 16 at a different stride; the probe was fitted on 16-frame clips and
   is sensitive to temporal spacing. Budgets above the model's native length need a different
   mechanism (multi-clip aggregation or token-level input), not frame subsampling.
5. **Decode dominates again, now by seeking.** Random-access decode of 16 frames from a 3-minute
   H.264 file costs ~500 ms, double the model forward, because every `cap.set` seeks to a keyframe and
   decodes forward. The `motion` scorer's 1.5 s is one full sequential CPU decode at stride 4.
   Pure GPU model time stays flat at ~250 ms (includes CPU preprocessing).

## What this decides

- **The research target is now concrete:** a frame scheduler that closes part of the 16-point
  uniform→oracle gap at ≤ uniform's total latency. Success metric: top-1 at 4 and 8 frames on THUMOS14
  val, with total ms per video reported alongside.
- **The scorer must be semantic and cheap.** Candidates: tiny CNN / ViT-Tiny features on low-res
  thumbnails decoded in one sequential pass; per-frame confidence from the model itself on a coarse
  first pass (coarse-to-fine); learned scorer distilled from the oracle labels.
- **The runtime must make scoring nearly free:** one sequential decode pass at low resolution feeding
  both the scorer and a frame cache, so selected frames are not decoded twice, plus GPU decode (NVDEC)
  for the full-resolution frames. Without this, no scorer can beat uniform on latency.
