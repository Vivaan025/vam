# Experiment 004 results: device-side preprocessing

**Date:** 2026-11-01 · **GPU:** RTX 5090 · setup identical to `thumos14_keyframe.md` (frozen VideoMAE-base,
fixed UCF101 probe, 200 THUMOS14 val videos). Only change: Hugging Face PIL image processors replaced by
`vam.runtime.preprocess.Preprocessor` (batched tensor ops on the device) for both VideoMAE frames and CLIP
thumbnails. Raw data: `thumos14_preprocess_rtx5090.csv`. Previous numbers (003) in brackets.

| frames | sampler | top-1 | hit | select ms | decode ms | infer ms | total ms |
|---|---|---|---|---|---|---|---|
| 4 | uniform | 0.720 (0.730) | 0.29 | 0 | 128 | **20** (287) | **148** (415) |
| 4 | kf-clip | 0.795 (0.785) | 0.41 | **159** (272) | 85 | **20** (383) | **264** (739) |
| 8 | uniform | 0.795 (0.795) | 0.30 | 0 | 260 | **19** (297) | **279** (553) |
| 8 | kf-clip | 0.875 (0.875) | 0.40 | **162** (268) | 166 | **19** (349) | **348** (783) |
| 16 | uniform | 0.880 (0.885) | 0.30 | 0 | 478 | **7** (252) | **485** (724) |
| 16 | kf-clip | 0.935 (0.925) | 0.39 | **162** (243) | 155 | **8** (371) | **325** (770) |

## Findings

1. **Accuracy unchanged.** Every top-1 is within ±1 point of 003 (noise on 200 videos). The tensor path
   reproduces the HF processors: identical top-1 on 5 A/B videos, feature cosine > 0.999.
2. **Inference span fell from ~300 ms to ~20 ms per video** (7 ms at 16 frames where no padding is
   needed). The HF processor path was *bimodal inside the harness* (p50 496 ms, ~40% of calls at 20 ms)
   while taking 6 ms in isolation on the same frames; the cause was not confirmed (CPU thread contention
   between OpenCV's decoder threads and the PIL processor is the leading suspect). The tensor path is a
   flat 19 ms across all 200 videos.
3. **CLIP selection fell from ~270 ms to ~160 ms.** CLIP scoring itself is now ~8 ms; the remaining
   ~150 ms is the keyframe scan (PyAV decode + thumbnail conversion), which is now the dominant selection
   cost and the next target.
4. **Success criterion met.** kf-clip-8 (0.875, **348 ms**) vs uniform-16 (0.880, **485 ms**): equal
   accuracy, 28% less latency, half the frames. And **kf-clip-16 beats uniform-16 on both axes**:
   +5.5 points top-1 (0.935 vs 0.880) and 33% less latency (325 vs 485 ms), because windowed decode
   after keyframes (155 ms) is three times cheaper than seeking to 16 scattered frames (478 ms).

## Accuracy vs latency (the trade-off curve)

| policy | top-1 | total ms |
|---|---|---|
| uniform-4 | 0.720 | 148 |
| kf-clip-4 | 0.795 | 264 |
| uniform-8 | 0.795 | 279 |
| kf-clip-16 | 0.935 | 325 |
| kf-clip-8 | 0.875 | 348 |
| uniform-16 | 0.880 | 485 |

kf-clip-4 matches uniform-8 at lower latency; kf-clip-16 dominates everything above it. The only region
where uniform still wins is the very low-latency end (< 250 ms), where the fixed ~150 ms scan cost
cannot be amortised. Cutting the scan cost extends the win to that region.

## Decision
Keep tensor preprocessing as the default. Next: make the keyframe scan cheaper (thumbnail conversion
and PyAV overhead), then the adaptive window, then a learned scorer.
