# Experiment 005 results: process-parallel keyframe scan

**Date:** 2026-11-01 · **GPU:** RTX 5090 (24-core host) · setup identical to `thumos14_preprocess.md`.
Only change: `scan_keyframes` splits the timeline into 8 ranges scanned by a persistent process pool.
Raw data: `thumos14_scan_rtx5090.csv` (stride 1) and `thumos14_scan_stride2_rtx5090.csv`. 004 numbers in brackets.

| frames | sampler | top-1 | hit | select ms | decode ms | infer ms | total ms |
|---|---|---|---|---|---|---|---|
| 4 | uniform | 0.720 | 0.29 | 0 | 127 | 20 | 147 |
| 4 | **kf-clip** | **0.795** | 0.41 | **47** (159) | 85 | 20 | **152** (264) |
| 8 | uniform | 0.795 | 0.30 | 0 | 258 | 19 | 277 |
| 8 | **kf-clip** | **0.875** | 0.40 | **46** (162) | 166 | 18 | **230** (348) |
| 16 | uniform | 0.880 | 0.30 | 0 | 479 | 7 | 486 |
| 16 | **kf-clip** | **0.935** | 0.39 | **45** (162) | 157 | 8 | **210** (325) |

Stride-2 ablation (keep every other keyframe): select 40–41 ms; top-1 0.810 / 0.890 / 0.920 at 4 / 8 / 16.
Differences from stride 1 are ±1.5 points, within noise on 200 videos; the saving is ~6 ms because per-worker
fixed cost, not keyframe count, now dominates the scan. Stride stays at 1 by default.

## Findings

1. **Selection cost fell from ~160 ms to ~46 ms**, flat across budgets (the scan does not depend on the
   budget). Of the 46 ms, CLIP scoring is ~8 ms and the scan ~38 ms.
2. **kf-clip now beats uniform at every budget on both axes except 4 frames, where it is 5 ms slower
   and 7.5 points more accurate.** At 8 frames: +8.0 points and 17% faster. At 16 frames: +5.5 points
   and 2.3× faster (210 vs 486 ms), because decoding windows after keyframes costs a third of seeking to
   16 scattered frames.
3. **The whole accuracy/latency frontier is now kf-clip.** Sorted by latency: uniform-4 (0.720, 147),
   kf-clip-4 (0.795, 152), kf-clip-16 (0.935, 210), kf-clip-8 (0.875, 230), uniform-8 (0.795, 277),
   uniform-16 (0.880, 486). Every uniform point except the 147 ms one is dominated.
4. **Decode is now the largest remaining cost** for kf-clip (85–166 ms): seeking to the chosen
   keyframes and reading the windows with OpenCV on the CPU. NVDEC or a PyAV-based windowed reader that
   reuses the scan's containers is the next systems lever. Accuracy-wise, the gap to the oracle
   (0.890 / 0.925 / 0.955) is now the scorer's: 9.5 / 5 / 2 points.

## Experiment 005 hypothesis check

- Scan ≤ 50 ms: ✔ (38 ms within a 46 ms select).
- kf-clip-4 ≤ 170 ms: ✔ (152 ms).
- Accuracy unchanged: ✔ (identical keyframes, identical top-1 to 004).
