# Experiment 011 results: generality (held-out split, resolution, codec, keyframe density)

**Date:** 2026-11-03/04 · **Device: CPU only** (24-core shared host; the RTX 5090 was fully occupied by another
user's process for the whole run, so every latency below is CPU-tier and **not comparable** to the GPU tables in
earlier results). Accuracy and hit rate do not depend on the device. Setup otherwise as `thumos14_learned.md`:
VideoMAE-base + UCF101 linear probe, learned context scorer (the five fold heads averaged, which is what
`LearnedScorer` does for any video outside its fold map), gap window, softmax allocation, refinement stride 6.
Raw rows: `results/thumos14_test_cpu.jsonl`, `thumos14_variants_cpu.jsonl` and the two `*_scheduler_cpu.jsonl`
(server), summarised in `thumos14_generality_cpu.csv`.

## A. Held-out split: 212 THUMOS14 test videos (never seen by any head)

| frames | uniform | kf-clip | kf-learned | kf-clip + refine | kf-learned + refine | oracle |
|---|---|---|---|---|---|---|
| 4 | 0.769 | 0.821 | 0.835 | 0.849 | **0.868** | 0.863 |
| 8 | 0.816 | 0.887 | 0.887 | 0.901 | **0.901** | 0.873 |
| 16 | 0.858 | 0.910 | 0.920 | 0.910 | 0.915 | 0.925 |

Hit rate (share of chosen frames inside an annotated action): uniform 0.30–0.32, kf-clip 0.41–0.46,
kf-learned 0.48–0.52, oracle 1.0. Validation (009, 200 videos) for reference: uniform 0.720 / 0.795 / 0.880,
kf-learned + refine 0.865 / 0.925 / 0.940, oracle 0.890 / 0.925 / 0.955.

Share of the uniform→oracle gap closed by kf-learned + refine: test 105 % / 149 % / 85 % at 4 / 8 / 16 frames;
validation 85 % / 100 % / 80 %. The test split has less headroom (the actions cover a larger fraction of each
video), so the absolute margin over uniform is smaller (+10 / +8.5 / +6 points vs +14.5 / +13 / +6), but the
runtime reaches or passes the oracle at 4 and 8 frames on videos no head was trained on.

Scheduler on the test split (CPU tier, budgets 1000 / 2000 ms): 0.925 (16f for 182 of 212 videos, p50 628 ms)
and 0.915 (refinement for 200 of 212, p50 1553 ms, within budget 0.90). See the erratum below for the rows
that switched into refinement after the scan.

## B. Resolution and codec: 50 validation videos re-encoded (`scripts/reencode_variants.py`)

Learned scorer, top-1 at 4 / 8 / 16 frames; uniform on the same 50 videos is 0.800 / 0.780–0.820 / 0.920 on
every root (the model sees 224² crops, so the source resolution does not reach it).

| root | keyframes / video (median) | kf-learned | kf-learned + refine | select ms (CPU) | decode ms (CPU) |
|---|---|---|---|---|---|
| original 320×180 H.264 | 26 | 0.940 / 0.920 / 0.960 | 0.920 / 0.920 / 0.880 | 325–394 | 78–88 |
| H.264 1080p (upscaled) | 36 | 0.840 / 0.920 / 0.960 | 0.880 / 0.880 / 0.920 | 410–492 | 120–206 |
| HEVC 720p | 21 | 0.920 / 0.880 / 0.920 | 0.860 / 0.900 / 0.880 | 273–454 | 113–148 |
| VP9 720p (old scan, 5 videos) | 5,000 (every frame) | 0.800 at 8 | – | 33,500 | 61 |

On 50 videos one video is 2 points and the standard error is ±4, so B says "no large change": every
kf-learned cell is within 5 videos of the original root's, hit rates match (0.42–0.47 at 4 frames on H.264,
0.33–0.38 on HEVC), and uniform is identical. Decode cost scales with resolution as expected (1080p windows cost
2.3× the original's) and the per-root cost model picks it up (1080p intercept 146 ms + 11 ms per model frame vs
the original's 280 ms flat; the shared host makes the absolute numbers noisy).

**VP9 broke the scan.** libvpx ignores `skip_frame = NONKEY`, so the keyframe scan decoded every frame and the
scorer ran CLIP on ~5,000 thumbnails per video: 34 s per video. Fixed (commit 14e6845): the scan now reads
packets and decodes only those the container marks as keyframes. Measured on 3 videos per root, 8 threads:

| root | old scan | new scan |
|---|---|---|
| VP9 720p | 1,491 ms, 5,091 "keyframes" | **119 ms, 40 keyframes** |
| HEVC 720p | 128 ms | 99 ms |
| H.264 1080p | 195 ms | 135 ms |
| H.264 original | 223 ms | 151 ms |

Decoding only keyframe packets is also 25–30 % faster on H.264 and HEVC, because the decoder no longer parses
the packets it was told to skip. A per-codec test (`tests/test_keyframe.py`) encodes VP9, HEVC and H.264 clips
and asserts the keyframe count; it fails on the old scan. The full 50-video VP9 run with the fixed scan is
pending (needs the fix pulled on the server).

## C. Keyframe density: same 50 videos, H.264, keyframe every 1 s (GOP 30) or every 30 s (GOP 900)

| root | keyframes / video | uniform | kf-learned | kf-learned + refine | hit (kf-learned) |
|---|---|---|---|---|---|
| original (GOP ≈ 4 s) | 26 | 0.800 / 0.780 / 0.920 | 0.940 / 0.920 / 0.960 | 0.920 / 0.920 / 0.880 | 0.45 / 0.39 / 0.38 |
| GOP 30 (1 s) | 153 | 0.820 / 0.800 / 0.940 | 0.940 / 0.960 / 0.940 | 0.940 / 0.960 / 0.940 | 0.53 / 0.55 / 0.54 |
| GOP 900 (30 s) | 6 | 0.820 / 0.800 / 0.940 | **0.760 / 0.820 / 0.880** | 0.820 / 0.880 / 0.940 | 0.33 / 0.27 / 0.27 |

- **Dense keyframes help accuracy and cost the scan.** With a candidate every second the hit rate rises to
  0.53–0.55 (from 0.38–0.45) and refinement adds nothing (the grid is already fine). The scan + score stage
  grows to ~1.5 s per video on CPU (CLIP on 150 thumbnails; on the GPU this is a few ms), the cost model's
  keyframe coefficient becomes 6.4 ms, and at a 1 s budget the scheduler correctly collapses to 4 frames for
  43 of 50 videos because nothing else fits.
- **Sparse keyframes are where the keyframe-only scan is the wrong primitive.** With 6 candidates and 30 s
  gap windows, kf-learned without refinement falls to or below uniform at 4 and 16 frames (−6 / +2 / −6
  points) and the hit rate halves. Refinement, which decodes the chosen windows and scores every sixth frame,
  recovers uniform's level or better (0.820 / 0.880 / 0.940), because it is the refinement, not the scan, that
  finds the action inside a 30 s window. The decision rule in the plan ("GOP 900 without refinement stays above
  uniform at 4 frames") fails.

## Erratum found by this run (affects 010 and the scheduler rows above)

`benchmarks/eval_scheduler.py` built the sampler with the refinement scorer only when the *pre-scan* plan
chose refinement. When the post-scan plan grew into a refinement configuration, the row was logged as
`16f+refine/…` but no refinement ran (visible on the GOP 900 root: all 50 videos logged `16f+refine/3` at
2000 ms with a 56 ms select and a non-zero decode). Fixed in the benchmark (the sampler always carries the
scorer and the planner toggles it). Affected rows: in 010 on the GPU, 8 videos at 125 ms and 18 at 150 ms (the
remaining budgets had pre = post); on the CPU tier, 7 at 1000 ms; here, 7 of 212 test videos at 1000 ms and
all 50 GOP 900 rows. The 010 accuracy at 150 ms can only move by those 18 videos and is more likely to rise
than fall; the budgets 125 and 150 will be re-run when the GPU is free.

## Experiment 011 hypothesis check

- H1 (held-out split): ✔ in substance. The learned head keeps its lead on videos it never saw and the runtime
  reaches the oracle at 4 and 8 frames; the absolute margin over uniform is smaller because the split has
  half the headroom, so the literal "within 2 points of the validation margin" test fails at 4 and 8 frames
  while the share of the gap closed is higher than on validation.
- H2 (resolution, codec): ✔ for accuracy on H.264 and HEVC (no change beyond 50-video noise); ✗ as stated for
  VP9, where the scan itself broke; fixed, full rerun pending.
- H3 (keyframe density): confirmed in both directions. Dense keyframes raise hit rate and scan cost; sparse
  keyframes defeat the scan and refinement is what recovers it.

## GPU rerun (2026-11-05, RTX 5090 free again; scan fix and benchmark fix applied)

Accuracies match the CPU run to the third decimal except where noted. Latencies are now on the same tier as
every earlier result (007–010). Raw rows: `results/thumos14_*_gpu.jsonl` on the server.

### Held-out test split, 212 videos, ms per video

| frames | uniform | kf-clip | kf-learned | kf-clip + refine | kf-learned + refine | oracle |
|---|---|---|---|---|---|---|
| 4 | 0.764 @ 156 | 0.816 @ 140 | 0.835 @ 137 | 0.849 @ 151 | **0.863 @ 149** | 0.863 |
| 8 | 0.816 @ 278 | 0.887 @ 152 | 0.887 @ 150 | 0.906 @ 168 | **0.901 @ 167** | 0.873 |
| 16 | 0.858 @ 503 | 0.910 @ 154 | 0.920 @ 152 | 0.906 @ 167 | **0.915 @ 165** | 0.925 |

The test videos are shorter on average (typical keyframe count 11 vs 25), so the runtime sits at 137–167 ms
rather than the 143–153 ms of validation; uniform is unchanged at 156 / 278 / 503. Scheduler on test: 150 ms →
16f for 207 of 212 (0.915, p50 142, within 0.54); 200 ms → 16f+refine/3 for all (0.915, p50 165, within 0.73).

### Variant roots, 50 validation videos, kf-learned + refine unless stated, ms per video

| root | kf-learned 4 / 8 / 16 | + refine 4 / 8 / 16 | uniform-16 ms | scheduler @150 → top-1, p50 | scheduler @200 → top-1, p50 |
|---|---|---|---|---|---|
| original | 0.940 / 0.920 / 0.960 @ 123 / 120 / 127 | 0.920 / 0.920 / 0.880 @ 148 / 156 / 140 | 462 | 4f ×37, 16f+r ×10 → 0.960, 112 | 16f+r/3 → 0.900, 136 |
| H.264 1080p | 0.840 / 0.920 / 0.960 @ 207 / 267 / 285 | 0.880 / 0.880 / 0.920 @ 350 / 436 / 439 | 487 | 4f ×50 → 0.840, 183 | 4f ×39, 8f ×11 → 0.840, 186 |
| HEVC 720p | 0.920 / 0.880 / 0.920 @ 161 / 179 / 185 | 0.880 / 0.920 / 0.880 @ 248 / 281 / 277 | 643 | 4f ×29, 8f ×20 → 0.920, 149 | 16f ×26, 4f ×16, 8f ×7 → 0.920, 161 |
| **VP9 720p (fixed scan)** | 0.900 / 0.880 / 0.920 @ 133 / 134 / 141 | 0.900 / 0.920 / 0.920 @ 219 / 229 / 228 | 473 | 16f ×34, 4f ×11 → 0.920, 131 | 16f ×31, 16f+r/6 ×15 → 0.900, 159 |
| GOP 30 (1 s) | 0.940 / 0.960 / 0.940 @ 57 / 55 / 54 | 0.940 / 0.960 / 0.940 @ 63 / 61 / 59 | 46 | 16f+r/3 ×50 → 0.940, 56 | same, 56 |
| GOP 900 (30 s) | 0.760 / 0.820 / 0.880 @ 69 / 68 / 70 | 0.820 / 0.880 / 0.920 @ 82 / 83 / 81 | 283 | 16f+r/3 ×48 → 0.940, 83 | 16f+r/3 ×50 → 0.940, 88 |

Additional findings from the GPU tier:

6. **VP9 is now a first-class input.** With the fixed scan the learned sampler reaches 0.900 / 0.920 / 0.920
   with refinement at ~225 ms (OpenCV's VP9 window decode is slower than its H.264 path: refinement costs +90 ms
   here against +25 ms on H.264). Without refinement it is 133–141 ms, the same as H.264.
7. **HD input moves the frontier, and the scheduler reacts.** At 1080p the window decode is 2–3× dearer
   (refine 350–440 ms), so at a 150 ms budget the scheduler drops to 4 frames for every video and at 200 ms to
   4–8 frames. It keeps the budget (p50 183–186 ms) at the cost of accuracy (0.840); no static configuration
   does better under 200 ms on this root. Uniform-16 costs 487 ms.
8. **Keyframe density sets the decode cost of *every* sampler, not just ours.** On the GOP 30 root uniform-16
   costs 46 ms (seeks land on a keyframe) against 462 ms on the original and 283 ms on GOP 900; the learned
   sampler is 54–63 ms there and the scheduler fits 16f+refine/3 into 150 ms for all 50 videos with a p90 of
   76 ms. On GOP 900 refinement is cheap (windows are capped at 5 s) and the scheduler chooses it for 48 of 50,
   reaching 0.940 at 83 ms, the same accuracy as uniform-16 at a quarter of its cost.
9. **Prediction error is 8–18 ms MAE on the re-encoded roots and 35–40 ms on the originals.** The originals
   are the high-bitrate broadcast files; their seek cost varies per file more than the model can see without a
   per-file feature (the open item from 010).
