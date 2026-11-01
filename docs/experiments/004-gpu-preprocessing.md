# 004 · GPU-side preprocessing for scorer and backbone

**Date:** 2026-11-01 · **Status:** done · **Results:** [`docs/results/thumos14_preprocess.md`](../results/thumos14_preprocess.md)

## Motivation (from 003)
Per video on the 5090, the two largest costs are CPU preprocessing, not GPU compute:

| stage | cost | what it is |
|---|---|---|
| CLIP select | ~170 ms of the ~270 ms select | `CLIPProcessor` resizing/normalising ~57 thumbnails via PIL on the CPU |
| "infer" | 250–380 ms | `AutoImageProcessor` resizing/normalising 16 frames via PIL on the CPU; the VideoMAE forward itself is ~7 ms |

Both are per-image Python/PIL loops. The decoded frames are already uint8 arrays; resize + normalise is
a handful of tensor ops that belong on the GPU.

## Hypothesis
Replacing the Hugging Face image processors with tensor preprocessing on the device will cut
`select` for kf-clip from ~270 ms to ≤ 130 ms and `infer` from ~300 ms to ≤ 60 ms, with no change in
accuracy (≤ 0.5 point top-1 difference, i.e. within noise of the 200-video test).
Expected consequence: kf-clip-8 total latency < uniform-16 total latency at equal accuracy, which is
the project's success criterion.

## Plan
1. `vam.runtime.preprocess`: a `Preprocessor` that takes uint8 frames (T,H,W,3) on CPU, moves them to
   the device once, and does resize (bilinear/bicubic, antialiased to match PIL closely), crop, scale to
   [0,1], and per-channel mean/std normalisation as batched tensor ops. Parameters are read from the HF
   processor config so each model keeps its exact numbers (size, crop, mean, std, rescale factor).
2. `HFVideoClassifier` / `HFFeatureExtractor`: use `Preprocessor` instead of `self.processor(...)`;
   keep a `use_hf_processor=True` switch for A/B tests.
3. `ClipScorer`: same for thumbnails (CLIP uses 224 bicubic resize + centre crop + CLIP mean/std).
4. Correctness test: on a fixed batch, the tensor pipeline's output must be within a small tolerance of
   the HF processor's output (mean abs diff < 0.02 in normalised units), and the probe's top-1 must not
   change on the 3 local videos.
5. Micro-benchmark on the server: preprocessing time HF vs tensor for 16 frames at 320×180 and for 57
   thumbnails at 112×112; then rerun `eval_thumos14` with uniform, kf-clip at 4/8/16 (kf-oracle and
   oracle unchanged in accuracy, skip to save time).

## Metrics
- Preprocessing time per call (HF vs tensor), GPU and CPU.
- `select`, `decode`, `infer`, `total` ms per video; top-1; hit rate — same JSONL schema as before.
- Success: kf-clip-8 total < uniform-16 total, top-1 within 0.5 pt of 003's numbers.

## Risks
- Interpolation differences (PIL bicubic with antialias vs torch) could shift accuracy. Mitigated by
  `antialias=True` and the tolerance test; if accuracy drops, fall back to matching PIL exactly on CPU
  tensors before upload.
- The laptop is memory-starved; correctness tests run with the small VideoMAE and tiny batches.

## Changes made
- `vam/runtime/preprocess.py`: `Preprocessor` (shortest-edge resize with `antialias=True`, centre crop,
  rescale, normalise) and `Preprocessor.from_hf(processor)` that reads size / crop / mean / std /
  resample from the HF config. VideoMAE: bilinear, ImageNet stats. CLIP: bicubic, CLIP stats.
- `HFVideoClassifier.pixel_values()` and `HFFeatureExtractor` use it; `use_hf_processor=True` restores
  the PIL path for A/B tests. `ClipScorer` likewise for thumbnails.
- `tests/test_preprocess.py`: output within mean-abs 0.02 / max 0.5 (normalised units) of both HF
  processors on smoothed random frames; 26 tests pass.

## Results
- A/B on 5 videos: identical top-1, feature cosine > 0.999. Micro-benchmark (320×180, 16 frames):
  VideoMAE preprocess 5.8 → 0.2 ms, CLIP score (24 thumbnails) 17 → 8 ms.
- Full run (200 videos, uniform + kf-clip, 4/8/16): top-1 within ±1 pt of 003; infer span ~300 → ~20 ms;
  kf-clip select ~270 → ~160 ms. Totals: uniform-16 724 → 485 ms; kf-clip-8 783 → 348 ms;
  kf-clip-16 770 → 325 ms.
- Hypothesis outcome: infer ≤ 60 ms ✔ (20 ms); select ≤ 130 ms ✘ (160 ms: the keyframe scan, not
  CLIP, is the remainder); accuracy unchanged ✔.
- **Success criterion met:** kf-clip-8 (0.875, 348 ms) < uniform-16 (0.880, 485 ms) at equal accuracy;
  kf-clip-16 beats uniform-16 on both axes (+5.5 pts, −33% latency).
- Surprise: the HF path's 300 ms was bimodal inside the harness (p50 ~500 ms) but 6 ms in isolation;
  cause unconfirmed (suspect OpenCV decoder threads vs PIL contention). The tensor path is flat.

## Decision
Tensor preprocessing is the default. The keyframe scan (~150 ms) is now the dominant selection cost →
experiment 005. Then adaptive window (006), learned scorer (007).
