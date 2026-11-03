# 011 · Generality: held-out split, resolution, codec, keyframe density

**Date:** 2026-11-03 · **Status:** planned · **Results:** *(to fill)*

## Motivation (from 009 and 010)
Every number so far is 200 THUMOS14 validation videos at 320×180 H.264 with the encoder's default keyframe
interval. Three things in the runtime were fitted or tuned on exactly those videos: the learned scorer (per-fold
heads, cross-validated on the same 200), the accuracy table the scheduler orders by, and the cost model's shape
(the keyframe-count term fitted to zero because every file has 25–100 keyframes). Roadmap R3 asks whether any of
it holds off that set. This entry covers the within-domain half: a split the heads never saw, and the same videos
re-encoded so that only resolution, codec or keyframe density changes. A second domain (indoor, handheld,
multi-action footage: Charades) is 012.

## Hypotheses
- **H1, held-out split.** On the 212 annotated THUMOS14 *test* videos, the learned scorer (the five fold heads
  averaged, which is what `LearnedScorer` does for an unseen video) keeps its advantage over zero-shot CLIP and
  uniform at 4 / 8 / 16 frames, within 2 points of its validation numbers relative to the oracle on the same
  split. The scheduler at 150 and 200 ms picks the same configurations and lands within 1 point of the
  hindsight-best static one.
- **H2, resolution and codec.** On 50 validation videos re-encoded to 1080p H.264, 720p HEVC and 720p VP9,
  top-1 and hit rate are unchanged (the scorer works on 112² thumbnails; the model on 224²), while the scan and
  decode stages cost more and the cost model, re-calibrated per variant in 6 s, predicts it: scheduler
  within-budget rate and prediction bias no worse than on the originals.
- **H3, keyframe density.** With a keyframe every second (GOP 30) the scan has ~10× more candidates: select
  cost rises and the scheduler's keyframe-count coefficient stops being zero; accuracy at 4 frames rises or
  holds (finer candidate grid). With a keyframe every 30 s (GOP 900) the scan sees 5–7 candidates per video:
  gap windows become 30 s, refinement must do the work, and accuracy without refinement falls toward uniform.
  This is the case where a keyframe-only scan is the wrong primitive, and we want to know by how much.

## Plan
1. **Data.** `scripts/download_thumos14.py --split test` (URLs fixed: the test videos live under
   `test_set/TH14_test_set_mp4/`, annotations under `TH14_Temporal_annotations_test.zip`); 212 videos.
   `scripts/reencode_variants.py --limit 50` makes five variant roots under `data/thumos14_variants/<name>/`
   (ffmpeg, libx264 / libx265 / libvpx-vp9), each a THUMOS14-shaped root so `--root` selects it.
2. **Benchmarks.** `eval_thumos14.py --split test` with samplers uniform, oracle, kf-clip (+refine) and
   kf-learned (+refine) at 4 / 8 / 16; `eval_scheduler.py --split test --budgets 150 200 --no-cache`. For each
   variant root: uniform, kf-learned + refine at 4 / 8 / 16 and the scheduler at 150 and 200 ms with `--no-cache`
   (cost model per variant). The 50-video subset of the originals is run with the same commands as the control.
3. **Report** (`docs/results/thumos14_generality.md`): test vs val table; per-variant table with select / decode /
   total ms, keyframes per video, top-1, hit rate; per-variant cost-model coefficients; scheduler choices.

## Metrics
H1: kf-learned minus uniform on test within 2 points of the val margin at each budget; scheduler within 1 point
of hindsight at 150 and 200 ms. H2: |Δ top-1| ≤ 2 points on 50 videos (noise ≈ ±4), scheduler prediction bias
within ±20 ms after per-variant calibration. H3: report the numbers; the decision rule is whether GOP 900 without
refinement stays above uniform at 4 frames.

## Risks
- 50 videos per variant is noisy (±4 points); conclusions about accuracy are "no large change", not equality.
- Upscaled 1080p carries no new information; it measures decode cost, not recognition at HD.
- VP9 in mp4 may take a different OpenCV path; if `cv2.VideoCapture` fails, record it as a robustness finding
  (R4) rather than silently falling back.
- The test split has the same 20 classes and camera styles as val; passing H1 says the head did not overfit its
  folds, not that it generalises to other footage.

## Results
*(to fill)*

## Decision
*(to fill)*
