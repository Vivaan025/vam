# 003 · Keyframe-scan two-stage runtime

**Date:** 2026-11-01 · **Status:** done · **Results:** [`docs/results/thumos14_keyframe.md`](../results/thumos14_keyframe.md)
*(written retroactively on 2026-11-01 when the experiment log was introduced)*

## Hypothesis
Decoding only keyframes gives a scan cheap enough to score semantically, and decoding short windows
forward from the best keyframes recovers most of the oracle's accuracy at lower decode cost than uniform.

## Feasibility measurements (before building)
- Keyframe scan: 96 ms/video over all 200 videos (vs ~500 ms random-access read of 16 frames).
- Median 57 keyframes/video, gap 5 s; actions median 2.9 s. 36% of actions contain a keyframe;
  98% have one within 4 s. ⇒ scan locates the region; a window after the keyframe finds the action.
- Thumbnail conversion costs more than the keyframe decode itself ⇒ convert at low resolution only.

## Changes
- `vam.runtime.scan.scan_keyframes` (PyAV, `skip_frame=NONKEY`).
- `vam.sampling.keyframe.KeyframeSampler`: score keyframes → temporal NMS → split budget → windows of
  `window_s`=5 s after chosen keyframes.
- Scorers: `OracleKeyframeScorer` (mechanism test), `ClipScorer` (zero-shot, class vs background
  prompts), `uniform_scores`.
- `Video.read` reads forward sequentially when indices are within 64 frames instead of seeking per frame.
- `eval_thumos14.py`: `kf-oracle`, `kf-clip`, `--window`, `--device`.

## Metrics
Top-1, hit rate, select / decode / infer / total ms at 4/8/16 frames.

## Results
kf-oracle closes 72% of the gap at 4 frames and 100% at 8 (= oracle). kf-clip beats uniform at every
budget (+5.5/+8.0/+4.0); kf-clip-8 ≈ uniform-16 with half the frames. Windowed decode is cheaper than
uniform decode (156 vs 472 ms at 16). But select costs ~270 ms (scan ~100 + CLIP ~170, mostly CPU PIL
preprocessing), so total latency kf-clip-8 (783 ms) ≈ uniform-16 (724 ms). A 0.5 s action was missed
by the fixed 5 s window.

## Decision
Runtime validated; the scorer and the preprocessing path now decide the latency race.
Next: 004 GPU-side preprocessing, then adaptive window, then learned scorer.
