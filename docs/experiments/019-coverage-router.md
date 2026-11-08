# 019 · Coverage-aware routing: decide uniform vs keyframe from the scan, before paying for scoring (roadmap R10)

*Opened 2026-11-08 · Status: in progress*

## Motivation
012, 013, 017 and 018 draw one line through every result: frame selection pays when the action covers a small
part of the video (THUMOS14, median coverage 0.23, oracle +17 at 4 frames) and does nothing or hurts when it
fills the clip (Charades 0.95, ActivityNet 0.72, oracle +2.9). The library already falls back to uniform, but
only *after* scoring (flat scores, chosen windows covering > 0.5), so on ActivityNet it pays 34 CLIP images per
video and still loses 4.9 mAP at 4 frames with the retrained scorer (017 H3), and fires on only 5–12 % of
videos. 013's Decision called a real coverage estimate "a research item". This is it.

## Hypotheses (acceptance criteria)
- **H1, coverage is predictable from the scan alone.** A linear model on ~12 statistics of the keyframe scan
  (duration, keyframe rate, thumbnail-difference and self-similarity statistics; no CLIP, no classifier)
  reaches Spearman ≥ 0.5 against annotated action coverage on held-out videos of three datasets, and at one
  threshold routes ≥ 80 % of ActivityNet val and Charades test to uniform while keeping ≥ 80 % of THUMOS14 on
  the keyframe path.
- **H2, the router removes the loss without removing the gain.** With the router in front of scoring, the
  budget sampler on ActivityNet val is within 0.5 mAP of uniform at matched milliseconds at 100 / 150 ms (017:
  −0.4 / −2.6 against uniform-6 / uniform-10), and THUMOS14 val / test keyframe results change by < 1 point at
  4 / 8 / 16 frames with the fallback rate < 20 %.
- **H3, the router is cheap.** Feature extraction + prediction < 5 ms per video on top of the scan.

## Plan
1. `vam/sampling/coverage.py`: `scan_features(kf)` (fixed-length vector from `Keyframes`), `CoverageEstimator`
   (standardised ridge, `predict(kf) -> float in [0, 1]`, `.save / .load`, packaged default in `vam/weights/`).
2. `scripts/train_coverage.py`: scan every video of THUMOS14 val + test, ActivityNet train subset + val, Charades
   train + test (cached per dataset), target = `UntrimmedVideo.action_fraction(duration)`; fit on THUMOS14 val
   + ActivityNet train + Charades train, report Spearman / MAE and the routing confusion per held-out set at
   thresholds 0.3–0.7; write `vam/weights/coverage_ridge.npz`.
3. `KeyframeSampler(coverage_estimator=...)`: after the scan and before `scorer(kf)`, fall back
   (`last_fallback = "predicted-coverage"`) when the estimate exceeds `max_coverage`. Works with and without a
   planner, so `BudgetedSampler` inherits it through its keyword pass-through.
4. Benchmarks: `--coverage-router` in `eval_activitynet.py` and `eval_thumos14.py`; rerun ActivityNet budget-100 /
   150 and keyframe-4 / 8 (ActivityNet scorer), THUMOS14 val + test kf-learned at 4 / 8 / 16 and budget arms.
5. Record routing rates, accuracy at matched ms, and the per-video ms of the router.

## Metrics
Spearman and MAE of predicted vs annotated coverage; routing rate per dataset; mAP / top-1 at matched ms;
fallback counts by reason.

## Risks
- Coverage may not be visible in thumbnails: a long static shot of one activity looks like a long static shot
  of nothing. Duration and keyframe rate carry part of the signal; if H1 fails, a cheaper proxy (a few CLIP
  images at the scan's coarsest level) is the next step, not a release item.
- Training across three datasets with different label conventions (THUMOS14 segments, ActivityNet segments,
  Charades multi-action) may bias the threshold; evaluation is per dataset.

## Results
### 2026-11-08: H1 fails for pixel-level scan features
Ridge on 13 scan statistics, trained on THUMOS14 val + ActivityNet train + Charades train (9,323 videos), held
out THUMOS14 test / ActivityNet val / Charades test:

| set | n | coverage median | predicted median | Spearman | MAE | routed to uniform at 0.5 / 0.6 / 0.7 |
|---|---|---|---|---|---|---|
| THUMOS14 test (held) | 212 | 0.26 | 0.63 | −0.09 | 0.35 | 0.80 / 0.58 / 0.29 |
| ActivityNet val (held) | 4,926 | 0.73 | 0.66 | 0.27 | 0.23 | 0.93 / 0.70 / 0.39 |
| Charades test (held) | 1,814 | 1.00 | 0.92 | 0.12 | 0.10 | 1.00 / 1.00 / 1.00 |

- Within a dataset the features carry almost no coverage signal (THUMOS14 ρ ≈ 0). Between datasets only
  Charades separates (30 s clips: duration does it); THUMOS14 sports broadcasts and ActivityNet YouTube videos
  are indistinguishable at the thumbnail level (predicted medians 0.63 vs 0.66). No threshold keeps ≥ 80 % of
  THUMOS14 on the keyframe path while sending ≥ 80 % of ActivityNet to uniform. The feature pass also costs
  30–40 ms on long videos (thousands of thumbnails), a third of the scan.
- Reading: "how much the picture changes" is not "how much of the timeline is the action". Coverage is
  semantic. Next (the Risks fallback): statistics of the *learned scorer's* keyframe scores, which the normal
  path already computes, as the estimate; the saving is then the window decode and refinement, not the scan.


## Decision
pending
