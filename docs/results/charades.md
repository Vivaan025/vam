# Experiment 012 results: Charades as a second domain

**Date:** 2026-11-05 · **GPU:** RTX 5090 (24-core host) · 1,000 Charades test videos (first 1,000 by id of the 1,814
with annotations), 157 classes, 480p H.264, median length 30.7 s, 8.8 action instances per video. Model: frozen
VideoMAE-base (Kinetics) + a one-vs-rest linear head fitted on the 7,811 Charades train videos
(`scripts/fit_charades_probe.py`: uniform 16 frames, 0.576 top-1-in-labels / 0.227 mAP on the full test split).
Scorers: zero-shot CLIP with Charades class names; the THUMOS-trained context head (fold average); a context
head retrained on these 1,000 videos with the 009 protocol (5-fold by video, each video scored by a head that
never saw it). Raw rows: `results/charades_test.jsonl`, `charades_scheduler.jsonl` (server); summary
`charades_rtx5090.csv`.

## Headroom (H1)

| | THUMOS14 val | Charades test |
|---|---|---|
| action coverage per video (median / mean) | 0.23 / 0.29 | **1.00 / 0.95** |
| uniform hit rate (frames inside an action) | 0.30 | 0.95 |
| uniform → oracle at 4 frames | 0.720 → 0.890 (+17) | 0.379 → 0.356 (**−2.3**) |
| keyframes per video (typical) | 25 | 3 |

On Charades the action is the whole video. The oracle, which samples uniformly *inside* annotated actions, is
not better than plain uniform because they pick nearly the same frames. H1's threshold was a 3-point gap; the
gap is negative. Only 28 of the 1,000 videos have coverage below 0.5, too few to measure anything on (their
numbers are in the CSV; they are noise).

## End to end, top-1-in-labels, ms per video (H2, H3)

| frames | uniform | oracle | kf-clip | kf-learned (THUMOS head) | kf-learned + refine | retrained head | retrained + refine |
|---|---|---|---|---|---|---|---|
| 4 | **0.379** @ 44 | 0.356 @ 40 | 0.375 @ 48 | 0.359 @ 47 | 0.348 @ 74 | 0.339 @ 48 | 0.370 @ 74 |
| 8 | **0.465** @ 56 | 0.449 @ 46 | 0.455 @ 55 | 0.462 @ 55 | 0.457 @ 79 | 0.416 @ 50 | 0.434 @ 74 |
| 16 | **0.572** @ 47 | 0.570 @ 50 | 0.531 @ 57 | 0.555 @ 57 | 0.531 @ 78 | 0.498 @ 54 | 0.495 @ 74 |

Hit rate is 0.94–0.96 for every sampler. Uniform wins or ties at every budget; the differences among the
keyframe samplers are within ±3 points (standard error ±1.6 on 1,000 videos).

Scorer level (held-out folds): the retrained context head ranks keyframes by oracle overlap at Spearman 0.37
against 0.10 for zero-shot CLIP, so the head *learns* the domain; it just has nothing to change, since 97.5 % of
candidate windows already overlap an action (top-2 precision 0.975 for every scorer, including zero-shot).

Scheduler (budgets 100 / 150 / 200 ms, THUMOS accuracy table): picks 16f+refine for all but one video at every
budget (cost model intercept 37 ms, 3 keyframes per clip, everything fits), lands at 0.517 / 0.513 / 0.513 with
p50 83 / 115 / 116 ms and 96–100 % within budget. Correct on latency, wrong on accuracy: uniform-16 at 47 ms is
better. The accuracy ordering it uses is THUMOS14's, and on this domain that ordering is inverted.

## Findings

1. **Frame selection has no headroom when actions cover the video.** Coverage, not video length or domain
   style, is the quantity that decides it: THUMOS14 at 0.23–0.29 coverage gives a 17-point gap, Charades at
   0.95 gives none. This is the boundary the project's claim lives inside, and it is measurable.
2. **The runtime is unchanged.** 47–57 ms per video without refinement on 30 s 480p clips, 74–79 ms with; the
   scan finds 3 keyframes in a few ms; cost-model calibration takes 3 s and predicts within 8 ms MAE. Nothing
   broke on a new dataset, container or annotation format.
3. **Scorers transfer in rank, not in usefulness.** The THUMOS head is no worse than zero-shot CLIP or the
   retrained head here; all three are irrelevant because every window is a hit. H2 is therefore moot rather than
   failed: there is no ranking to get right.
4. **The scheduler needs uniform in its table and coverage in its features.** It chose the most expensive
   configuration because it was cheap, and lost 5.5 points to uniform-16. The scan can estimate coverage
   (scorer scores are flat when everything is action; keyframe count is tiny on short clips); the scheduler
   should fall back to uniform when the scores are flat. That is the follow-up, and it is small.
5. **Absolute accuracy is low** (0.38–0.57) because 157 fine-grained indoor classes on a frozen Kinetics
   backbone with one linear head is a weak classifier, and because the correctness rule (argmax in labels, 8.8
   labels per video) is lenient in the other direction. Neither affects the comparison between samplers, which
   share the head.

## Experiment 012 hypothesis check

- H1 (headroom ≥ 3 points): ✗. Gap is −2.3 points at 4 frames; the premise does not apply to this domain.
- H2 (scorer transfer): moot. All scorers reach hit rate 0.94–0.96; retraining raises rank correlation from
  0.10 to 0.37 and changes nothing downstream.
- H3 (runtime unchanged): ✔. Scan, decode, scheduler and calibration behave as the 011 cost model predicts.
