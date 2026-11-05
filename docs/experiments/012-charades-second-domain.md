# 012 · Second domain: Charades (indoor, handheld, multi-action)

**Date:** 2026-11-05 · **Status:** done · **Results:** [`docs/results/charades.md`](../results/charades.md)

## Motivation (from 011)
011 showed the runtime holds on THUMOS14 videos it never saw, and that resolution and codec are neutral. What
it cannot show is whether anything transfers to footage that looks nothing like broadcast sports: indoor,
handheld, several actions per video, actions that fill most of the timeline. Charades (Allen AI, 9,848 videos,
157 classes, ~30 s each, 480p available as a direct download) is the standard such set and ships temporal
annotations for every action instance. It is also the harder case for the premise of the project: when actions
cover 60 % of a video, frame selection has less to find.

## Hypotheses
- **H1, headroom exists but is smaller.** On Charades test, the uniform→oracle gap at 4 frames is positive but
  under half of THUMOS14 validation's 17 points. If it is below ~3 points the premise does not apply to this
  domain and the honest result is "use uniform here".
- **H2, the zero-shot and learned scorers transfer in rank, not in level.** The THUMOS-trained context head,
  without retraining, still orders keyframes better than uniform (hit rate above uniform's) but below zero-shot
  CLIP with Charades class names. A head retrained on Charades train (same recipe, 5-fold by video) recovers
  most of the gap to the oracle, as on THUMOS.
- **H3, the runtime is the same.** Scan, decode and scheduler costs follow the 011 variant pattern (480p
  H.264, ~25 keyframes per 30 s video): the per-machine cost model needs no change, and refinement on by default
  is the right call on short videos too.

## Plan
1. **Data** (`scripts/download_charades.py`): `Charades_v1_480.zip` (13 GB) and the annotation CSVs into
   `data/charades/`; a `Charades` dataset class with the THUMOS14 interface (`videos`, `classes`, segments per
   video, `frame_in_action`), so every benchmark takes it through `--dataset charades`.
2. **Probe**: the UCF101 head does not cover Charades classes. Fit a linear probe on frozen VideoMAE-base
   features of Charades train (uniform 16 frames, one feature per video, multi-label → one-vs-rest logistic
   heads), evaluate on test; a video counts correct if the argmax class is among its labels, as on THUMOS.
   Report mAP too, because that is what Charades papers report, but the decision metric stays top-1-in-labels.
3. **Benchmarks** on 1,000 test videos (the full 1,863 if time allows): uniform, oracle, kf-clip (Charades
   class prompts), kf-learned with the THUMOS head, kf-learned with a Charades-trained head; 4 / 8 / 16 frames,
   refinement on; scheduler at 100 / 150 / 200 ms on the 5090.
4. **Report** `docs/results/charades.md`: headroom table, scorer transfer table, hit rates, latency and
   scheduler choices, side by side with THUMOS14.

## Metrics
H1: uniform→oracle gap at 4 frames (decision threshold 3 points). H2: hit rate and top-1 of the THUMOS head vs
uniform and vs kf-clip; retrained head within 2 points of the oracle at 8 frames. H3: select ms within 20 %
of the 011 480p-equivalent, scheduler within-budget rates comparable to 010 at the same budgets.

## Risks
- 13 GB download and feature extraction on 7,985 train videos (~40 min on the 5090) before any result.
- Charades labels are noisy and multi-label; "argmax in labels" is lenient. mAP is the second opinion.
- If H1 fails the experiment still produces a result: the boundary of where frame selection pays off,
  stated as action coverage per video, which the scheduler could read from the scan.

## Changes made
- `vam.data.Charades` (THUMOS14-shaped), `scripts/download_charades.py`, `scripts/fit_charades_probe.py`
  (one-vs-rest head on frozen features: 0.576 top-1-in-labels / 0.227 mAP), `--dataset charades` on both
  benchmarks, `train_scorer.py --dataset/--split/--limit`.

## Results
- **H1 fails, and that is the result.** Action coverage per video is 0.95 (THUMOS14: 0.29); uniform frames are
  already inside an action 95 % of the time; the oracle is 2.3 points *below* uniform at 4 frames. There is no
  gap for any selector to close.
- End to end, 1,000 test videos: uniform 0.379 / 0.465 / 0.572, kf-learned 0.359 / 0.462 / 0.555, with
  refinement 0.348 / 0.457 / 0.531, retrained head 0.339 / 0.416 / 0.498; all hit rates 0.94–0.96. Uniform
  wins or ties at every budget.
- H2 moot: the retrained head ranks keyframes at Spearman 0.37 vs 0.10 for zero-shot, but 97.5 % of windows are
  hits for every scorer. H3 holds: 47–57 ms per video, scheduler calibrates in 3 s and predicts within 8 ms.
- The scheduler picked 16f+refine everywhere (it was cheap) and lost 5.5 points to uniform-16, because its
  accuracy ordering is THUMOS14's.

## Decision
- **State the boundary in the README and the pitch:** frame selection pays when action coverage is low (the
  long-video regime the project is about) and does nothing when the action fills the clip. Coverage is the
  variable, not domain or length.
- **Scheduler follow-up (R4, small):** add `uniform` to the configuration table and fall back to it when the
  scan's scores are flat / the keyframe count is tiny; log the scan's coverage estimate. Until then the
  scheduler is documented as a long-video tool.
- No retraining or redesign follows from this experiment. The next dataset worth running is long untrimmed
  video with sparse actions outside sports (ActivityNet-style), which is the regime the method claims and the one
  that would extend 011's held-out result to a new domain.
