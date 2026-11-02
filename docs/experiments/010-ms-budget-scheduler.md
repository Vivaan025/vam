# 010 · Millisecond-budget scheduler

**Date:** 2026-11-02 · **Status:** done · **Results:** [`docs/results/thumos14_scheduler.md`](../results/thumos14_scheduler.md)

## Motivation (from 007 and 009)
The runtime has five measured knobs: frame budget (4 / 8 / 16), number of windows, refinement on/off,
refinement stride (3 / 6), and scorer. Each configuration is a point on an accuracy-versus-latency plane that
we have measured on one GPU. A user does not think in frames; they have a latency budget ("50 videos per
second on this box", "under 200 ms per clip") and want the best accuracy that fits. Today they would have to
read our tables and pick by hand, and the tables are for an RTX 5090 with a 24-core host: on a laptop every
stage costs differently (001: the 3050 is 4–8× slower on decode-bound stages).

This is the piece the project has promised since 002: *frame selection as a latency-constrained runtime
problem.* The scheduler turns the measured curves into a policy.

## Hypothesis
A scheduler that (a) calibrates a per-stage cost model on the target machine in a few seconds, (b) picks the
configuration with the highest expected accuracy whose predicted latency fits the budget, and (c) adapts per
video using the scan (number of keyframes, duration) matches the best static configuration at every budget
and keeps the realised latency within budget for ≥ 90% of videos, without the user naming frames or strides.

## Plan
1. **Cost model** (`vam/runtime/scheduler.py`, `CostModel`): per-video latency =
   `scan(K) + score(K) + decode(W, L) + refine(W, L, stride) + infer(F)` with
   - `scan(K) = a_s + b_s · K` (keyframes), `score(K) = a_c + b_c · K`,
   - `decode(W, L) = seek + b_d · L` (parallel windows: the longest window dominates, plus a per-window term for
     the pool), `refine = decode + b_r · (W · L / stride)` (CLIP on thumbnails), `infer(F) = a_i + b_i · F`.
   Coefficients are fitted by least squares from the Profiler spans logged in 007/009 (we already have 200 ×
   many rows), and re-fitted by `calibrate(videos, n=3)` on the target machine, which runs the pipeline on three
   videos at two budgets and records the spans. Calibration is cached per GPU name + CPU count.
2. **Accuracy model**: a table of (frames, refine, stride, scorer) → top-1 measured on THUMOS14 (from the
   results files), interpolated in frames. Shipped as data; the scheduler only needs its ordering.
3. **Policy** (`Scheduler(budget_ms, cost, accuracy)`): before the scan, choose the frame budget / refine /
   stride whose predicted latency for a *typical* video fits; after the scan (K and duration known), re-solve
   with the actual K and the windows' lengths, and shrink (drop refinement → drop frames → drop windows) if the
   prediction exceeds the budget, or grow if there is slack. Returns a `KeyframeSampler` configured for this
   video. Logs predicted vs realised ms.
4. **Benchmark** (`benchmarks/eval_scheduler.py`): budgets {100, 125, 150, 200, 300} ms on the 5090; per budget
   report top-1, realised p50 / p90 ms, share of videos within budget, and the prediction error. Baselines:
   the best static configuration that fits the budget on average (hindsight), uniform at the largest frame count
   that fits, and the 009 default ignoring the budget.
5. **Second machine**: repeat calibration + the 150 and 300 ms budgets on the laptop (RTX 3050, CPU decode) to
   show the policy adapts rather than replays the 5090 table. (Small run, 50 videos, if the laptop can hold the
   models; otherwise a CPU-only run on the server with `--device cpu` as the slow-tier stand-in.)
6. Unit tests: cost model fit on synthetic spans; policy picks the max-accuracy config under a budget; shrink
   order; calibration cache key.

## Metrics
Per budget: top-1 vs the hindsight-best static config (target: within 1 point), within-budget rate (≥ 90%),
p90 realised ms ≤ 1.15 × budget, mean absolute prediction error ≤ 15 ms. On the slow tier: the same, with
the chosen configurations visibly different from the 5090's.

## Risks
- Latency on a shared host is noisy (the scan is CPU-bound and other processes compete); the policy should
  use a safety margin (predict for p75, not mean) and the metric should be p90, not max.
- The accuracy table is THUMOS14-specific; the scheduler's *ordering* of configurations is what transfers,
  and that is what the policy uses. Document this.
- Growing per video after the scan can thrash; limit to one re-solve per video.

## Changes made
- `vam/runtime/scheduler.py`: `Config` / `ACCURACY` (measured ordering), `Features` + `features_for`, `CostModel`
  (linear in K, longest window, refinement candidates, frames, refinement flag; non-negative least squares on the
  median of repeated rows; p75 residual as safety margin), `calibrate` (warm-up pass discarded, 3 videos × 4
  configurations × repeats, cached per GPU + CPU key under `$VAM_CACHE`), `Scheduler` (`pre_plan` from duration,
  `post_plan` from the scan, `planner()` callback). `KeyframeSampler(planner=)` applies the post-scan plan.
- `benchmarks/eval_scheduler.py`; tests for the fit, the policy, the cache, the planner hook and a one-off stall.

## Results
- GPU, 200 videos: **0.855 / 0.855 / 0.910 / 0.945 / 0.945 at budgets 100 / 125 / 150 / 200 / 300 ms**, realised
  p50 123 / 124 / 132 / 156 / 158 ms. Uniform at those budgets: 0.720 up to 200, 0.795 at 300. Hindsight-best
  static: 0.855 / 0.855 / 0.935 / 0.940 / 0.940. The scheduler picks 4f at the bottom, 8f at 150 and
  16f+refine/3 from 200 up, without being told frames or strides.
- Within-budget 0.24 / 0.51 / 0.63 / 0.81 / 0.99; p90 180–217 ms. Per-video jitter on the shared host is 55–60
  ms for every configuration, static ones included, so the ≥ 90 % target is only met at 300 ms by anyone.
- Cost model: fitted in 6 s; conservative by 11–16 ms; MAE 36–38 ms (jitter, not bias). The 150 ms loss of 2.5
  points is this conservatism in a band where configurations differ by a few ms of true cost. `--margin 1.1`
  makes it worse (0.875).
- CPU slow tier (40 videos): coefficients 4–50× the GPU's; chooses 16f without refinement at 1 s and refinement
  at 2–4 s; nothing from the 5090 table is replayed.
- The first pass used naive least squares on 12 rows and fitted a 282 ms intercept with zero slopes, which made
  the scheduler pick 4f up to 150 ms. Fixed before the reported run (median of repeats, NNLS, warm-up pass).

## Decision
- `vam.runtime.Scheduler` is the budget-facing entry point; `calibrate` runs once per machine. Margin stays 1.0:
  the p75-residual term already makes predictions conservative, and extra margin costs accuracy in the flat
  125–160 ms band.
- Hypothesis partly confirmed: within 1 point of hindsight at four of five budgets, slow tier adapts, but the
  within-budget and prediction-error targets were set for a quiet machine and this host is not one. Report p50
  and p90 and the hindsight comparison; do not promise a within-budget rate without stating the host's jitter.
- Open: a per-file decode-cost feature (container seek cost) would cut the jitter the model cannot see; and the
  accuracy table is THUMOS14-only, so R3 (generality) is next.
