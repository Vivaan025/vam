# Experiment 010 results: millisecond-budget scheduler

**Date:** 2026-11-03 · **GPU:** RTX 5090 (24-core host, shared) · setup identical to `thumos14_learned.md` (200
THUMOS14 val videos, VideoMAE-base + UCF101 linear probe, learned context scorer with per-fold heads, gap window,
softmax allocation, thread-pool window decode, CLIP fp16). Slow tier: the same host with `--device cpu`, first 40
videos. Raw per-budget rows: `thumos14_scheduler_rtx5090.csv`. Benchmark: `benchmarks/eval_scheduler.py`.

The scheduler is given a budget in milliseconds and nothing else. Before the scan it picks the most accurate
configuration (frames ∈ {4, 8, 16}, refinement on/off, stride ∈ {3, 6}) whose predicted latency fits for a typical
video; after the keyframe scan it re-solves once with the real keyframe count and window lengths. The cost model is
fitted on the target machine in 6 s (3 videos × 4 configurations × 3 repeats after a discarded warm-up pass).

## Cost model as fitted

| machine | fit time | intercept | per window frame | per refinement candidate | per model frame | per refinement | safety margin (p75 resid) |
|---|---|---|---|---|---|---|---|
| RTX 5090 | 6.1 s | 63.9 ms | 0.43 ms | 0.15 ms | 0.14 ms | 0 | 20.6 ms |
| CPU only | 51 s | 289 ms | 1.9 ms | 7.5 ms | 0 | 46 ms | 10.7 ms |

The keyframe-count term fits to zero on both machines (the scan is parallel and its cost hides under the
intercept). On the GPU the model frame term is 0.14 ms: between 4 and 16 frames the pipeline's cost barely moves,
which is the 007 finding restated as a coefficient.

## GPU, 200 videos (margin 1.0)

| budget | chosen configurations | top-1 | within budget | p50 ms | p90 ms | prediction error mean / MAE | hindsight-best static (009) |
|---|---|---|---|---|---|---|---|
| 100 | 4f ×200 | 0.855 | 0.24 | 123 | 180 | −16 / 37 | nothing fits; 4f 0.855 (mean 132, within 0.24) |
| 125 | 4f ×192, 16f(+refine) ×7, 8f ×1 | 0.855 | 0.51 | 124 | 182 | −16 / 38 | nothing fits; 4f 0.855 (within 0.48) |
| 150 | 8f ×171, 16f ×11, 16f+refine/3 ×10, 16f+refine/6 ×8 | 0.910 | 0.63 | 132 | 189 | −11 / 36 | 16f 0.935 (mean 142, within 0.60) |
| 200 | 16f+refine/3 ×200 | **0.945** | 0.81 | 156 | 214 | −14 / 38 | 16f+refine/6 0.940 (mean 152, within 0.86) |
| 300 | 16f+refine/3 ×200 | **0.945** | 0.99 | 158 | 217 | −13 / 38 | 16f+refine/6 0.940 (within 0.99) |

Uniform sampling at the same budgets: 0.720 (uniform-4, 147 ms) up to 200 ms, 0.795 (uniform-8, 277 ms) at 300.
Prediction error is realised minus predicted: negative means the model is conservative. Post-scan re-planning
changed the pre-scan choice for 29 videos at 150 ms and for 8 at 125 ms; never at 100, 200 or 300.

With `--margin 1.1` (predictions inflated by 10 % before the comparison): 150 ms drops to 0.875 (4f ×164,
16f variants ×36, within 0.73, p50 124); the other budgets are unchanged.

## CPU only, 40 videos

| budget | chosen configurations | top-1 | within budget | p50 ms | p90 ms | prediction error mean / MAE |
|---|---|---|---|---|---|---|
| 1000 | 16f ×33, 16f+refine ×7 | 0.950 | 0.80 | 655 | 1253 | +129 / 228 |
| 2000 | 16f+refine/3 ×26, 16f+refine/6 ×14 | 0.900 | 0.82 | 1702 | 2219 | +248 / 295 |
| 4000 | 16f+refine/3 ×40 | 0.900 | 1.00 | 1908 | 2273 | +150 / 233 |

Per configuration on CPU: 16f realised p50 607 ms, 16f+refine/6 1413 ms, 16f+refine/3 1870 ms. The policy
therefore changes shape with the machine: refinement, free on the GPU, is a 2–3× cost here and is only chosen when
the budget allows it. (The accuracy drop from 1000 to 2000 ms is two videos out of 40, inside noise; the
accuracy table the scheduler orders by comes from the 200-video GPU runs.)

## Findings

1. **The scheduler walks the frontier in the right direction without being told frames or strides.** 0.855 at
   100 ms, 0.910 at 150, 0.945 at 200 and above, versus uniform's 0.720 / 0.720 / 0.795 at the same budgets. At
   200 ms it picks 16f+refine/3, the best configuration the runtime has produced (0.945, 414 videos across runs).
2. **At 150 ms it loses 2.5 points to the hindsight-best static configuration.** The cost model is conservative
   by 11–16 ms (margin included), and in the 125–160 ms band the configurations differ by only a few ms of
   realised cost, so a conservative model trades 16f (0.935, 142 ms) for 8f (0.905, 138 ms). A 10 % margin
   makes this worse (0.875), not better.
3. **Within-budget rate is bounded by the host, not the policy.** Per-video latency on this shared machine
   has a p90 − p50 spread of 55–60 ms for *every* configuration, static or scheduled (the static 16f+refine/6
   also stays within 200 ms only 86 % of the time). Budgets below the ~120 ms floor cannot be met by any
   configuration; above 250 ms everything fits. The mean absolute prediction error of 36–38 ms is this jitter,
   not bias.
4. **Calibration must be robust to one-off stalls.** The first pass (same code, naive least squares on 12 rows)
   fitted an intercept of 282 ms with every slope clamped to zero, chose 4f at every budget up to 150, and was
   indistinguishable from "no scheduler" at those budgets. Median of repeats, non-negative least squares and a
   discarded warm-up pass fixed it (commit 6d5c5bf); the per-machine cache must be cleared when the fit changes.
5. **The cost model transfers as a procedure, not as numbers.** On CPU the fitted coefficients are 4–50× the
   GPU's and the chosen configurations differ at every budget; nothing from the 5090 table is replayed.

## Experiment 010 hypothesis check

- Top-1 within 1 point of the hindsight-best static configuration: ✔ at 100, 125, 200, 300 ms; ✗ at 150 (−2.5).
- Within-budget rate ≥ 90 %: ✗ except at 300 ms (0.99); no static configuration reaches it either below 250 ms
  on this host (finding 3).
- p90 ≤ 1.15 × budget: ✔ at 200 (214 ≤ 230) and 300; ✗ at 150 (189).
- Mean absolute prediction error ≤ 15 ms: ✗ (36–38 ms, host jitter); bias is −11 to −16 ms.
- Slow tier chooses visibly different configurations: ✔.
