# 016 · Docs for v0.1: quickstart, API reference, two notebooks (roadmap R7)

**Date:** 2026-11-05 · **Status:** done · **Results:** below

## Motivation
After 015 a stranger can `pip install` the wheel. They still have to read `docs/experiments/` to find out
what to call. This entry writes the three documents a first-time user needs and, where writing them shows the
API is clumsy, adds the small helper that makes the documented path short. Documentation is the last gate to
v0.1 (`docs/roadmap.md`).

## Hypotheses (acceptance criteria)
- **H1, a user reaches a result in five lines.** `docs/quickstart.md`: install, make or bring a clip, run a
  classifier with `UniformSampler`, switch to `KeyframeSampler(8)` (the packaged scorer, no class names), give
  a latency budget in milliseconds, stream, run a policy. Every snippet is executed as written on CPU by a test
  (`tests/test_docs.py` runs the quickstart code blocks that need no weights, and the CI wheel job runs the
  rest).
- **H2, the budget path needs no benchmark code.** Today a millisecond budget takes `calibrate` with a
  hand-written `run` callable plus `Scheduler.planner()` wired into `KeyframeSampler` (see
  `benchmarks/eval_scheduler.py`). `vam.BudgetedSampler(budget_ms, model, calibrate_on=[clips])` wraps that:
  calibrates once (cached per machine), plans per video, exposes `last_plan`. One class, no new behaviour.
- **H3, the API is enumerable.** `docs/api.md` lists every public name in `vam`, `vam.sampling`,
  `vam.runtime`, `vam.models`, `vam.data`, `vam.errors` with its signature, one line of purpose and the
  experiment that justifies it; a test asserts `__all__` of each module is covered.
- **H4, two notebooks run top to bottom.** `examples/01_frame_selection.ipynb` (uniform vs keyframe on a long
  clip, with the profiler's stage breakdown) and `examples/02_latency_budget.ipynb` (calibrate, sweep budgets,
  plot accuracy-vs-ms with the frontier from `docs/results/`). Executed on the server once; outputs committed.

## Plan
1. `vam/runtime/budget.py`: `BudgetedSampler` (H2), exported from `vam`; test with `FakePolicy`-style model.
2. `docs/quickstart.md` (H1), `docs/api.md` (H3), `README.md` trimmed to pitch + install + the three result
   tables + links; the experiment log stays where it is.
3. Notebooks (H4) written as scripts first (`examples/01_frame_selection.py`, `02_latency_budget.py`), converted
   with `jupytext`-style cell markers and executed with `nbconvert` on the server.
4. `docs/study_guide.html` gets a "how to use it" section pointing at the quickstart; `docs/roadmap.md` R7 ✔;
   tag `v0.1.0`.

## Metrics
H1/H3: tests pass. H2: the quickstart's budget example is ≤ 6 lines. H4: `nbconvert --execute` exits 0 on both
notebooks.

## Risks
- Snippet tests that download weights are slow; only the no-weights snippets run in the unit suite, the others
  in the CI wheel job, which already downloads VideoMAE.
- A long clip for the notebooks: THUMOS14 is not redistributable in the repo; the notebook downloads one
  Creative-Commons sports clip or synthesises a 3-minute clip with `examples/make_sample.py --seconds 180`.

## Results
- **H1.** `docs/quickstart.md`: install, synthetic clip, classifier, `UniformSampler` → `KeyframeSampler(8)`,
  `BudgetedSampler(150, ...)`, `Stream`, a policy, the errors. `tests/test_docs.py` runs the snippets that need
  no weights (fake policy, stream, profiler, errors, and the budget example with the oracle scorer) as written;
  the CI wheel job runs the VideoMAE ones through `examples/quickstart.py`.
- **H2.** `vam.BudgetedSampler(budget_ms, model, calibrate_on=[...])` is one line in the quickstart; it wraps
  `calibrate` + `Scheduler` + `KeyframeSampler(planner=...)` with no new behaviour. Writing it exposed that
  `KeyframeSampler(n)` reloaded CLIP on every construction; `default_scorer` now caches one instance per
  (device, model).
- **H3.** `docs/api.md` lists every public name with signature, purpose and originating experiment; the test
  asserts each module's `__all__` is covered (`vam`, `vam.sampling`, `vam.runtime`, `vam.models`, `vam.data`,
  `vam.errors`).
- **H4.** Both notebooks executed top to bottom on the server with `nbconvert` (RTX 5090, THUMOS14 val clips),
  outputs committed. `01_frame_selection`: on a 170 s clip, 24 keyframes scanned, 4 windows; uniform-4 120 ms,
  uniform-16 639 ms, keyframe-4 and keyframe-16 both 143 ms (the 007 result on one video). `02_latency_budget`:
  calibration 6.1 s on 24 timings (intercept 63 ms, 0.42 ms per window frame, 0.17 per candidate, 0.11 per
  model frame, 19 ms margin); over 17 clips the sampler chose 4 frames at 100 ms (p50 101 ms, within 0.41),
  16 frames at 150 ms (p50 110, within 0.76) and 16 frames with stride-3 refinement from 200 ms (p50 133,
  within 0.88; 1.00 at 300 ms): the 010 staircase reproduced from one class. An earlier run of the same notebook
  on the same machine chose 4 frames at 150 ms because its calibration fitted a higher per-model-frame cost; the
  choice at the 150 ms knee is sensitive to a 6 s calibration, which is 010's "within budget" caveat restated.
- Jupyter's runtime, config and matplotlib caches were pointed inside the project folder for the run; the
  scan pool's exit hook now waits for its workers, which removed eight worker tracebacks at kernel exit.

## Decision
R7 done; v0.1.0 is complete on the roadmap's terms pending a green CI run on the first push. Tag `v0.1.0`
after that run. Post-v0.1 research: the trained action-coverage estimator for the scheduler (012/013), and
per-backbone accuracy tables for the scheduler (014).
