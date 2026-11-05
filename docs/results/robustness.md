# Experiment 013 results: robustness and the scheduler's uniform fallback

**Date:** 2026-11-05 · **GPU:** RTX 5090 · tests on synthetic clips (local Windows + server Linux); scheduler
runs on 1,000 Charades test videos and the THUMOS14 val / test sets. Raw rows: `results/charades_scheduler_v2.jsonl`
(flat-score rule only), `charades_scheduler_v3.jsonl` (flat + window-coverage rule), `thumos14_scheduler_v3.jsonl`,
`thumos14_test_scheduler_v2.jsonl`, `thumos14_scheduler_v4.jsonl` (server).

## H1: defined errors, no hangs

| input | before | now |
|---|---|---|
| missing path | `IOError` from OpenCV | `VideoOpenError(path, "no such file")` |
| empty file, random bytes | `IOError` / PyAV exception | `VideoOpenError` |
| truncated mp4 (moov atom cut) | `IOError` or `ValueError` depending on the path | `VideoOpenError` / `DecodeError` / `ScanError` |
| still image renamed `.mp4` | `num_frames = 0`, sampler returned `[]` | one-frame video; sampler answers `[0]`, fallback `short-clip` |
| impossible frame indices | `ValueError` | `DecodeError` |
| scan worker on a corrupt packet | PyAV exception from inside the pool | `ScanError` |
| a hung scan worker | hung forever | `ScanError` after 120 s, pool discarded |

All four exceptions subclass `VamError`; `VideoOpenError` is also an `IOError` and the other two are
`ValueError`s, so existing `except` clauses keep working.

**The hang was real.** Running the new tests on the server deadlocked two pytest sessions: the scan's
process pool used `fork`, and a child forked after thread pools and torch / OpenCV threads existed inherited
their locks. The benchmarks never saw it because they create the pool before any thread. The pool now uses
`forkserver` (workers fork from a clean helper; the first scan pays one import of vam) and every scan has a
timeout. The same test files pass on the server in 2.7 s.

## H2: degenerate inputs give the uniform answer

| input | `last_fallback` | indices |
|---|---|---|
| clip with fewer frames than the budget | `short-clip` | every frame |
| one keyframe in the stream (GOP > clip) | `single-keyframe` | `UniformSampler`'s |
| scorer returns a constant | `flat-scores` | `UniformSampler`'s |
| chosen windows would cover > 50 % of the video | `high-coverage` | `UniformSampler`'s |
| planner says uniform | `planner-uniform` | `UniformSampler`'s |
| scorer raises | – | the exception propagates (a bug, not an input) |

15 tests in `tests/test_robustness.py`; the suite is 62 tests.

## H3: the scheduler on Charades and THUMOS14 with the fallback

| run | rule | budget ms | top-1 | fallbacks | p50 ms |
|---|---|---|---|---|---|
| Charades (012, no fallback) | – | 100 / 150 / 200 | 0.517 / 0.513 / 0.513 | 0 | 83 / 115 / 116 |
| Charades | flat scores only | 100 / 150 / 200 | 0.520 / 0.517 / 0.517 | 71 | 83 / 116 / 117 |
| **Charades** | **flat + coverage > 0.5** | 100 / 150 / 200 | **0.551 / 0.544 / 0.544** | **643** | 61 / 61 / 61 |
| uniform-16 (reference) | – | – | 0.572 | – | 47 |
| THUMOS14 val | flat only | 100 / 150 / 200 | 0.855 / 0.960 / 0.945 | 0 | 126 / 137 / 155 |
| THUMOS14 val | flat + coverage | 150 | 0.920 | 14 | 134 |
| THUMOS14 test | flat only | 150 / 200 | 0.915 / 0.915 | 0 | 143 / 158 |

Per video on Charades with the coverage rule: on the 643 clips that fell back, the scheduler *is* uniform-16
(0.572 on both, same frames). On the 357 clips it kept (longer clips or sparser keyframes, so the windows cover
≤ 50 %), its keyframe configurations score 0.493–0.513 against uniform-16's 0.571 on the same clips. The
residual 2–3-point gap to uniform is entirely those clips: window coverage is a proxy for action coverage, and
on Charades even the low-window-coverage clips are ~90 % action.

On THUMOS14 the coverage rule fires on 14 of 200 videos, all 9–38 s long (the set's shortest), 13 of which are
classified correctly either way. The 150 ms point moved from 0.960 to 0.920 in that run, but the configuration
mix shows why: the re-fitted cost model chose 8f for 158 videos instead of 16f for 175, the calibration
sensitivity already recorded in 010; the fallbacks account for at most 1 of the 8 points.

## Hypothesis check

- H1 ✔: every bad input raises a `VamError` subclass within seconds; no orphan workers after the test session;
  the fork deadlock found on the server is fixed and reproduced by a test.
- H2 ✔: each degenerate input returns `UniformSampler`'s indices and names the reason.
- H3 partly ✔: Charades +3.1 to +3.4 points, within 2–3 points of uniform-16 rather than 1; THUMOS14 unchanged
  within the calibration noise already documented. The flat-score rule alone was not enough (71 of 1,000);
  the window-coverage rule is what carries it (643 of 1,000) and it fires on 7 % of THUMOS14, all short
  videos, without hurting them.
