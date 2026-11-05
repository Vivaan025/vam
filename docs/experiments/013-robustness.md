# 013 · Robustness: defined errors, degenerate inputs, scheduler fallback

**Date:** 2026-11-05 · **Status:** planned · **Results:** *(to fill)*

## Motivation (roadmap R4, and 012)
Every benchmark so far ran on files the harness chose. A library meets files it did not choose: truncated
downloads, images renamed `.mp4`, streams with one keyframe, clips shorter than a window, VP9 in a container
the scan has not seen. Today these surface as `IOError`, `ValueError`, a hung pool or a wrong answer. 012 added
a policy-level failure: on a clip where the scan's scores are flat (everything is action, or nothing is) the
scheduler still picks a keyframe configuration and loses to uniform sampling.

## Hypotheses
- **H1.** Every failure in the decode / scan / select / read path can be raised as one of three documented
  exceptions (`VideoOpenError`, `DecodeError`, `ScanError`, all subclasses of `VamError`) with the path in the
  message, and no input hangs the process or leaks a worker.
- **H2.** Degenerate inputs give the *uniform* answer rather than an error or a worse answer: a stream with one
  keyframe, a clip shorter than `min_window_s`, a video with fewer frames than the budget, a scorer that returns
  a constant. The runtime's accuracy on such inputs equals uniform's by construction.
- **H3.** A scheduler fallback that reads the scan (keyframe count, score spread) and chooses `uniform` when the
  scan carries no information recovers uniform's accuracy on Charades (0.572 at 16 frames vs 0.513 for the
  scheduler in 012) without changing any THUMOS14 number (the scan is informative there: 25 keyframes, score
  spread high).

## Plan
1. `vam/errors.py`: `VamError`, `VideoOpenError`, `DecodeError`, `ScanError`; `core.py`, `runtime/decode.py`,
   `runtime/scan.py` raise them; `IOError` / `ValueError` call sites updated; the process pool registers an
   `atexit` shutdown and a `shutdown_pools()` function.
2. `KeyframeSampler`: `fallback="uniform"` (default) when the scan has < 2 keyframes or the scores are flat
   (`std(scores) < flat_eps · |mean|`, default 1e-3 relative) or the video is shorter than one window;
   `last_fallback` records why. Windows and allocation clamp to the video length and to ≥ 1 frame.
3. `Scheduler`: `Config(frames, refine, stride, uniform=False)`; the accuracy table gains uniform rows (THUMOS14
   uniform 0.720 / 0.795 / 0.880); `post_plan(K, windows, flat)` picks a uniform configuration when `flat`;
   the planner protocol becomes `(kf, scores, chosen, wins) -> Plan-like (frames, refine, stride, uniform)`.
   The benchmark logs `fallback`.
4. Tests on synthetic files: a 4 KB truncated mp4, a PNG renamed `.mp4`, an empty file, a 1 s single-keyframe
   clip, a 0.3 s clip, a 10-frame clip with a 16-frame budget, a constant scorer, a scorer that raises; pool
   shutdown leaves no child processes. Each asserts the exception type or the uniform indices.
5. Re-run on the server: Charades scheduler at 100 / 150 / 200 ms with the fallback (expect ≈ uniform-16), and
   THUMOS14 val at 150 / 200 ms (expect unchanged, 0 fallbacks).

## Metrics
H1: every synthetic bad input raises the documented type within 5 s; `pgrep` after the test session shows no
orphan workers. H2: indices equal `UniformSampler`'s on each degenerate input. H3: Charades scheduler top-1 within
1 point of uniform-16; THUMOS14 scheduler unchanged within noise and `fallback` count 0.

## Risks
- A relative flatness threshold may fire on legitimately uniform scores (a long video that is all crowd); then
  uniform is the right answer anyway. The risk is the other direction: too strict a threshold leaves Charades
  unchanged. Report the fallback rate on both datasets.
- OpenCV and PyAV disagree on what a truncated file is; test both paths.

## Results
*(to fill)*

## Decision
*(to fill)*
