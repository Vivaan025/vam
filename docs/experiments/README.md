# Experiment log

One entry per experiment or significant change, written **before** the work starts and completed
with results after. Each entry has: Hypothesis, Plan, Metrics, Results, Decision.
Raw numbers live in `docs/results/`; this log is the reasoning trail.

| # | date | title | status | outcome |
|---|---|---|---|---|
| 001 | 2026-10-02 | [UCF101 baseline: frame budget vs accuracy vs latency](001-ucf101-baseline.md) | done | uniform ≈ motion on trimmed clips; CPU decode/preprocess dominate |
| 002 | 2026-10-03 | [THUMOS14 long-video benchmark with oracle ceiling](002-thumos14-long-video.md) | done | 16-pt oracle gap at 4 frames; pixel motion fails; seek-bound decode |
| 003 | 2026-11-01 | [Keyframe-scan two-stage runtime](003-keyframe-runtime.md) | done | kf-clip beats uniform at every budget; kf-oracle = oracle at 8; select ~270 ms |
| 004 | 2026-11-01 | [GPU-side preprocessing for scorer and backbone](004-gpu-preprocessing.md) | done | infer 300→20 ms; kf-clip-8 beats uniform-16 on latency at equal accuracy; kf-clip-16 wins both axes |
| 005 | 2026-11-01 | [Cheaper keyframe scan](005-cheaper-keyframe-scan.md) | done | select 160→46 ms; kf-clip dominates uniform at 8 and 16 frames on both axes, ties latency at 4 |
| 006 | 2026-11-01 | [Adaptive window and coarse-to-fine refinement](006-adaptive-window.md) | done | gap window default (+2 pts at 4 frames, free); refinement reaches 0.865 / 0.900 at 4 / 8 but costs +67 / +148 ms (OpenCV seek 35–60 ms per window): opt-in until 007 |
| 007 | 2026-11-02 | [Parallel windowed decode](007-parallel-window-decode.md) | planned | |
