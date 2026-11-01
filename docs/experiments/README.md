# Experiment log

One entry per experiment or significant change, written **before** the work starts and completed
with results after. Each entry has: Hypothesis, Plan, Metrics, Results, Decision.
Raw numbers live in `docs/results/`; this log is the reasoning trail.

| # | date | title | status | outcome |
|---|---|---|---|---|
| 001 | 2026-10-02 | [UCF101 baseline: frame budget vs accuracy vs latency](001-ucf101-baseline.md) | done | uniform ≈ motion on trimmed clips; CPU decode/preprocess dominate |
| 002 | 2026-10-03 | [THUMOS14 long-video benchmark with oracle ceiling](002-thumos14-long-video.md) | done | 16-pt oracle gap at 4 frames; pixel motion fails; seek-bound decode |
| 003 | 2026-11-01 | [Keyframe-scan two-stage runtime](003-keyframe-runtime.md) | done | kf-clip beats uniform at every budget; kf-oracle = oracle at 8; select ~270 ms |
| 004 | 2026-11-01 | [GPU-side preprocessing for scorer and backbone](004-gpu-preprocessing.md) | in progress | |
