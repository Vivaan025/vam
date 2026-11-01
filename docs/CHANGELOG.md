# Changelog

Newest first. One line per change; experiments link to `docs/experiments/`.

| date | change | where | experiment |
|---|---|---|---|
| 2026-11-02 | `KeyframeSampler` options `window="gap"`, `alloc="softmax"` (+`temperature`), `refine=scorer` (+`refine_stride`): coarse-to-fine pass keeps the decoded window frames and `Sampler.read` hands them over; `Video.iter_frames(step)`; benchmark flags `--kf-window/--kf-alloc/--kf-temp/--kf-refine/--tag`; 4 tests | `vam/sampling/keyframe.py`, `vam/sampling/base.py`, `vam/core.py`, `benchmarks/eval_thumos14.py`, `tests/test_keyframe.py` | [006](experiments/006-adaptive-window.md) |
| 2026-11-01 | Plan for experiment 006 (adaptive window, coarse-to-fine) | `docs/experiments/006-adaptive-window.md` | [006](experiments/006-adaptive-window.md) |
| 2026-11-01 | Results of experiment 005 | `docs/results/thumos14_scan.md` | [005](experiments/005-cheaper-keyframe-scan.md) |
| 2026-11-01 | Process-parallel keyframe scan (`workers`, `stride`, `backend`), persistent pool; sampler and benchmark flags | `vam/runtime/scan.py`, `vam/sampling/keyframe.py`, `benchmarks/eval_thumos14.py` | [005](experiments/005-cheaper-keyframe-scan.md) |
| 2026-11-01 | Plan for experiment 005 (cheaper keyframe scan) | `docs/experiments/005-cheaper-keyframe-scan.md` | [005](experiments/005-cheaper-keyframe-scan.md) |
| 2026-11-01 | Results of experiment 004 | `docs/results/thumos14_preprocess.md` | [004](experiments/004-gpu-preprocessing.md) |
| 2026-11-01 | `vam.runtime.preprocess.Preprocessor`: device-side resize/crop/normalise read from the HF processor config; used by `HFVideoClassifier`, `HFFeatureExtractor` and `ClipScorer` (`use_hf_processor=True` keeps the old path); tolerance tests vs both HF processors | `vam/runtime/preprocess.py`, `vam/models/hf.py`, `vam/models/probe.py`, `vam/sampling/keyframe.py` | [004](experiments/004-gpu-preprocessing.md) |
| 2026-11-01 | Experiment log and changelog introduced; entries 001–003 written retroactively | `docs/experiments/`, `docs/CHANGELOG.md` | |
| 2026-11-01 | Results of the keyframe runtime on THUMOS14 | `docs/results/thumos14_keyframe.md` | [003](experiments/003-keyframe-runtime.md) |
| 2026-11-01 | `ClipScorer` uses `text_model`/`vision_model` + projections (transformers 4.x and 5.x) | `vam/sampling/keyframe.py` | 003 |
| 2026-11-01 | `--device` flag for `eval_thumos14.py` | `benchmarks/eval_thumos14.py` | 003 |
| 2026-11-01 | Keyframe scan (`vam.runtime.scan`), `KeyframeSampler`, oracle/CLIP/uniform scorers, sequential grouped `Video.read`, `av` dependency | `vam/runtime/scan.py`, `vam/sampling/keyframe.py`, `vam/core.py` | 003 |
| 2026-10-31 | Study guide and pitch | `docs/study_guide.html` | |
| 2026-10-31 | THUMOS14 long-video results written up | `docs/results/thumos14_phase2.md` | [002](experiments/002-thumos14-long-video.md) |
| 2026-10-03 | THUMOS14 dataset loader, annotated-only downloader, long-video benchmark with oracle sampler and hit rate | `vam/data/thumos14.py`, `scripts/download_thumos14.py`, `benchmarks/eval_thumos14.py` | 002 |
| 2026-10-02 | UCF101 results written up | `docs/results/ucf101_phase1.md` | [001](experiments/001-ucf101-baseline.md) |
| 2026-10-02 | UCF101 loader, linear probe, accuracy-vs-compute evaluation, summariser | `vam/data/ucf101.py`, `vam/models/probe.py`, `benchmarks/` | 001 |
| 2026-10-02 | Profiler probes CUDA once and falls back to CPU timing on an occupied GPU | `vam/runtime/profiler.py` | |
| 2026-10-02 | v0.1 scaffold: core types, uniform/motion samplers, HF adapter, profiler, feature cache, streaming, tests | `vam/` | |
