# vam

**A unified library and runtime for video-action models.**

PyTorch gives you tensors and autograd. Hugging Face gives you models. Nothing gives you the
layer in between for video: deciding which frames to process, reusing computation across time,
and running recognition or action-prediction models on long or live video without the cost
growing linearly with video length. `vam` is that layer.

```bash
pip install vam-runtime[hf]        # import name: vam · Python 3.10–3.12 · Linux, Windows, macOS (CPU)
```

```python
import vam

video = vam.load_video("clip.mp4")
model = vam.get_model("videomae-base-kinetics")

out = model(video, sampler=vam.KeyframeSampler(8))        # keyframe scan + packaged class-free scorer
print(out.top_k(3))

sampler = vam.BudgetedSampler(150, model, calibrate_on=["a.mp4", "b.mp4", "c.mp4"])
out = model(video, sampler=sampler)                        # frames, refinement and stride chosen for 150 ms

for chunk in vam.Stream(0, window=16, hop=8):              # webcam
    print(model(chunk).top_k(1))
```

[Quickstart](https://github.com/Vivaan025/vam/blob/main/docs/quickstart.md) · [API reference](https://github.com/Vivaan025/vam/blob/main/docs/api.md) · [Results](https://github.com/Vivaan025/vam/tree/main/docs/results/) ·
[Experiment log](https://github.com/Vivaan025/vam/blob/main/docs/experiments/README.md) · [Roadmap](https://github.com/Vivaan025/vam/blob/main/docs/roadmap.md)

## Design

```
         vam API            Video · FrameBatch · ActionOutput · Stream
      ───────────────
         Samplers           Uniform · Motion · Keyframe (learned scorer) · Budgeted (ms)
      ───────────────
       Model adapters       Hugging Face · torchvision · policies (OpenVLA)
      ───────────────
         Runtime            keyframe scan · parallel window decode · cost model + scheduler · Profiler · FeatureCache
      ───────────────
      PyTorch / CUDA        backend, never reimplemented
```

Three rules keep this a library rather than a model zoo:

1. **Models never touch video.** They receive a `FrameBatch` and return an `ActionOutput`.
   The runtime can swap samplers, cache features and stream without any model knowing.
2. **One output type for both meanings of "action model".** Recognition fills `logits`;
   agent policies fill `actions`. Everything above the model works for both.
3. **PyTorch is the backend.** Custom kernels only replace an op after profiling proves it matters.

## Status: v0.1.0 · wheel builds and installs on Linux and Windows (CI); see `docs/experiments/` for the log

On 200 untrimmed THUMOS14 videos (RTX 5090, frozen VideoMAE-base + a linear probe fitted on UCF101):

| policy | 4 frames | 8 frames | 16 frames |
|---|---|---|---|
| uniform sampling | 0.720 @ 147 ms | 0.795 @ 277 ms | 0.880 @ 486 ms |
| **vam** (keyframe scan + learned scorer + refinement) | **0.865 @ 143 ms** | **0.925 @ 153 ms** | **0.940 @ 152 ms** |
| oracle (sees the annotations) | 0.890 | 0.925 | 0.955 |

Given a budget in **milliseconds** instead of frames, the scheduler calibrates a cost model on the machine in
6 s and picks frames, refinement and stride itself: 0.855 at 100 ms, 0.960 at 150, 0.945 at 200 and above
(`docs/results/thumos14_scheduler.md`). On a CPU-only host it picks different configurations at every budget.

On the 212 **held-out** THUMOS14 test videos, which no head was trained on, the same runtime reaches 0.863 / 0.901 /
0.915 at 4 / 8 / 16 frames in 149 / 167 / 165 ms, against uniform 0.764 / 0.816 / 0.858 in 156 / 278 / 503 ms and the
oracle's 0.863 / 0.873 / 0.925 (`docs/results/thumos14_generality.md`). Resolution and codec (H.264, HEVC, VP9) do not
change accuracy; keyframe density changes the decode cost of every sampler, and the scheduler follows it.

**Where it does not help:** on Charades, where the action covers 95 % of every 30 s clip, even the oracle is no
better than uniform sampling and neither is `vam` (`docs/results/charades.md`). Frame selection pays when actions
are a small fraction of the video. Since 013 the scheduler falls back to uniform frames when its scan carries no
information (one keyframe, flat scores, windows that would cover most of the clip). The sampler cannot detect this case from the video itself (019): on footage where the action fills the clip, use `UniformSampler` or `OCSampler`; `BudgetedSampler` falls back to uniform on degenerate scans, not on high coverage.

**The same runtime serves other models.** With torchvision's R(2+1)D-18 in place of VideoMAE the selection gain is
the same size (0.655 / 0.675 / 0.700 vs uniform 0.565 / 0.520 / 0.545 at 4 / 8 / 16 frames, ~145 ms vs 135–498 ms),
and OpenVLA-7B, a robot policy, runs through `Stream` as an `ActionPolicy` at 113 ms a step with its actions tracking
the recorded ones on BridgeData (Δxyz correlation 0.97) and the scheduler's cost model fitting it unchanged
(`docs/results/model_coverage.md`).


- [x] `Video` with lazy decoding, `FrameBatch`, `ActionOutput`
- [x] `UniformSampler`, `MotionSampler`
- [x] Hugging Face adapter (VideoMAE, TimeSformer, any `AutoModelForVideoClassification`)
- [x] torchvision adapter (R(2+1)D-18, R3D-18, MViT-v2-S) and `ActionPolicy` + OpenVLA-7B (`docs/results/model_coverage.md`)
- [x] `Profiler` (wall time + peak GPU memory per stage), `FeatureCache` (LRU, memory budget)
- [x] `Stream` with overlapping windows
- [x] Benchmark harness writing JSONL with GPU name, for cross-machine comparison
- [x] UCF101 dataset + linear-probe evaluation: accuracy vs frame budget vs latency
      (`docs/results/ucf101_phase1.md`)
- [x] THUMOS14 untrimmed long-video benchmark with oracle upper bound and hit-rate metric
- [x] Keyframe-scan two-stage sampler with oracle / CLIP scorers (`docs/results/thumos14_keyframe.md`)
- [x] Device-side tensor preprocessing (`docs/results/thumos14_preprocess.md`)
- [x] Process-parallel keyframe scan (`docs/results/thumos14_scan.md`)
- [x] Gap window, score-proportional allocation, coarse-to-fine refinement inside windows
      (`docs/results/thumos14_adaptive.md`)
- [x] Parallel windowed decode: latency flat at ~150 ms for any budget (`docs/results/thumos14_pdecode.md`)
- [x] Learned, class-free keyframe scorer: equals the oracle at 8 frames (`docs/results/thumos14_learned.md`)
- [x] Millisecond-budget scheduler: calibrated cost model + pre/post-scan policy (`docs/results/thumos14_scheduler.md`)
- [x] Generality within domain: held-out split, 1080p, HEVC, VP9, keyframe density (`docs/results/thumos14_generality.md`)
- [x] Second domain: Charades, a negative result that states the method's boundary (`docs/results/charades.md`)
- [x] Robustness: documented errors, no hangs, degenerate inputs answer uniform, scheduler fallback (`docs/results/robustness.md`)
- [x] Standard benchmarks: ActivityNet-1.3 and Mini-Kinetics under the FrameExit protocol, FrameExit reproduced from its weights (`docs/results/activitynet_017.md`)
- [x] Prior samplers on one machine: OCSampler, AR-Net, SCSampler ported from released weights and reproduced; `OCSampler` is a `vam` sampler (`docs/results/activitynet_018.md`)
- [x] Coverage-aware routing, negative: action coverage is not predictable from the scan or the scorer's scores, so the sampler cannot detect high-coverage footage itself (`docs/experiments/019-coverage-router.md`)
- [x] Packaging: `vam-runtime` wheel with the packaged scorer, bounded dependency ranges, CI on Linux + Windows (015)
- [x] Docs: quickstart, API reference, `BudgetedSampler`, executable snippets (016)
- [ ] GPU windowed decode (NVDEC) — deferred past v0.1
- [ ] Feature reuse in the model forward path
- [ ] Async decode + GPU pipeline
- [ ] Custom kernels where profiling justifies them

## Research question

> How far can adaptive frame scheduling and feature reuse reduce the compute of a
> video-action model on long video before accuracy drops, and does the scheduling
> overhead eat the savings?

Deliverable: an accuracy-vs-compute trade-off curve across samplers, budgets and two GPU
tiers (RTX 3050 4 GB laptop, RTX 5090), with ablations and open benchmark scripts.

## Setup

```bash
pip install -e ".[hf,dev]"
pytest
python benchmarks/bench_sampling.py path/to/clip.mp4 --budgets 4 8 16
```

## Roadmap

| Phase | Weeks | Goal |
|---|---|---|
| 1 Baseline | 1–2 | Profile a pretrained model end to end on short and long clips |
| 2 Library | 3–5 | Clean API, samplers, reproducible benchmarks (this release) |
| 3 Research | 6–10 | Adaptive scheduler; accuracy vs compute; failure cases |
| 4 GPU | 11–14 | Profile, then optimise one operator in CUDA/Triton if justified |
| 5 Artifact | 15–18 | Ablations, write-up, release |
