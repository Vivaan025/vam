# vam

**A unified library and runtime for video-action models.**

PyTorch gives you tensors and autograd. Hugging Face gives you models. Nothing gives you the
layer in between for video: deciding which frames to process, reusing computation across time,
and running recognition or action-prediction models on long or live video without the cost
growing linearly with video length. `vam` is that layer.

```python
import vam

video = vam.load_video("clip.mp4")
model = vam.get_model("videomae-base-kinetics")

out = model(video, sampler=vam.MotionSampler(16))
print(out.top_k(3))

for chunk in vam.Stream(0, window=16, hop=8):   # webcam
    print(model(chunk).top_k(1))
```

## Design

```
         vam API            Video · FrameBatch · ActionOutput · Stream
      ───────────────
         Samplers           Uniform · Motion · (Adaptive, learned: research)
      ───────────────
       Model adapters       Hugging Face video classifiers · (policies / VLAs: next)
      ───────────────
         Runtime            Profiler · FeatureCache · (async pipeline, scheduler: research)
      ───────────────
      PyTorch / CUDA        backend, never reimplemented
```

Three rules keep this a library rather than a model zoo:

1. **Models never touch video.** They receive a `FrameBatch` and return an `ActionOutput`.
   The runtime can swap samplers, cache features and stream without any model knowing.
2. **One output type for both meanings of "action model".** Recognition fills `logits`;
   agent policies fill `actions`. Everything above the model works for both.
3. **PyTorch is the backend.** Custom kernels only replace an op after profiling proves it matters.

## Status: v0.1 (Phase 1 of the roadmap)

- [x] `Video` with lazy decoding, `FrameBatch`, `ActionOutput`
- [x] `UniformSampler`, `MotionSampler`
- [x] Hugging Face adapter (VideoMAE, TimeSformer, any `AutoModelForVideoClassification`)
- [x] `Profiler` (wall time + peak GPU memory per stage), `FeatureCache` (LRU, memory budget)
- [x] `Stream` with overlapping windows
- [x] Benchmark harness writing JSONL with GPU name, for cross-machine comparison
- [x] UCF101 dataset + linear-probe evaluation: accuracy vs frame budget vs latency
      (`docs/results/ucf101_phase1.md`)
- [x] THUMOS14 untrimmed long-video benchmark with oracle upper bound and hit-rate metric
- [ ] GPU decode (NVDEC) + GPU preprocessing; one decode pass shared by scorer and model
- [ ] Adaptive frame scheduler (research contribution)
- [ ] Feature reuse in the model forward path
- [ ] Async decode + GPU pipeline
- [ ] Policy / VLA adapter (video-to-action)
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
