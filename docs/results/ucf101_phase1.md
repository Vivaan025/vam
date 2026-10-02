# Phase 1 baseline: UCF101, frame budget vs accuracy vs latency

**Date:** 2026-10-02 · **GPU:** RTX 5090 (32 GB) · **Backbone:** VideoMAE-base (Kinetics-400 pretrained, frozen)
**Head:** linear probe fitted once on uniform-16 features from 2000 train clips (fold 1), then held fixed.
**Test:** 1000 random clips from testlist01 (seed 0), 101 classes. Raw data: `ucf101_phase1_rtx5090.csv`.

| sampler | frames | top-1 | select ms | decode ms | infer ms | total ms |
|---|---|---|---|---|---|---|
| uniform | 2  | 0.811 | 0.0  | 7.0  | 38.3 | 45.3 |
| motion  | 2  | 0.812 | 34.4 | 6.5  | 34.2 | 75.1 |
| uniform | 4  | 0.877 | 0.0  | 12.1 | 27.8 | 39.9 |
| motion  | 4  | 0.873 | 35.1 | 12.2 | 31.6 | 78.9 |
| uniform | 8  | 0.938 | 0.0  | 20.8 | 27.1 | 48.0 |
| motion  | 8  | 0.939 | 36.1 | 20.9 | 24.6 | 81.6 |
| uniform | 16 | 0.960 | 0.0  | 35.1 | 30.0 | 65.1 |
| motion  | 16 | 0.960 | 37.6 | 41.5 | 23.0 | 102.1 |

## Findings

1. **Accuracy saturates quickly on trimmed clips.** 2 frames already give 81% top-1; 8 frames give 94%,
   within 2.2 points of the full 16. UCF101 clips are short (~7 s) and trimmed to the action, so most
   frames are redundant.
2. **Motion-based sampling buys nothing here.** Accuracy is identical to uniform at every budget, and the
   scoring pass costs a flat ~35 ms per clip (a full CPU decode of every frame). On short trimmed video,
   uniform sampling is already near-optimal. Adaptive sampling has to be evaluated on **long, untrimmed**
   video where the action occupies a small fraction of the timeline.
3. **The GPU is not the bottleneck.** Pure model forward on this GPU is ~7 ms (see bench_sampling).
   The 25–38 ms "infer" column is dominated by the Hugging Face image processor resizing and normalising
   frames on the CPU. Decode scales linearly with frames (7 → 35 ms). The runtime work in Phase 2 is
   therefore: GPU decode, GPU preprocessing, and a single shared decode pass for scoring + inference.
4. **Compute does not scale with the frame budget yet.** VideoMAE needs exactly 16 frames, so smaller
   budgets are padded to 16: the budget changes information, not FLOPs. Making compute follow the budget
   requires variable-length temporal input or token dropping inside the transformer. That moves the
   research question from *which frames* to *which tokens*.

## Implications for the plan

- Build a **long-video benchmark** (concatenated UCF101 clips with distractor segments, or an untrimmed
  dataset such as THUMOS14 / ActivityNet) where frame selection can actually matter.
- Phase 2 targets: GPU decode (NVDEC), GPU preprocessing, one decode pass shared by scorer and model,
  async pipeline. Success metric: motion/adaptive total latency ≤ uniform total latency at equal budget.
- Phase 3 adds token-level sparsity so FLOPs track the budget, then the adaptive scheduler trades
  tokens for accuracy under a millisecond budget.
