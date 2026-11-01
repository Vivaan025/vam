# 001 · UCF101 baseline: frame budget vs accuracy vs latency

**Date:** 2026-10-02 · **Status:** done · **Results:** [`docs/results/ucf101_phase1.md`](../results/ucf101_phase1.md)
*(written retroactively on 2026-11-01 when the experiment log was introduced)*

## Hypothesis
Motion-based frame sampling preserves action-recognition accuracy at lower frame budgets than uniform
sampling, and the GPU forward pass is the dominant cost.

## Plan
Frozen VideoMAE-base, linear probe fitted once on 2,000 UCF101 train clips (uniform-16 features), held
fixed. Evaluate uniform vs motion samplers at 2/4/8/16 frames on 1,000 test clips, logging per-stage
wall time and peak GPU memory. RTX 5090.

## Metrics
Top-1; select / decode / infer ms per clip; peak MB.

## Results
Top-1 identical for both samplers at every budget (0.81 → 0.96 from 2 → 16 frames). Motion scoring costs
a flat ~35 ms. Pure GPU forward ~7 ms; the rest is CPU decode and HF preprocessing. VideoMAE pads
budgets to 16 frames so FLOPs do not follow the budget.

## Decision
Both halves of the hypothesis were wrong on trimmed clips. Stop iterating on pixel-motion samplers.
Move to untrimmed long video (002), and treat decode/preprocessing as the runtime target.
