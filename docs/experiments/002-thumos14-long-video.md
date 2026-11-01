# 002 · THUMOS14 long-video benchmark with oracle ceiling

**Date:** 2026-10-03 · **Status:** done · **Results:** [`docs/results/thumos14_phase2.md`](../results/thumos14_phase2.md)
*(written retroactively on 2026-11-01 when the experiment log was introduced)*

## Hypothesis
On untrimmed video where the action is a small fraction of the timeline, frame selection materially
changes accuracy at a fixed budget.

## Plan
Same backbone and fixed UCF101 head, restricted to the 20 THUMOS classes (no training on THUMOS).
200 annotated validation videos (~3 min each). Samplers: uniform, motion, and an *oracle* that samples
uniformly inside annotated segments (upper bound). Budgets 4/8/16/32. Log hit rate (share of selected
frames inside an action) and per-stage latency.

## Metrics
Top-1 (correct if argmax ∈ annotated classes); hit rate; select / decode / infer ms.

## Results
Oracle beats uniform by 16 points at 4 frames (0.89 vs 0.73), 13 at 8, 7 at 16. Oracle-4 beats
uniform-16. Uniform's hit rate is ~30% regardless of budget. Pixel-motion scoring is *worse* than uniform
and costs 1.5 s/video. Random-access decode of 16 frames costs ~500 ms (keyframe seeking).

## Decision
Hypothesis confirmed; the headroom is measured. The scorer must be semantic, and the runtime must make
scoring nearly free. Design the keyframe-scan runtime (003).
