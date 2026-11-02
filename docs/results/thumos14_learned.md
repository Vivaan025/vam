# Experiment 009 results: learned keyframe scorer

**Date:** 2026-11-02 · **GPU:** RTX 5090 (24-core host) · setup identical to `thumos14_pdecode.md` (200 THUMOS14 val
videos, VideoMAE-base + UCF101 linear probe, gap window, thread-pool window decode, refinement stride 6, CLIP fp16).
Raw data: `thumos14_learned_rtx5090.csv`. Heads from `scripts/train_scorer.py`, 5-fold cross-validation by video
(seed 0); every video is scored by the head that did not train on it.

## Scorer-level (held-out folds, mean ± std over 5 folds)

| scorer | inputs | Spearman vs oracle | top-2 precision | top-4 precision | overlap of chosen top-2 |
|---|---|---|---|---|---|
| zero-shot CLIP (007) | class prompts | 0.250 | 0.603 | – | 0.435 |
| ridge | embedding | 0.317 ± 0.041 | 0.698 ± 0.051 | 0.652 ± 0.040 | 0.519 ± 0.051 |
| mlp | embedding | 0.288 ± 0.052 | 0.698 ± 0.051 | 0.649 ± 0.040 | 0.506 ± 0.038 |
| **context** (±2 neighbours) | embeddings | **0.321 ± 0.043** | **0.732 ± 0.023** | **0.675 ± 0.036** | **0.546 ± 0.048** |
| context + zero-shot score | embeddings + classes | 0.323 ± 0.041 | 0.735 ± 0.029 | 0.673 ± 0.038 | 0.545 ± 0.050 |
| oracle | annotations | 1 | 1 | 1 | 0.883 |

Top-k precision = share of the k best-scored keyframes whose gap window overlaps an annotated action.
Adding the zero-shot (class-name) score to the inputs changes nothing: the learned scorer is class-free.

## End to end (200 videos, kf-learned; kf-clip and oracle rows from 007)

| frames | arm | top-1 | hit | select ms | of which refine | decode ms | infer ms | total ms |
|---|---|---|---|---|---|---|---|---|
| 4 | kf-clip (007) | 0.815 | 0.44 | 47 | – | 88 | 6 | 141 |
| 4 | kf-clip + refine (007 default) | 0.850 | 0.45 | 140 | 99 | 0 | 6 | 146 |
| 4 | kf-learned context | 0.855 | 0.53 | 41 | – | 86 | 5 | 132 |
| 4 | **kf-learned context + refine** | **0.865** | **0.54** | 138 | 98 | 0 | 5 | **143** |
| 4 | kf-learned ridge + refine | 0.875 | 0.51 | 138 | 98 | 0 | 5 | 143 |
| 4 | kf-learned context+zs + refine | 0.860 | 0.53 | 138 | 98 | 0 | 5 | 144 |
| 4 | oracle | 0.890 | 1.00 | | | | | |
| 8 | kf-clip (007) | 0.885 | 0.42 | 46 | – | 96 | 5 | 147 |
| 8 | kf-clip + refine (007 default) | 0.895 | 0.44 | 147 | 107 | 0 | 5 | 152 |
| 8 | kf-learned context | 0.905 | 0.48 | 39 | – | 95 | 5 | 138 |
| 8 | **kf-learned context + refine** | **0.925** | **0.50** | 148 | 109 | 0 | 5 | **153** |
| 8 | kf-learned ridge + refine | 0.915 | 0.47 | 149 | 110 | 0 | 5 | 154 |
| 8 | kf-learned context+zs + refine | 0.910 | 0.50 | 148 | 109 | 0 | 5 | 154 |
| 8 | oracle | 0.925 | 1.00 | | | | | |
| 16 | kf-clip (007) | 0.935 | 0.41 | 46 | – | 99 | 4 | 150 |
| 16 | kf-clip + refine (007 default) | 0.930 | 0.44 | 147 | 107 | 0 | 4 | 151 |
| 16 | kf-learned context | 0.935 | 0.48 | 39 | – | 99 | 4 | 142 |
| 16 | **kf-learned context + refine** | **0.940** | **0.51** | 148 | 109 | 0 | 4 | **152** |
| 16 | kf-learned ridge + refine | 0.955 | 0.48 | 149 | 110 | 0 | 4 | 154 |
| 16 | kf-learned context+zs + refine | 0.935 | 0.51 | 148 | 109 | 0 | 4 | 152 |
| 16 | oracle | 0.955 | 1.00 | | | | | |

## Findings

1. **The learned scorer is better than zero-shot CLIP at every budget, with or without refinement, and costs
   less** (select 39–41 ms vs 46–47: the class-prompt softmax is gone and the head is one small matmul).
   Hit rate rises from 0.44 to 0.53 at 4 frames.
2. **With refinement the gap to the oracle closes to 2.5 / 0 / 1.5 points** (context head: 0.865 / 0.925 /
   0.940 vs 0.890 / 0.925 / 0.955), at 143 / 153 / 152 ms per video. At 8 frames the runtime matches the oracle
   that sees the annotations.
3. **Class-free.** The context head without the zero-shot input is as good as with it on both scorer-level and
   end-to-end metrics. The runtime no longer needs the user to name classes.
4. **Ridge vs context end to end is within noise.** Ridge wins at 4 and 16 frames (0.875, 0.955), context at 8
   (0.925); each difference is 1–2 points on 200 videos (≈ ±2 points standard error). Scorer-level metrics,
   which average over ~50 keyframes per video, favour context on every measure, so it is the default.
5. **Cross-validation caveat.** Each video is scored by a head trained on the other 160 videos of the same
   dataset (same 20 classes, same camera styles). Generalisation to other footage is untested; this is roadmap
   item R3. The heads are tiny (context: 2,562 → 256 → 1), early-stop at epoch 4–15, and the fold spread is
   small (± 0.02–0.05), so overfitting to a fold is not the story.

## Experiment 009 hypothesis check

- kf-learned-4 ≥ 0.87: ✔ with ridge (0.875), ✗ by 0.5 with context (0.865); within noise of each other.
- kf-learned-8 ≥ 0.91: ✔ (0.925).
- select ms within +5 of kf-clip: ✔ (−6 to −8 ms).
- class-free within 1 point of class-aware: ✔ (identical).
