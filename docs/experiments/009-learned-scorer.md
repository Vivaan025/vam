# 009 · Learned keyframe scorer

**Date:** 2026-11-02 · **Status:** planned · **Results:** *(to fill)*

(008, NVDEC decode, is deferred: the decode stage is a ~90 ms seek floor already below uniform sampling's
latency, and hardware decode adds an install burden. Numbering is kept so the plan in 007 stays true.)

## Motivation (from 007)
With the runtime fixed at ~150 ms per video, what remains is the scorer. At 4 frames kf-clip + refinement
reaches 0.850–0.865 against the oracle's 0.890; at 8 frames 0.895–0.900 against 0.925. The zero-shot CLIP
scorer has two weaknesses: it needs the class names (a product user often has none), and it scores each
keyframe in isolation, whereas actions span several keyframes and the keyframe *before* the action is the one
whose window covers it (006: the oracle scores a keyframe by the overlap of the window *after* it).

## Hypothesis
A small head on frozen CLIP image embeddings, trained to predict the oracle window-overlap score from the
embedding of a keyframe and its neighbours, (a) ranks keyframes closer to the oracle than zero-shot prompts,
(b) needs no class names at inference, and (c) costs < 2 ms per video. Expected: kf-learned-4 ≥ 0.87 and
kf-learned-8 ≥ 0.91 with refinement, at the latency of 007 (±5 ms).

## Plan
1. **Features.** Cache per video: keyframe times, CLIP image embeddings of the 112² thumbnails (512-d, the
   same forward the zero-shot scorer already runs), the zero-shot score, the gap to the next keyframe, and the
   oracle target (overlap of the gap window with annotated action, in [0, 1]). ~50 keyframes × 200 videos.
2. **Split.** THUMOS14 has only the 200 annotated validation videos locally (the test set is a separate 213-video
   download), so use **5-fold cross-validation by video**: every video is scored by a head that never saw it.
   Folds are fixed by a seed and stored with the heads.
3. **Models**, each trained per fold with early stopping on the held-out fold's ranking metric:
   - `ridge`: linear on the embedding (+ zero-shot score, gap) — the floor.
   - `mlp`: 512 → 256 → 1 with the same inputs.
   - `context`: the embeddings of keyframes i−2 … i+2 concatenated (zeros at the edges) → MLP, so the head can
     tell "action is about to start" from "action is ending".
   Loss: MSE on the oracle score (it is a fraction) plus a pairwise ranking term within a video.
4. **Scorer-level metrics** on held-out folds: Spearman correlation with the oracle score per video; precision
   of the top-2 keyframes against the oracle's top-2 (the 4-frame case); mean oracle score of the chosen windows.
5. **End-to-end**: `LearnedScorer` (CLIP embed → head; the embedding is reused for refinement candidates) in the
   benchmark as `kf-learned`, with the same gap + softmax + refinement defaults as kf-clip, at 4/8/16; the
   benchmark picks each video's fold head. Compare to kf-clip and kf-oracle from 007.
6. Ablate: `context` vs `mlp` vs `ridge`; with and without the zero-shot score as an input (if the head is as good
   without it, the scorer is class-free).

## Metrics
Spearman vs oracle, top-2 precision, top-1 at 4/8/16, hit rate, select ms. Success: kf-learned-4 ≥ 0.87,
kf-learned-8 ≥ 0.91, select ms within +5 of kf-clip; and the class-free variant within 1 point of the one that
uses the zero-shot score.

## Risks
- 200 videos is small; the head must be tiny and regularised, and the CV estimate will be noisy (±2 points).
  Report fold spread, not just the mean.
- The oracle target is defined by the window rule, so the head learns our window, not "actionness"; that is
  intended (the scorer should rank windows), but it means a window change needs a retrain.
- CLIP embeddings of 112² thumbnails may lack the detail to tell a warm-up from the action; if Spearman stalls
  below ~0.5, try 224² thumbnails for the scorer only (scan cost +10 ms, 005 showed conversion is cheap).

## Results
*(to fill)*

## Decision
*(to fill)*
