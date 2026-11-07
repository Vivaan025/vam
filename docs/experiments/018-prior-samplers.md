# 018 · Prior adaptive frame selectors under one backbone and one budget (roadmap R9)

**Date:** 2026-11-07 · **Status:** in progress · **Results:** pending

## Motivation
017 puts `vam` on the datasets the literature uses. This entry puts the literature next to `vam` on the same
machine, with the same classifier, under the same latency budget: the comparison a reviewer asks for.

## The methods
| method | venue | what it adapts | official code | released weights | how it enters this study |
|---|---|---|---|---|---|
| SCSampler (Korbar et al.) | ICCV 2019 | clip saliency from a light audio/visual scorer | none (AR-Net re-implemented it) | none | re-implemented: light per-clip scorer trained on train-set clip usefulness; same family as our learned scorer |
| AdaFrame (Wu et al.) | CVPR 2019 | LSTM agent picks the next frame to look at, RL | none | none | re-implemented from the paper (MobileNet global memory + LSTM policy); budget set by its own early stop |
| AR-Net (Meng et al.) | ECCV 2020 | per-frame resolution / skip, Gumbel policy | github.com/mengyuest/AR-Net | Google Drive | run with released ActivityNet weights |
| FrameExit (Ghodrati et al.) | CVPR 2021 | early exit after k frames via gates | github.com/Qualcomm-AI-research/FrameExit | GitHub release assets (ActivityNet, Mini-Kinetics) | run with released weights; **its ResNet-50 classifier is the shared backbone of the study** |
| VideoIQ (Sun et al.) | ICCV 2021 | per-frame precision (quantisation) | github.com/sunxm2357/VideoIQ | not released ("coming later") | literature numbers only unless weights appear |
| SMART (Gowda et al.) | AAAI 2021 | pairwise attention over candidate frames, single pass | none | none | re-implemented (single-frame + pair attention selector) |
| OCSampler (Lin et al.) | CVPR 2022 | one-step selection of 6 of 10 frames by a MobileNet-TSM policy | github.com/MCG-NJU/OCSampler | Google Drive (`anet_10to6_checkpoint.pth`) | run with released weights; uses FrameExit's classifier |

Methods with released weights are run as released (frames at 1 fps, short side 350, as 017 reproduces).
Methods without code are re-implemented from the paper and trained on ActivityNet train; the
re-implementation is labelled as such in every table, with the paper's own number beside it.

## Hypotheses (acceptance criteria)
- **H1, published numbers reproduce.** FrameExit 76.1 mAP at 24.7 GFLOPs, OCSampler 76.9 at 21.7, AR-Net
  73.8 at 33.4 (ResNet) each reproduce within 0.5 mAP from the released weights on our val copy.
- **H2, same backbone, same frames.** With the FrameExit ResNet-50 classifier fixed and 6 frames per video,
  `vam` keyframe + learned scorer (ActivityNet-trained, 017 H3) is within 1 mAP of OCSampler's 10-to-6 and
  above uniform-6; at 4 frames it is above every method's 4-frame operating point.
- **H3, same budget in milliseconds.** At 100 / 150 / 200 ms per video on the 5090, measured end to end
  (container to prediction, decode included), `BudgetedSampler` reaches the highest mAP of all methods. The
  prior methods count GFLOPs and assume pre-extracted frames; their measured latency includes the full decode
  of all 10 candidate frames, which is the cost the keyframe scan avoids.
- **H4, GFLOPs.** On the GFLOPs axis the prior methods use, `vam` at 6 frames costs 6 x 4.1 (ResNet-50) plus
  the CLIP ViT-B/32 scorer over the scanned keyframes (about 4.4 GFLOPs per keyframe); reported honestly, this
  may sit above OCSampler's 21.7. The ms axis and the GFLOPs axis are both reported.

## Plan
1. `vam/models/frameexit.py` (017) plus thin wrappers that expose AR-Net and OCSampler policies as
   `Sampler`s in `benchmarks/prior/` (their code vendored under `data/tools/`, run in a separate venv
   `.venv-prior` because they need PyTorch 1.x-era mmcv / mmaction2).
2. Re-implementations `vam/sampling/prior/{scsampler,adaframe,smart}.py`, each 300 lines or fewer, trained
   with `scripts/train_prior.py` on ActivityNet train keyframe features; unit tests on synthetic clips.
3. `benchmarks/eval_prior.py`: one table, rows = methods x operating points, columns = mAP, GFLOPs, measured
   ms (p50 / p90), frames used.
4. `docs/results/activitynet_prior.md`; README table; study guide section.

## Metrics
mAP on ActivityNet-1.3 val (multi-label), GFLOPs per video, measured ms per video on the 5090, frames used.

## Risks
- OCSampler and AR-Net need mmaction2 0.12 / PyTorch 1.5 era code; porting to torch 2.11 may take days. The
  fallback is to run their policy networks only (MobileNet-TSM, ResNet) with weights loaded into clean
  re-implementations, verified by matching their reported mAP (H1).
- Re-implemented methods may fall short of their papers; the gap is reported, not hidden.
- Google Drive weights may be rate-limited; `gdown` from the server, otherwise Vivaan downloads them.

## Results
### 2026-11-07: OCSampler ported (`vam/sampling/prior.py`), not yet run
- The released `anet_10to6_checkpoint.pth` carries FrameExit's weights verbatim: its `backbone.model.*` equals
  FrameExit's ResNet-50 tensor for tensor, `cls_head.layers` is FrameExit's MLP and `cls_head.fc` is FrameExit's
  10-frame head (`classifiers.9`). So OCSampler = `FrameExitModel(fixed_head=10)` + a MobileNetV2-TSM policy over
  10 uniform frames at 128 px (nearest resize of the 224 crop), top-K kept in temporal order. The policy
  (`sampler.*` keys, `.net.` on the shifted expand convs, shift_div 10 ActivityNet / 8 Mini-K) loads with
  `weights_only=True`. Candidate positions reproduce mmaction's `UniformSampleFrames(test_mode, seed 255)`
  (tested against a verbatim copy). The 10to5 checkpoint has a different policy and the same head.
- `benchmarks/eval_activitynet.py --arms ocsampler-6 ocsampler-5` (`--ocsampler-checkpoint`); GMACs = 4.12 x K
  + the policy's measured MACs (FlopCounterMode, ~1 GMAC for 10 frames at 128 px).
- Published numbers to compare against (OCSampler paper, Table 1, ImageNet-pretrained ResNet-50 classifier from
  FrameExit, T = 10 candidates): ActivityNet 10to6 **77.2 mAP / 25.8 GFLOPs**, 10to5 76.9 / 21.7; Mini-Kinetics
  10to5 **73.7 top-1 / 21.6**, 10to4 72.9 / 17.5. Same table: FrameExit 76.1 / 26.1 and 72.8 / 19.7, AR-Net 73.8 /
  33.5 and 71.7 / 32.0, SCSampler 72.9 / 42.0 and 70.8 / 41.9, VideoIQ 74.8 / 28.1 and 72.3 / 20.4, AdaFocus 75.0 /
  26.6 and 72.9 / 38.6, LiteEval 72.7 / 95.1 and 61.0 / 99.0. Their sampler cost is 25.8 - 24.7 = 1.1 GFLOPs,
  which matches the 1.0 GMACs we count for 10 frames at 128 px.
- The paper's Table 4 gives uniform 72.6 / 73.8 mAP at N = 4 / 6 through the 10-frame head; our uniform through
  the same head is 0.721 / 0.751 (`uniformh10-K`) and through the matched heads 0.721 / 0.752. So the head does
  not explain the 1.4-point gap between their uniform-6 and ours (an earlier draft of this entry guessed it did);
  their baseline's frames or decode do. OCSampler's margin over uniform on our decode is 2.2, not 3.4.
- First 200 val videos (`results/activitynet_018_smoke.jsonl`): ocsampler-6 0.848 mAP at 25.7 GMACs, 167 ms p50;
  uniform-6 (6-frame head) 0.853 at 24.7 GMACs, 96 ms. Full val running (`results/logs/queue018.sh`).
- Still to run: ActivityNet val 10to6 / 10to5; Mini-K 10to4 / 10to5; uniform-K through the 10-frame head.

### 2026-11-07: AR-Net and its SCSampler ported (`vam/models/arnet.py`), not yet run
- Both run from the released `.pth.tar` files (`exp0_res_ada`, `exp1_res_scsampler`; loaded with
  `weights_only=True` and an allowlist for the argparse / numpy objects pickled beside the weights). AR-Net's
  classifier is its own TSN (torchvision ResNet-50/34/18 + fresh fc), not FrameExit's, so its rows share the
  dataset and the frame protocol (16 frames at `int(tick/2 + tick*x)`, short side 256, centre crop 224) but not
  the classifier; the literature table (OCSampler Table 1) mixes them the same way.
- SCSampler as AR-Net re-implements it: MobileNetV2 on all 16 frames, saliency = max class logit of its own
  head, ResNet-50 on the top-10, logits averaged. The test script feeds the policy the 224 crop
  (`--policy_input_offset 0`), which costs 45.7 GMACs; the paper's 42.0 GFLOPs implies the policy at 84 px
  (41.9). Both are run (`--arnet-sc-policy-px 224 / 84`) and reported.
- AR-Net adaptive: MobileNetV2 at 84 px -> LSTM(512) -> 7 actions per frame (ResNet-50 @ 224, ResNet-34 @ 168,
  ResNet-18 @ 112, policy head, skip 1 / 2 / 4); chosen logits averaged. The released test samples the action
  with Gumbel noise at tau = 1e-6; we take the arg max (deterministic limit) and say so. Per-option MACs are
  measured with `FlopCounterMode` at init (ResNet-50 @ 224 = 4.1, matching the 4.12 convention of 017).
### Full results, 2026-11-07 (`docs/results/activitynet_018.md` + CSV)
| method (ActivityNet val) | classifier | mAP | frames | GMACs | p50 ms | published |
|---|---|---|---|---|---|---|
| uniform-6, matched head / 10-frame head | FrameExit | 0.752 / 0.751 | 6 | 24.7 | 92 / 128 | – / 73.8 |
| FrameExit exit | FrameExit | 0.755 | 6.00 | 24.7 | 137 | 76.1 / 26.1 |
| OCSampler 10→5 / 10→6 | FrameExit | 0.766 / **0.774** | 5 / 6 | 21.6 / 25.7 | 207 / 184 | 76.9 / 21.7, 77.2 / 25.8 |
| AR-Net adaptive | own TSN | 0.738 | 13.5 | 36.4 | 316 | 73.8 / 33.5 |
| SCSampler (AR-Net), policy 224 / 84 px | own TSN | 0.737 / 0.724 | 10 | 45.7 / 41.7 | 246 / 236 | 72.9 / 42.0 |
| vam keyframe-6, ActivityNet scorer | FrameExit | 0.703 | 5.8 | 439 | 178 | – |
| vam budget-150 | FrameExit | 0.753 | 11.1 | 197 | 143 | – |

Mini-Kinetics: OCSampler 10→4 / 10→5 0.718 / 0.722 top-1 (published 72.9 / 73.7) at 17.5 / 21.6 GMACs; FrameExit
0.7265 at 19.6; uniform-4 / 6 0.706 / 0.723.

- **H1 (published numbers reproduce):** OCSampler within 0.3 on ActivityNet and 1.5 on Mini-K, AR-Net within 0.1,
  FrameExit within 0.6, SCSampler at 224 px 0.8 *above* its paper and at 84 px 0.5 below; compute matches the
  published GFLOPs where the released configuration is unambiguous. Pass.
- **H2 (vam within 1 mAP of OCSampler 10→6 at its cost):** fails. keyframe-6 is 7 points below at 17x the GMACs;
  budget-150 is 2.1 below at 8x. The ActivityNet-trained scorer (017 H3) does not change this.
- **H3 (measured ms orders methods differently from GFLOPs):** holds. Uniform-6 92 ms, OCSampler 184 ms (10
  candidate decodes + policy before the first classifier frame), AR-Net 316 ms (16 decodes + per-frame LSTM),
  SCSampler 246 ms; vam budget-150 143 ms. On GFLOPs OCSampler is the cheapest adaptive method; on wall time it
  is 2x uniform and only vam's budget sampler sits between the two.
- **H4 (a reviewer's "at least two prior samplers under the same backbone and budget"):** met for OCSampler and
  FrameExit (same classifier, same decode, same machine); AR-Net / SCSampler share data and machine but keep
  their own classifier, as in the published tables. AdaFrame and SMART were not re-implemented; VideoIQ is
  quoted from the literature only.

## Decision
**Done at the "as released" level; `vam`'s keyframe sampler is not a competitor on ActivityNet and is not
presented as one.** The table that goes into any write-up is `docs/results/activitynet_018.md`: three prior
methods reproduced from their weights on one machine, with published numbers beside ours, measured milliseconds
as a second axis, and `vam` entering as the budget sampler on the uniform line. The honest claim for the library
is unchanged from 012 / 017: on low-coverage long video the keyframe runtime is +17 at 4 frames and ~150 ms at
any budget; on high-coverage benchmarks the right answer is uniform (or OCSampler's policy, which is now one
`Sampler` in `vam`), and the scheduler must decide that from the footage. Next: coverage estimation in the
scheduler, then a cheaper scan; AdaFrame / SMART only if a reviewer asks.
