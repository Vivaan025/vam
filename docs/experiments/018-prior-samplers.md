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
- Still to run: ActivityNet val 10to6 / 10to5 against the published 76.9 / 21.7 GFLOPs; Mini-K 10to4 / 10to5.

## Decision
pending
