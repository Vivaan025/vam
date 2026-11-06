# 017 · Standard benchmarks: ActivityNet-1.3 and Mini-Kinetics (roadmap R8)

**Date:** 2026-11-07 · **Status:** in progress · **Results:** pending

## Motivation
Every accuracy number so far is top-1 of a linear probe on frozen VideoMAE features over 200 THUMOS14 videos
with 20 classes (plus Charades as the negative case and R(2+1)D as a second backbone). That is a controlled
engineering study, not a benchmark the adaptive-frame-selection literature recognises. The papers this work
must be read against (AdaFrame, SCSampler, AR-Net, FrameExit, VideoIQ, OCSampler, SMART) report mAP on
ActivityNet-1.3 val and top-1 on Mini-Kinetics (the AR-Net 200-class subset of Kinetics-400), and FCVID,
all with an end-to-end trained ResNet-50 (or MobileNet / EfficientNet) classifier and a GFLOPs axis. Until
`vam` has numbers on those sets under that protocol, "how does it do against prior work" has no answer.

## Datasets and sources
| set | protocol in the literature | source available to us | size | status |
|---|---|---|---|---|
| ActivityNet-1.3 val (4,926 videos, 200 classes, untrimmed) | 1 fps frames, short side 350, 10 frames, ResNet-50 (FrameExit protocol); mAP | Hugging Face mirrors: `YimuWang/ActivityNet` (full resolution, 577 GB) and `interlive/stream-data` (`videos/activitynet_part01..12.tar`, 115 GB, resolution to be checked) | server has 697 GB free | one shard being inspected |
| Mini-Kinetics val (9,863 videos of the 200 AR-Net classes) | 5 fps frames, short side 256, ResNet-50; top-1 | Kinetics-400 val from CVDF (`s3.amazonaws.com/kinetics/400/val`, 20 shards, ~30 GB); class lists from the OCSampler repo (`data/mini_kinetics/*.txt`) | 30 GB | downloading |
| FCVID (91k videos, 239 classes) | ResNet-50; mAP | request-only from Fudan (`bigvid.fudan.edu.cn/FCVID`, agreement form by e-mail) | 4,232 h | **blocked: Vivaan must send the request** |
| Kinetics-400 full | top-1 | CVDF, 685 GB | exceeds free disk | not planned |

Frames are not pre-extracted. `vam` selects frames from the container, so each video is kept as a video and
the FrameExit / OCSampler frame protocol (1 fps, short side 350, 224 centre crop) is reproduced at decode
time; a check that this reproduces FrameExit's published 0.7612 mAP with its own checkpoint is H1.

## Hypotheses (acceptance criteria)
- **H1, the protocol is reproduced.** FrameExit's released ActivityNet checkpoint (ResNet-50, 10 frames,
  `activitynet_checkpoint_25gmac.pth`) evaluated through `vam`'s decode gives mAP within 0.5 of 0.7612, and
  its Mini-Kinetics checkpoint within 0.5 of 0.7331 top-1. Without this the rest is not comparable.
- **H2, the keyframe sampler transfers.** With the FrameExit classifier fixed and the sampler swapped from
  uniform-at-1 fps to `KeyframeSampler` (packaged scorer trained on THUMOS14, no retraining), mAP at 4, 6
  and 8 frames is at least that of uniform at the same count. The gap is expected to be smaller than on
  THUMOS14 because ActivityNet actions cover about half of the video (the 012 coverage rule).
- **H3, a scorer trained on ActivityNet train closes more.** The class-free head retrained on ActivityNet
  train keyframes (same recipe as 009) beats the THUMOS head by at least 1 mAP at 6 frames.
- **H4, Mini-Kinetics is the negative control.** Trimmed 10 s clips: expect uniform to equal keyframe, as on
  UCF101. Reported as such.

## Plan
1. Data: K400 val shards + Mini-Kinetics lists; ActivityNet val from the mirror whose resolution matches the
   FrameExit protocol (short side at least 350) or, failing that, the full-resolution mirror streamed through
   `tar` with a val-only file list so only the 4.9k val videos land on disk. Train subset (2k or more videos)
   for H3.
2. `vam/models/frameexit.py`: adapter that loads the FrameExit ResNet-50 + classifier as a `VideoActionModel`
   (frozen torchvision ResNet-50 features, max pooling, 200-way head) so every sampler can drive it.
3. `benchmarks/eval_activitynet.py`, `benchmarks/eval_minik.py`: mAP / top-1, GFLOPs (counted), measured ms
   on the 5090, per sampler and frame count; CSVs in `docs/results/`.
4. H3: keyframe scan + CLIP embeddings over the train subset, oracle labels from the temporal annotations,
   `scripts/train_scorer.py` as in 009.

## Metrics
mAP (ActivityNet, multi-label as in FrameExit's `anet_val_video_multi_label.txt`), top-1 (Mini-Kinetics),
GFLOPs per video (sampler + classifier), measured select / decode / infer / total ms on the 5090.

## Risks
- Mirror videos may be re-encoded at low resolution; if the short side is below 350 the FrameExit protocol
  degrades and H1 may miss. Recorded, and the full-resolution mirror used for val if so.
- Missing videos: mirrors are incomplete; mAP is reported on the videos present with the count stated.
- FrameExit's checkpoint was trained with PyTorch 1.3; state-dict keys may need mapping.

## Results
### Interim, 2026-11-07: smoke run on the first 339 val videos (v1-2 part of val), RTX 5090
The FrameExit checkpoint loads and runs through `vam`'s decode. On this subset:

| arm | mAP | hit@1 | frames | GMACs | p50 / p90 ms |
|---|---|---|---|---|---|
| frameexit (its early exit) | 0.790 | 0.667 | 6.17 | 25.4 | 120 / 300 |
| fe-uniform-10 (its frames, no exit) | 0.796 | 0.673 | 10 | 41.2 | 117 / 319 |
| uniform-4 | 0.781 | 0.646 | 4 | 16.5 | 56 / 139 |
| uniform-6 | 0.797 | 0.687 | 6 | 24.7 | 80 / 206 |
| uniform-8 | 0.816 | 0.696 | 8 | 33.0 | 106 / 291 |
| uniform-10 | 0.825 | 0.690 | 10 | 41.2 | 129 / 355 |
| keyframe-4 (THUMOS scorer, refine) | 0.718 | 0.602 | 3.9 | 340 | 129 / 386 |
| keyframe-6 | 0.754 | 0.661 | 5.8 | 437 | 130 / 387 |
| keyframe-8 | 0.765 | 0.649 | 7.7 | 535 | 141 / 436 |

- FrameExit's own numbers sit where its paper puts them (0.790 on this easier subset vs 0.761 published on
  the full val at 24.7 GFLOPs); H1 is decided on the full val.
- **The THUMOS-trained keyframe sampler loses to uniform on ActivityNet at every frame count, on both axes.**
  ActivityNet val action coverage is median 0.72 (mean 0.66, 71 % of videos above 0.5; THUMOS14 0.23,
  Charades 0.95), so by the 012 rule the headroom for any selector is small, and the FrameExit heads were
  trained on uniformly spread frames. The GMACs column is dominated by the CLIP scorer over ~70 keyframes
  plus refinement candidates per two-minute video.
- **The ceiling is flat.** Same subset, same classifier, frames taken uniformly *inside the annotated
  segments* (ground truth):

  | frames | uniform | oracle | gap | kfnr (THUMOS scorer, no refinement) | GMACs kfnr | scored keyframes |
  |---|---|---|---|---|---|---|
  | 4 | 0.781 | 0.794 | +1.4 | 0.748 | 166 | 34 |
  | 6 | 0.797 | 0.810 | +1.3 | 0.772 | 174 | 34 |
  | 8 | 0.816 | 0.814 | −0.2 | 0.789 | 181 | 34 |
  | 10 | 0.825 | 0.818 | −0.7 | – | – | – |

  A perfect selector gains 1.4 mAP at 4 frames and nothing from 8 frames up. On THUMOS14 the same gap was
  +17 points at 4 frames. ActivityNet under the FrameExit protocol is therefore Charades-shaped (012): the
  action fills most of the video, a per-frame ResNet-50 with max pooling sees it from almost any frame, and
  "which frames" is not the question the benchmark asks. The published gains on this benchmark (FrameExit,
  OCSampler, AR-Net) are compute savings at near-constant accuracy, not accuracy from finding the action.
- Without refinement the scan costs 34 CLIP images per video (166 GMACs at 4 frames against 16.5 for the
  ResNet-50 itself) and still loses 3 points to uniform: the THUMOS head's ranking does not transfer, and the
  gap-window placement concentrates frames where the FrameExit heads, trained on spread-out frames, expect
  coverage.
- Consequences for 018: the comparison on ActivityNet is on the cost axis at matched accuracy, and the honest
  entry for `vam` there is the scheduler's uniform fallback (013), which should choose uniform on this
  footage; the accuracy contribution of keyframe selection is a long-video, low-coverage result (THUMOS14
  shape) and is reported as such. H2 is expected to fail on ActivityNet and H4 to hold on Mini-Kinetics.
- Still to run: full val (queued), H3 (ActivityNet-trained head: can it recover the 1.4 points at 4 frames
  at a scan cost that fits?), and one clip-level backbone (VideoMAE probe) on ActivityNet to separate
  "coverage" from "per-frame classifier" as the cause of the flat ceiling.


## Decision
pending
