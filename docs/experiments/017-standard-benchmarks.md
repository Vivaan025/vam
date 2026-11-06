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
pending

## Decision
pending
