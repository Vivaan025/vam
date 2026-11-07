# Experiment 017 results: ActivityNet-1.3 and Mini-Kinetics under the FrameExit protocol

**Date:** 2026-11-07 · **GPU:** RTX 5090 · **classifier:** FrameExit's released ResNet-50 + MLP with one head per
frame count (ActivityNet `activitynet_checkpoint_25gmac.pth`, Mini-K `minikinetics_checkpoint_19.7gmac.pth`), run
through `vam`'s decode (frames from the container, short side 224, centre crop, ImageNet norm; 1 fps positions,
Mini-K 5 fps). mAP = per-class AP over videos averaged over classes with positives; hit@1 = top-1. GMACs =
4.12 per ResNet-50 frame + 4.4 per CLIP image the sampler scores. ms = select + decode + infer per video, p50.
Raw rows: `activitynet_017_rtx5090.csv`; per-video JSON on the server (`results/*_017*.jsonl`).

## ActivityNet-1.3 val, 4,926 videos (one undecodable in the mirror)
| arm | mAP | hit@1 | frames | GMACs | p50 ms | fallback | published |
|---|---|---|---|---|---|---|---|
| **frameexit** (its early exit) | **0.755** | 0.707 | 6.00 | 24.7 | 137 | – | 0.761 at 24.7 (FrameExit, Table 1) |
| frameexit, 350-px intermediate resize | 0.753 | 0.704 | 6.03 | 24.8 | 138 | – | same |
| fe-uniform-10 (its 10 frames, no exit) | 0.771 | 0.717 | 10 | 41.2 | 136 | – | – |
| uniform-4 / 6 / 8 / 10 (matched heads) | 0.721 / 0.752 / 0.768 / 0.779 | 0.673 / 0.697 / 0.715 / 0.721 | 4–10 | 16.5–41.2 | 64 / 92 / 117 / 146 | – | – |
| oracle-4 / 6 / 8 / 10 (uniform inside annotated segments) | 0.750 / 0.772 / 0.784 / 0.793 | 0.694 / 0.721 / 0.725 / 0.742 | 4–10 | 16.5–41.2 | 63 / 88 / 109 / 133 | – | – |
| keyframe-4 / 6 / 8, THUMOS scorer | 0.616 / 0.663 / 0.693 | 0.570 / 0.616 / 0.644 | 3.9–7.6 | 338 / 429 / 522 | 140 / 151 / 161 | 233 / 420 / 610 | – |
| keyframe-4 / 6 / 8, ActivityNet-trained scorer (H3) | 0.672 / 0.703 / 0.721 | 0.629 / 0.661 / 0.675 | 3.9–7.6 | 344 / 439 / 532 | 143 / 178 / 204 | 242 / 433 / 619 | – |
| kfnr-4 / 6 / 8, ActivityNet scorer (no refinement) | 0.674 / 0.699 / 0.718 | 0.630 / 0.651 / 0.670 | 3.9–7.7 | 168 / 175 / 183 | 127 / 137 / 129 | 242 / 433 / 619 | – |
| budget-100 / 150, ActivityNet scorer | 0.717 / 0.753 | 0.673 / 0.700 | 6.3 / 11.1 | 178 / 197 | 120 / 143 | 1037 | – |
| budget-100 / 150 / 200, THUMOS scorer | 0.676 / 0.723 / 0.739 | 0.628 / 0.672 / 0.687 | 6.3 / 11.1 / 14.6 | 177 / 197 / 271 | 113 / 130 / 151 | 1026 | – |

## Mini-Kinetics val, 9,858 of 9,863 clips (5 missing from the CVDF shards), 256-px intermediate resize
| arm | top-1 | mAP | frames | GMACs | p50 ms | fallback | published |
|---|---|---|---|---|---|---|---|
| **frameexit** (exit, first threshold 0.95) | **0.7265** | 0.780 | 4.77 | 19.6 | 32 | – | 0.7331 at 19.7 |
| fe-uniform-10 | 0.737 | 0.796 | 10 | 41.2 | 31 | – | – |
| uniform-4 / 6 / 8 / 10 | 0.706 / 0.723 / 0.734 / 0.739 | 0.765 / 0.781 / 0.790 / 0.796 | 4–10 | 16.5–41.2 | 46 / 28 / 29 / 29 | – | – |
| keyframe-4 / 8, THUMOS scorer | 0.693 / 0.718 | 0.751 / 0.772 | 4.0 / 7.9 | 98.7 / 114.9 | 63 / 45 | 8,190 | – |
| kfnr-4 / 8 | 0.694 / 0.720 | 0.751 / 0.774 | 4.0 / 8.0 | 26.2 / 42.6 | 59 / 43 | 8,190 | – |

## Reading
- **Reproduction (H1):** 0.6 and 0.7 below the published numbers with the frame budget exact on both sets. The
  intermediate resize (350 / 256) changes nothing; the residual is FrameExit's JPEG extraction, which `vam` does
  not reproduce because it reads the container. Every sampler below runs on the same decode, so the comparison
  between them is unaffected; the published number is quoted beside ours.
- **Ceiling (oracle):** +2.9 / +2.0 / +1.6 / +1.5 mAP over uniform at 4 / 6 / 8 / 10 frames. On THUMOS14 it was
  +17 at 4. ActivityNet actions cover a median 72 % of the video; the benchmark is Charades-shaped for a
  per-frame classifier. (The oracle here is "uniform inside the annotation", not the best possible frames:
  OCSampler's learned policy reaches 0.774 at 6 frames, above oracle-6's 0.772, see `activitynet_018.md`.)
- **H2 fails; H3 recovers half of it.** The THUMOS-trained scorer loses 10.5 mAP to uniform at 4 frames; retrained
  on ActivityNet it loses 4.9. Both fall back to uniform on 5–12 % of videos and cost 34 CLIP images per
  video (73–114 with refinement), which no ResNet-50 budget absorbs. The budget sampler sits on the uniform
  line at best (0.753 at 11 frames vs uniform-10 0.779).
- **H4 holds:** on trimmed 10 s clips the short-video rule sends 83 % to uniform and the rest is 1.3–1.5 below.
