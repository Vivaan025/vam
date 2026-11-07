# Experiment 018 results: prior adaptive samplers, as released, on the same data and machine

**Date:** 2026-11-07 · **GPU:** RTX 5090 · **data:** ActivityNet-1.3 val (4,926 videos) and Mini-Kinetics val
(9,858 clips), the same files and decode as `activitynet_017.md`. OCSampler rows use FrameExit's classifier (the
released OCSampler checkpoints carry FrameExit's ResNet-50, MLP and 10-frame head verbatim); AR-Net rows use
AR-Net's own TSN ResNet-50 (its released checkpoints), so they share the data and frame protocol with the rest of
the table but not the classifier, exactly as the published tables mix them. GMACs: 4.12 per ResNet-50 frame
plus the policy's MACs measured with `torch.utils.flop_counter` (OCSampler 1.0 for 10 frames at 128 px; AR-Net
per option). ms = select + decode + infer per video, p50. Raw rows: `activitynet_018_rtx5090.csv`.

## ActivityNet-1.3 val
| method | classifier | mAP | hit@1 | frames | GMACs | p50 ms | published mAP / GFLOPs |
|---|---|---|---|---|---|---|---|
| uniform-4 / 6, matched heads (017) | FrameExit | 0.721 / 0.752 | 0.673 / 0.697 | 4 / 6 | 16.5 / 24.7 | 64 / 92 | – |
| uniform-4 / 6, 10-frame head | FrameExit | 0.721 / 0.751 | 0.674 / 0.701 | 4 / 6 | 16.5 / 24.7 | 86 / 128 | 72.6 / 73.8 (OCSampler Table 4) |
| FrameExit, early exit (017) | FrameExit | 0.755 | 0.707 | 6.00 | 24.7 | 137 | 76.1 / 26.1 |
| **OCSampler 10→5** | FrameExit | 0.766 | 0.715 | 5 | 21.6 | 207 | 76.9 / 21.7 |
| **OCSampler 10→6** | FrameExit | **0.774** | 0.722 | 6 | 25.7 | 184 | 77.2 / 25.8 |
| AR-Net adaptive (arg-max policy) | AR-Net TSN | 0.738 | 0.693 | 13.5 | 36.4 | 316 | 73.8 / 33.5 |
| SCSampler (AR-Net re-impl.), policy at 224 px as in its test script | AR-Net TSN | 0.737 | 0.698 | 10 | 45.7 | 246 | 72.9 / 42.0 |
| SCSampler, policy at 84 px (the cost the paper states) | AR-Net TSN | 0.724 | 0.685 | 10 | 41.7 | 236 | 72.9 / 42.0 |
| vam keyframe-4 / 6 / 8, ActivityNet scorer (017) | FrameExit | 0.672 / 0.703 / 0.721 | 0.629 / 0.661 / 0.675 | 3.9–7.6 | 344–532 | 143–204 | – |
| vam budget-150, ActivityNet scorer (017) | FrameExit | 0.753 | 0.700 | 11.1 | 197 | 143 | – |
| oracle-6 (uniform inside annotations, 017) | FrameExit | 0.772 | 0.721 | 6 | 24.7 | 88 | – |

## Mini-Kinetics val
| method | top-1 | mAP | frames | GMACs | p50 ms | published top-1 / GFLOPs |
|---|---|---|---|---|---|---|
| uniform-4 / 6 (017) | 0.706 / 0.723 | 0.765 / 0.781 | 4 / 6 | 16.5 / 24.7 | 46 / 28 | – |
| FrameExit, early exit (017) | 0.7265 | 0.780 | 4.77 | 19.6 | 32 | 73.3 / 19.7 |
| **OCSampler 10→4** | 0.718 | 0.775 | 4 | 17.5 | 59 | 72.9 / 17.5 |
| **OCSampler 10→5** | 0.722 | 0.781 | 5 | 21.6 | 61 | 73.7 / 21.6 |

## Reading
- **Everything reproduces to within 1.5 points with the compute exact.** OCSampler on ActivityNet is 0.2–0.3
  short of its paper, AR-Net adaptive 0.05 short, FrameExit 0.6 short, OCSampler on Mini-K 1.1–1.5 short. The
  shared residual is the JPEG extraction the released pipelines used (017). Costs agree with the published GFLOPs
  where the published configuration is unambiguous; SCSampler's test script feeds the policy the 224 crop
  (45.7 GMACs) while the paper's 42.0 implies 84 px (41.7), and the two differ by 1.3 mAP.
- **OCSampler is the strongest selector on this benchmark**, +2.2 mAP over uniform-6 at +1.0 GMACs, and it
  edges the segment oracle, so a learned policy can find frames the classifier likes better than evenly spaced
  frames inside the annotation. Its price on this machine is 2x the latency of uniform (decoding 10 candidates
  and running the policy before any classifier frame).
- **The head is not where OCSampler's uniform baseline lost 1.4 points.** Through the 10-frame head uniform-6
  gives 0.751, through the matched head 0.752. The paper's 73.8 for uniform-6 is below both, so the gap to its
  own uniform is in its baseline's frames or decode, not in the head as the 018 entry first guessed.
- **AR-Net and SCSampler sit between uniform-4 and uniform-6 in accuracy at 1.5–2.8x their cost** with their own
  classifier; the ranking in the published tables (OCSampler > FrameExit > AR-Net > SCSampler) holds here.
- **`vam`'s keyframe sampler does not belong in this table as a competitor**, 7 points below OCSampler at 17x
  the GMACs; its budget sampler lands on the uniform line. The library's correct entry on ActivityNet is the
  uniform fallback. The runtime contribution is the millisecond axis on low-coverage footage (THUMOS14), where
  none of these methods were evaluated and where the oracle gap is +17, not +3.
