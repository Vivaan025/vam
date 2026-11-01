# 005 · Cheaper keyframe scan

**Date:** 2026-11-01 · **Status:** planned · **Results:** *(to fill)*

## Motivation (from 004)
After tensor preprocessing, kf-clip's selection cost is ~160 ms per video, of which CLIP scoring is
~8 ms. The remaining ~150 ms is `scan_keyframes`: PyAV demuxing every packet, decoding the ~25–60
keyframes, and converting each to a 112×112 RGB thumbnail. Uniform sampling has zero selection cost,
so this fixed ~150 ms is exactly the region (< 250 ms total) where uniform still wins.

Laptop measurements from 003 (3-minute 320×180 file, 24 keyframes) suggest where the time goes:
demux-only ≈ decode-keyframes-only (both dominated by reading/parsing ~5,000 packets), and
`to_ndarray(rgb24, resized)` costs more than the decode itself.

## Hypothesis
The scan can be brought from ~150 ms to ≤ 50 ms per video with no change in which keyframes are
found, by (a) avoiding per-frame swscale conversion, (b) reducing demux overhead, and (c) overlapping
the scan with other work. If so, kf-clip-4 total drops from ~264 ms to ~165 ms, within 20 ms of
uniform-4 (148 ms) while scoring +7.5 points higher.

## Plan
1. **Profile the scan** on the server with per-stage timers: open, demux loop, decode of keyframe
   packets, `to_ndarray` conversion. Repeat for 3 videos with different keyframe counts.
2. Candidates, measured one at a time against the profile:
   - **Conversion:** decode to the native `yuv420p` plane and build the thumbnail from the Y plane
     only (grey) or do the YUV→RGB + resize as a tensor op on the device, batched for all keyframes
     (reuse `Preprocessor`). CLIP accuracy on grey thumbnails must be checked.
   - **Demux:** iterate `container.demux()` and only call `decode()` on `packet.is_keyframe`; set
     `codec_context.thread_type = "AUTO"` / `thread_count`; try `skip_frame = "NONKEY"` vs manual
     keyframe filtering.
   - **Resolution:** 112×112 vs 64×64 thumbnails for CLIP (accuracy vs cost).
   - **Overlap:** run the scan in a worker thread while the previous video's windows are decoded
     (pipelining across videos is legitimate for a stream; within one video the scan must precede
     selection).
   - **GPU decode (stretch):** `torchcodec` / NVDEC keyframe-only decode if the above stalls above
     80 ms.
3. Verify `scan_keyframes` returns identical timestamps before/after (unit test on the synthetic clip
   plus a check on 3 real videos).
4. Rerun `eval_thumos14` with uniform and kf-clip at 4/8/16; compare to 004.

## Metrics
- Scan ms per video (p50, p90) on the server; breakdown by stage.
- CLIP hit rate and top-1 unchanged (±1 pt) if thumbnail format changes.
- kf-clip total ms at 4/8/16 vs 004; the goal is kf-clip-4 ≤ 170 ms.

## Risks
- Grey or very small thumbnails may hurt CLIP's zero-shot scores; keep 112×112 RGB as the fallback.
- Pipelining across videos changes what "latency per video" means; report both per-video latency
  and throughput, and keep the per-video number as the headline.

## Results
*(to fill)*

## Decision
*(to fill)*
