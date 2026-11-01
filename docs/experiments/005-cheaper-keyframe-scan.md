# 005 · Cheaper keyframe scan

**Date:** 2026-11-01 · **Status:** in progress · **Results:** *(to fill)*

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

## Profiling (step 1, server, 320×180 H.264)

| video | keyframes | packets | demux | decode keyframes | convert 112² RGB | scan_keyframes() |
|---|---|---|---|---|---|---|
| 0157 | 7 | 1,510 | 4 ms | 9 ms | 4 ms | 25 ms |
| 0165 | 31 | 3,192 | 25 ms | 49 ms | 3 ms | 101 ms |
| 0169 | 98 | 5,826 | 46 ms | 137 ms | 6 ms | 322 ms |

- The cost is **decoding the keyframes themselves, ~1.4 ms each** (measured in isolation, demux
  excluded), plus demux at ~8 µs/packet. Thumbnail conversion is negligible on the server (the
  laptop number in 003 was misleading).
- Decoder options do nothing: `thread_count=1`, `skip_loop_filter=all`, `skip_idct=all`, SLICE
  threading all give 164–165 ms on the 98-keyframe video. So the ~1.4 ms is per-packet overhead
  (avcodec send/receive + PyAV frame objects), not arithmetic.
- An `ffmpeg` subprocess pipe (`-skip_frame nokey -fps_mode passthrough`) is *slower*: 233–277 ms,
  dominated by process start and pipe I/O. (First attempt without `-fps_mode passthrough` emitted all
  5,640 frames: ffmpeg duplicates frames to keep a constant rate.)
- **Parallel time ranges.** Split the timeline into N ranges, seek each worker to its start, decode
  keyframes until its end; keyframe sets verified identical to the serial scan.

| workers | threads (0169 / 0165 / 0157) | processes (0169 / 0165 / 0157) |
|---|---|---|
| 1 | 242 / 78 / 17 ms | – |
| 2 | – | 138 / 63 / 21 ms |
| 4 | 85 / 58 / 38 ms | 76 / 45 / 15 ms |
| 8 | 142 / 114 / 116 ms (slower) | **44 / 38 / 15 ms** |

  Threads stop scaling at 4 and get *slower* beyond: decode releases the GIL but frame conversion
  and object creation hold it, and 8 threads convoy on it. An 8-thread pool across 8 videos was
  0.63× sequential. Processes scale: 8 processes give 5.5× on the worst video and **6.2× throughput**
  across 8 videos (645 → 104 ms). Per-worker fixed cost (open + seek + first frame) is ~6 ms, which
  is why the 7-keyframe video gains nothing.

## Changes made
- `scan_keyframes(..., workers=None, stride=1, backend="process")`: timeline split into `workers`
  ranges (default `min(8, cpu_count-1)`), each scanned by `_scan_range` with `thread_count=1`; a
  persistent `ProcessPoolExecutor` (fork on Linux) is created lazily and reused; `backend="thread"` and
  `"serial"` kept for A/B. `stride` keeps every k-th keyframe.
- `KeyframeSampler(scan_workers, keyframe_stride, scan_backend)` and `eval_thumos14.py`
  `--scan-workers / --kf-stride / --scan-backend`.
- Tests: thread and process backends produce keyframes and thumbnails identical to serial; stride=2
  keeps every other keyframe.

## Results
*(to fill after the server run)*

## Decision
*(to fill)*
