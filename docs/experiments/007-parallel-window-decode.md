# 007 · Parallel windowed decode

**Date:** 2026-11-02 · **Status:** planned · **Results:** *(to fill)*

## Motivation (from 006)
Coarse-to-fine refinement is the accuracy lever that works (kf-clip-4 0.795 → 0.865, kf-clip-8 0.875 → 0.900),
but it costs 165–325 ms per video because every chosen window must now be decoded in full, and the decode path
is slow in a way that has nothing to do with pixels: OpenCV charges a flat 35–60 ms per seek and 0.3 ms per
320×180 frame, PyAV 0.9 ms per frame. Even without refinement, decoding the windows is the largest cost of
kf-clip (80–171 ms). The windows are independent, and 005 showed that independent time ranges scale across
the process pool (8 processes: 5.5× on the scan).

## Hypothesis
Decoding the chosen windows in parallel, one worker per window on the persistent process pool, brings the
window decode (with or without refinement) to ≈ the cost of the single longest window plus transfer:
~60 ms for four 5 s windows instead of 170–360 ms. With refinement that makes kf-clip-4 ≈ 0.865 at ≤ 170 ms
and kf-clip-8 ≈ 0.900 at ≤ 230 ms, i.e. the accuracy of 006 at the latency of 005, and it also cuts the
non-refined baseline decode by ~2× at 8 and 16 frames.

## Plan
1. `vam.runtime.decode.read_windows(path, windows, stride, thumb, keep_edge, workers, backend)`: each worker
   opens its own `cv2.VideoCapture`, seeks once to its window start, reads forward, and returns
   `(indices, thumbnails, kept frames)`; results are concatenated in window order. Same pool and backend
   selection as `scan_keyframes` (process where fork exists, else threads). Kept frames are returned as one
   uint8 array per window to keep pickling cheap (a 5 s window at stride 3 and 320×180 is ~9 MB).
2. `KeyframeSampler` uses it for the refinement pass, and `read()` uses it for the non-refined path too
   (one worker per window, each reading only up to its last wanted frame), so the baseline gains as well.
3. Refinement thumbnails: try `refine_stride` 3 vs 6 (halves the CLIP cost of ~49 ms per 200 thumbnails) and
   CLIP in fp16; check accuracy is within 1 point.
4. Unit tests: parallel window read equals sequential `Video.read` on the synthetic clip; stride and
   `keep_edge` honoured.
5. Rerun kf-clip at 4/8/16 with: no refinement (new decode), refinement (gap + softmax 0.2 + refine), and the
   stride/fp16 variants. Same JSONL schema.

## Metrics
Top-1, hit rate, select / refine / decode / infer / total ms. Success: kf-clip-4 ≥ 0.86 at ≤ 170 ms and
kf-clip-8 ≥ 0.90 at ≤ 230 ms (both within +20 ms of 005 with the 006 accuracy); non-refined decode ms ≥ 1.5×
lower at 8 and 16 frames.

## Risks
- Pickling frames back from worker processes may cost tens of ms; if so, return thumbnails only and have the
  parent re-read the chosen frames (one seek per window, ~50 ms each), or move the pool to threads for this
  stage (OpenCV releases the GIL during decode, unlike PyAV's frame conversion in 005).
- Seek cost is per worker and per window, so it does not shrink; if seeks dominate, the real fix is a decoder
  that seeks cheaply (NVDEC via torchcodec), which becomes 008.

## Results
*(to fill)*

## Decision
*(to fill)*
