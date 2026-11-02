# Experiment 007 results: parallel windowed decode

**Date:** 2026-11-02 · **GPU:** RTX 5090 (24-core host) · setup identical to `thumos14_adaptive.md` (200 THUMOS14 val
videos, VideoMAE-base + UCF101 linear probe, 8-process keyframe scan, gap window). Raw data:
`thumos14_pdecode_rtx5090.csv`. 006 reference rows: gap (serial decode) and all 0.2 (serial refinement).

Arms: `process` / `thread` = windows decoded in parallel on the process pool / a thread pool; `refine` = gap +
softmax 0.2 + coarse-to-fine, stride 3 unless noted; `s6` = refinement stride 6 (half the thumbnails);
`fp16` = CLIP scorer in half precision. `refine ms` is inside `select ms`.

| frames | arm | top-1 | hit | select ms | of which refine | decode ms | infer ms | total ms |
|---|---|---|---|---|---|---|---|---|
| 4 | 006 gap, serial decode | 0.815 | 0.44 | 47 | – | 80 | 20 | 148 |
| 4 | process | 0.815 | 0.44 | 48 | – | 84 | 20 | 152 |
| 4 | thread | 0.815 | 0.44 | 47 | – | 88 | 6 | 141 |
| 4 | 006 refine, serial | 0.865 | 0.45 | 213 | 167 | 0 | 5 | 219 |
| 4 | refine, process | 0.865 | 0.45 | 171 | 123 | 0 | 6 | 177 |
| 4 | refine, thread | **0.865** | 0.45 | 163 | 116 | 0 | 6 | **169** |
| 4 | refine s6, thread | 0.850 | 0.45 | 153 | 106 | 0 | 6 | 160 |
| 4 | refine s6 fp16, thread | **0.850** | 0.45 | 140 | 99 | 0 | 6 | **146** |
| 8 | 006 gap, serial decode | 0.885 | 0.42 | 46 | – | 159 | 19 | 223 |
| 8 | process | 0.885 | 0.42 | 47 | – | 87 | 20 | 154 |
| 8 | thread | 0.885 | 0.42 | 46 | – | 96 | 5 | **147** |
| 8 | 006 refine, serial | 0.900 | 0.44 | 373 | 326 | 0 | 5 | 378 |
| 8 | refine, process | 0.900 | 0.44 | 195 | 148 | 0 | 5 | 200 |
| 8 | refine, thread | **0.900** | 0.44 | 191 | 144 | 0 | 5 | **196** |
| 8 | refine s6, thread | 0.890 | 0.44 | 168 | 121 | 0 | 5 | 173 |
| 8 | refine s6 fp16, thread | **0.895** | 0.44 | 147 | 107 | 0 | 5 | **152** |
| 16 | 006 gap, serial decode | 0.935 | 0.41 | 46 | – | 171 | 8 | 225 |
| 16 | process | 0.935 | 0.41 | 47 | – | 92 | 8 | 147 |
| 16 | thread | **0.935** | 0.41 | 46 | – | 99 | 4 | **150** |
| 16 | 006 refine, serial | 0.925 | 0.44 | 372 | 326 | 0 | 4 | 377 |
| 16 | refine, process | 0.925 | 0.44 | 195 | 147 | 1 | 4 | 199 |
| 16 | refine, thread | 0.925 | 0.44 | 190 | 144 | 0 | 4 | 195 |
| 16 | refine s6, thread | 0.930 | 0.44 | 168 | 121 | 0 | 4 | 172 |
| 16 | refine s6 fp16, thread | **0.930** | 0.44 | 147 | 107 | 0 | 4 | **151** |

## Findings

1. **Parallel windows remove the budget dependence of decode.** Without refinement, decode is 84–99 ms at every
   budget (it was 80 / 159 / 171 ms serially): one seek plus one short read per window, all windows at once. The
   floor is the seek (35–60 ms) plus the longest window; two windows at 4 frames gain nothing, four windows at
   8 and 16 frames gain ~1.8×.
2. **Threads are as good as processes here, and slightly better end to end** (141 / 147 / 150 vs 152 / 154 /
   147 ms). OpenCV releases the GIL while decoding, unlike PyAV's frame conversion in the scan (005), so the
   thread pool scales for this stage and does not need fork. Threads become the default for the window decode;
   the scan keeps processes.
3. **Refinement is now affordable.** Serial refinement cost 167 / 326 / 326 ms; parallel it costs 116 / 144 /
   144 ms at stride 3, and 99 / 107 / 107 ms at stride 6 with fp16 CLIP. The remaining cost is the seek plus
   decoding whole windows (~95 ms for four 5 s windows in parallel) and CLIP on 100–200 thumbnails.
4. **Stride 6 loses ≤ 1.5 points** (0.850 / 0.890 / 0.930 vs 0.865 / 0.900 / 0.925) and **fp16 loses
   nothing** (0.850 / 0.895 / 0.930). The stride-6 fp16 configuration is the new recommended default: within
   5 ms of the 005 kf-clip-4 latency at every budget while scoring 5.5 / 2.0 / −0.5 points above it.
5. **The frontier, same 200 videos** (ms per video; uniform rows from 005):

   | policy | 4 frames | 8 frames | 16 frames |
   |---|---|---|---|
   | uniform | 0.720 @ 147 | 0.795 @ 277 | 0.880 @ 486 |
   | kf-clip, 005 | 0.795 @ 152 | 0.875 @ 230 | 0.935 @ 210 |
   | kf-clip, gap, parallel decode | 0.815 @ 141 | 0.885 @ 147 | 0.935 @ 150 |
   | kf-clip + refine s6 fp16 | 0.850 @ 146 | 0.895 @ 152 | 0.930 @ 151 |
   | kf-clip + refine s3 | 0.865 @ 169 | 0.900 @ 196 | 0.925 @ 195 |
   | oracle (upper bound) | 0.890 | 0.925 | 0.955 |

   At the latency of uniform-4 (147 ms) the runtime now delivers 0.930–0.935, i.e. more than uniform-16 at
   3.2× less time, and with refinement at 4 frames it is 2.5–4 points below the oracle.
6. **Unexplained:** the inference span is 20 ms when frames arrive from the process pool and 4–6 ms when they
   come from threads or the refinement cache, at every budget. The frames are byte-identical (tests), so this is
   a handoff effect (pickled arrays, or GPU clock state after an idle wait), not model work. Threads avoid it.

## Experiment 007 hypothesis check

- kf-clip-4 ≥ 0.86 at ≤ 170 ms: ✔ (0.865 at 169 ms, stride 3, threads).
- kf-clip-8 ≥ 0.90 at ≤ 230 ms: ✔ (0.900 at 196 ms).
- Non-refined decode ≥ 1.5× lower at 8 and 16 frames: ✔ (159 → 87–96, 171 → 92–99 ms).
