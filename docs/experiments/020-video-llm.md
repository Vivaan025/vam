# 020 · Frame selection for a video-LLM: Video-MME under a frame budget

*Opened 2026-11-09 · Status: in progress*

## Motivation
The largest group of people who pick a handful of frames from a long video today are video-LLM users, and
almost all of them sample uniformly. `vam` has an API for exactly that step and a sampler that wins on
low-coverage long video (THUMOS14, +17 at 4 frames) and loses nowhere once it falls back. A video-LLM
benchmark is the demo that audience understands, and the first result of its kind for the library: same
model, same frame budget, same prompt, uniform vs `vam` frames, accuracy and milliseconds.

Video-MME (Fu et al., 2024): 900 YouTube videos in three duration buckets (short < 2 min, medium 4–15 min,
long 30–60 min), 2,700 four-way multiple-choice questions, many of which depend on one moment of the video.
Ungated on Hugging Face (`lmms-lab/Video-MME`, 101 GB). No segment annotations, so there is no oracle row;
the comparison is uniform vs `vam` at matched frame counts, plus the measured cost of each.

Model: `Qwen/Qwen3-VL-8B-Instruct` (ungated; transformers 5.18 has it), bf16 on the RTX 5090, frames
resized to ≤ 448 px on the long side, timestamps passed to the processor from the selected frames' times
(Qwen3-VL inserts them in the prompt, so irregularly spaced frames are told apart from uniform ones).

## Hypotheses (acceptance criteria)
- **H1, long bucket.** At 8 and 16 frames, `vam` keyframe frames beat uniform frames on the long bucket by
  ≥ 2 accuracy points. (Questions there are mostly about one moment in a 30–60 min video: the THUMOS14 shape.)
- **H2, no loss where coverage is high.** On the short bucket the difference is within ±1 point (the sampler
  should fall back to uniform on most 1–2 min clips).
- **H3, cost.** Select + decode for `vam` on long videos is at most the cost of uniform's seek-decode of the
  same number of frames (16 seeks into a 1-hour file are not cheap), and the whole pipeline's time is
  dominated by generation either way; report both.
- **H4, the budget sampler is usable here.** `BudgetedSampler(ms)` with the VideoLLM as the model reaches
  the long-bucket accuracy of the best fixed arm at its own measured ms.

## Plan
1. `vam/models/videollm.py`: `VideoLLM` adapter (Qwen3-VL through transformers; `answer(batch, question,
   options)`; `ActionOutput.text`). Registered as `qwen3-vl-8b`.
2. `vam/data/videomme.py`: `VideoMME(root)`: questions grouped by video, duration bucket, task type.
3. `benchmarks/eval_videomme.py`: arms `uniform-N`, `keyframe-N`, `kfnr-N`, `budget-B`; one sampling per
   video, three generations; per-question JSON rows; accuracy by bucket and task type; ms by stage.
4. Smoke on 30 videos, then the full 900 at N = 8 and 16 (≈ 2,700 × 2 × 3 generations).
5. Record; if H1 holds, this is the README's first example.
6. (added 2026-11-09 after the first run) **H5, hybrid:** N/2 evenly spaced anchors + N/2 keyframe picks
   (`hybrid-N`) is within 1 point of uniform-N overall and above it on the long bucket; plus `kfnr-16` (no
   refinement, so the 16 frames spread over 16 windows instead of 8) and `uniform-32` as the reference.

## Metrics
Accuracy (letter match) overall and per bucket (short / medium / long) and per task type; select, decode and
generate ms per video (p50 / p90); fallback counts.

## Risks
- Video-MME questions in the long bucket may rely on subtitles or on global content, not on one moment; then
  no selector helps and H1 fails for the benchmark's reason, not the sampler's. The per-task-type breakdown
  (Counting, Temporal Reasoning, Information Synopsis, ...) will say which.
- Qwen3-VL's own frame handling (temporal patch of 2, 768 max video tokens) may prefer evenly spaced frames;
  the timestamps mitigate this. If accuracy at 16 uniform frames is far below the published ~70 %, the
  adapter is wrong before the sampler is judged.
- 101 GB download and ~3 h of generation per arm.

## Results
### 2026-11-09: full run, 900 videos / 2,700 questions, Qwen3-VL-8B-Instruct bf16, frames ≤ 448 px, RTX 5090
| arm | acc | short | medium | long | select p50 | decode p50 | generate p50 | fallback |
|---|---|---|---|---|---|---|---|---|
| uniform-8 | **0.559** | 0.649 | 0.527 | 0.501 | 0 | 119 | 113 | – |
| keyframe-8 | 0.523 | 0.612 | 0.477 | 0.481 | 283 | 0 | 83 | 12 |
| uniform-16 | **0.606** | 0.724 | 0.564 | 0.528 | 0 | 233 | 167 | – |
| keyframe-16 | 0.529 | 0.626 | 0.482 | 0.480 | 284 | 0 | 129 | 12 |

Per task type at 16 frames (uniform / keyframe): Action Reasoning 0.593 / 0.519, Action Recognition 0.569 /
0.537, Attribute Perception 0.739 / 0.590, Counting 0.377 / 0.306, Information Synopsis 0.759 / 0.678, OCR
0.619 / 0.532, Object Reasoning 0.593 / 0.535, Object Recognition 0.638 / 0.551, Spatial Perception 0.667 /
0.593, Spatial Reasoning 0.732 / 0.714, Temporal Perception 0.673 / 0.545, Temporal Reasoning 0.469 / 0.379.
Per question at 16 frames: both right 1,253, both wrong 889, uniform only 382, keyframe only 176.

- **H1 fails, everywhere.** Keyframe frames lose 3.6 points at 8 frames and 7.7 at 16, on every bucket and
  on 11 of 12 task types (Spatial Reasoning is the exception, +1.8 at 8 frames, 69 questions). Going from 8 to
  16 keyframe frames gains 0.6 points; uniform gains 4.7. The sampler fell back on only 12 of 900 videos, so
  this is the selection policy itself, not a fallback artefact.
- **Why:** Video-MME questions are about the video as a whole or about a moment the asker chose, not about
  the "most action-like" moment. The class-free scorer was trained (009) to find THUMOS14 action windows; on
  a lecture or a documentary it concentrates frames in a few windows (refinement puts two frames 1–2 s apart
  in each) and the model loses sight of the rest. The task types that collapse most, Attribute Perception,
  Temporal Perception and Information Synopsis, are exactly the ones that need the whole timeline.
- **H2 fails** too: the short bucket loses 3.7 / 9.8, because the sampler does not fall back there (1–2 min
  clips are long enough for a scan) and concentrates frames just the same.
- **H3 holds on the decode side only.** Selection + decode is 283 ms for keyframe (scan + CLIP on every
  keyframe of up to an hour of video + windowed decode) against 119 / 233 ms for uniform's seeks; generation
  is 25–30 ms cheaper with keyframe frames (fewer visual tokens after resizing, since the chosen frames are
  often the same shot), which does not matter next to the accuracy loss.
- **Reference check:** uniform-16 at 448 px gives 0.606; Qwen3-VL-8B's reported Video-MME score (without
  subtitles, many more frames at higher resolution) is around 0.71, so the adapter is in the expected range
  for 16 small frames and the timestamps are being used (keyframe prompts carry irregular times).
- H4 (budget sampler) was not run: the fixed arms settle the question.


## Decision
pending
