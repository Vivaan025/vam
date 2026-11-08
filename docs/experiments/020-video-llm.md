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
pending

## Decision
pending
