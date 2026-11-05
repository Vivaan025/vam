# Experiment 014 results: a second recognition family and a policy through the same runtime

**Date:** 2026-11-05 · **GPU:** RTX 5090 (24-core host, shared) · THUMOS14 val (200 videos) with torchvision's
R(2+1)D-18 (Kinetics-400) and a UCF101 linear probe (2,000 train clips, 0.850 top-1 on 200 UCF101 test clips at
16 frames); OpenVLA-7B (bfloat16, eager attention) on two BridgeData V2 episodes (LeRobot export, 5 fps, 7-D
end-effector actions) and one Charades clip. Raw rows: `results/thumos14_r2plus1d.jsonl`,
`results/thumos14_scheduler_r2plus1d.jsonl`, `results/policy.jsonl` (server). Per-row summary:
`model_coverage_rtx5090.csv`. Benchmarks: `eval_thumos14.py --backbone r2plus1d-18-k400`, `eval_scheduler.py`,
`eval_policy.py`.

## H1: a second model family, no runtime change

The scorer head, the gap windows, the refinement pass and the scheduler's accuracy table were all built on
VideoMAE and never see the model. Swapping in R(2+1)D-18 (112² crops, its own normalisation, 16-frame clips)
takes a 110-line adapter and nothing else.

| sampler | 4 frames | 8 frames | 16 frames |
|---|---|---|---|
| uniform | 0.565 @ 135 ms | 0.520 @ 271 ms | 0.545 @ 498 ms |
| **kf-learned + refine** | **0.655 @ 140 ms** | **0.675 @ 150 ms** | **0.700 @ 147 ms** |
| gain | +9.0 | +15.5 | +15.5 |

Medians over 200 videos, select + decode + inference. Uniform's cost is decode (120 / 256 / 494 ms), the keyframe
path's is select (137–139 ms) with decode under 1 ms and inference 3 ms. The absolute accuracies are lower than
VideoMAE's (0.865 / 0.925 / 0.940) because R(2+1)D-18 is a weaker backbone (0.850 vs 0.93 on the UCF101 probe),
but the selection gain is the same size: VideoMAE +14.5 / +13.0 / +6.0, R(2+1)D +9.0 / +15.5 / +15.5. The metric
in the plan (≥ +5 at 4 frames, runtime within 10 % of VideoMAE's) is met.

Scheduler, same cost-model recipe (24 rows, 4.9 s; intercept 62 ms, 0.41 ms per window frame, 0.15 per
refinement candidate, 0 per model frame, p75 residual 21 ms):

| budget | chosen configurations | top-1 | within budget | p50 / p90 ms | best static |
|---|---|---|---|---|---|
| 100 | 4f ×197, uniform ×2 | 0.645 | 0.26 | 123 / 178 | 4f 0.655 |
| 150 | 16f ×160, 16f+refine ×26, uniform ×14 | 0.690 | 0.60 | 136 / 194 | 16f+refine 0.700 |
| 200 | 16f+refine/3 ×186, uniform ×14 | 0.700 | 0.83 | 156 / 213 | 16f+refine 0.700 |

The scheduler's accuracy table is still the VideoMAE one from 009; its *ordering* transfers (4f < 16f < 16f +
refine holds for R(2+1)D too), so the choices are right even though the predicted accuracies are not. A per-backbone
table is a one-line change if a user cares about the absolute number. Uniform-16 would be 0.545 at every budget.

## H2: a policy fits the same contract

`OpenVLAPolicy` is an `ActionPolicy`, which is a `VideoActionModel`: `vam.Stream(clip, window=1, hop=1)` feeds it
one frame at a time and it answers `ActionOutput(actions=(1, 7))`. No special path. The benchmark compares the
predicted action at each recorded frame with the action the human teleoperator actually took at that frame
(open-loop imitation agreement on the recorded observations; not a closed-loop success rate).

| clip | instruction | steps | ms / step p50 (p90) | corr Δx / Δy / Δz | sign agree xyz | gripper agree | gripper flips pred / true |
|---|---|---|---|---|---|---|---|
| Bridge episode 0 | put small spoon from basket to tray | 26 | 113 (212) | 0.97 / 0.97 / 0.96 | 0.97 | 0.85 | 3 / 2 |
| Bridge episode 1 | put the blue cube … on top of the rectangular block | 45 | 115 (213) | 0.90 / 0.88 / 0.71 | 0.74 | 0.78 | 10 / 6 |
| Charades 001YG (control) | pick up the object | 60 | 114 (211) | – | – | – | 21 / – |

On the data OpenVLA was trained on, the translation deltas it predicts track the recorded ones (ρ ≈ 0.97 on the
first episode) and the gripper opens and closes where the human did. On the indoor Charades clip, which is not
a robot scene, the gripper channel flips 21 times in 60 steps: the model emits actions, as the contract requires,
and they are incoherent, as the wrong domain predicts. Peak memory 14.8 GB; 7 action tokens decoded
autoregressively cost 113 ms a step on the 5090, so a 5 fps Bridge stream runs at 1.7× real time.

## H3: the runtime's levers apply

The cost model, fitted with the policy as the model over `Config(1 / 2 / 4 frames)`, comes out as **11 ms + 114 ms
per frame** with zero residual: the policy's cost is exactly linear in frames, the per-frame slope is the per-step
latency above, and the same `calibrate` code that fitted VideoMAE and R(2+1)D fits it. `FeatureCache` cannot help a
single-frame policy and was not measured (the plan said so); the honest statement is that nothing above the model
changed.

## What broke on the way (and what it taught the library)

- **OpenVLA's remote code is pinned to transformers 4.40.** On the server's 5.18 it fails at import (the
  `AutoModelForVision2Seq` entry point was removed; tokenisation helpers moved). The adapter resolves the class
  from the checkpoint's `auto_map` when the entry point is missing and spells `torch_dtype` / `dtype` per
  version, but the remote code itself still needs 4.40, so the policy runs from a second venv inside the project
  folder (`.venv-vla`: transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10, numpy < 2, pyarrow < 16, same torch).
  `vam` itself keeps no transformers pin.
- **`predict_action` extends `input_ids` with an end token but not `attention_mask`**, and the eager Llama path
  then dies on a (T) vs (T−1) causal-mask mismatch. One unpadded sequence needs no mask; the adapter drops it.
- **The LeRobot export is AV1**, which the pip OpenCV build cannot decode. `Stream` opened the file and yielded
  zero frames, so the benchmark crashed on an empty stack. `Stream` now raises `VideoOpenError` /
  `DecodeError` (013's errors) when a source opens but decodes nothing; the episodes were re-encoded to H.264 for
  the run (`data/bridge_h264/`, parquet and meta symlinked).

## Decision

R5 done. Six models through one contract: VideoMAE (HF), R(2+1)D-18 / R3D-18 / MViT-v2-S (torchvision), OpenVLA-7B,
and a deterministic `FakePolicy` for tests. The selection gain is model-agnostic in the one way that matters (it
survives a change of backbone family), and a policy runs through `Stream` and the cost model untouched. MViT-v2-S
and R3D-18 are registered and tested for shape but not benchmarked on THUMOS14; one second family is the claim.
Next: R6 packaging (wheel, pinned ranges, CI on Linux and Windows) and R7 docs.
