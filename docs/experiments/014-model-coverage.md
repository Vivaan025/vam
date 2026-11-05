# 014 · Model coverage: a second recognition family and a policy / VLA adapter

**Date:** 2026-11-05 · **Status:** planned · **Results:** *(to fill)*

## Motivation (roadmap R5)
The README claims one output type serves both meanings of "action model": recognition fills `logits`, agent
policies fill `actions`, and everything above the model (samplers, scheduler, cache, stream) works for both.
So far only one adapter exists (Hugging Face `AutoModelForVideoClassification`), so the claim is asserted, not
demonstrated. This entry adds a second recognition family with its own preprocessing conventions and a
vision-language-action policy, and runs the same runtime over all three.

## Hypotheses
- **H1, a second family needs no runtime change.** torchvision's Kinetics-400 video models (R(2+1)D-18,
  MViT-v2-S) plug in as `VideoActionModel`s with their own normalisation and frame counts, and the THUMOS14
  benchmark runs unchanged on them: `kf-learned` beats uniform on the new backbone too (the scorer and the
  scheduler never see the model), with the same ~150 ms runtime and the model's own inference cost on top.
- **H2, a policy fits the same contract.** An `ActionPolicy` adapter takes a `FrameBatch` (plus an
  instruction and optional proprioceptive state) and returns `ActionOutput(actions=(T, action_dim))`.
  `OpenVLA-7B` (`openvla/openvla-7b`, Hugging Face, Apache-2) is the concrete policy: image + instruction →
  7-D end-effector action tokens, decoded to continuous values. It runs on the 5090 in bfloat16 (~15 GB) and
  `vam.Stream` feeds it frames without a special path.
- **H3, the runtime's levers apply to policies.** For a policy the sampler question becomes "which recent
  frame(s) to act on"; `FeatureCache` reuse and the scheduler's cost model apply unchanged. Measured as
  actions per second for OpenVLA under `Stream(hop=1)` with and without the cache, on a robot-manipulation clip
  (BridgeData V2 sample, the data OpenVLA was trained on) and on an unrelated indoor clip (Charades) as the
  "wrong domain" control.

## Plan
1. `vam/models/torchvision.py`: `TorchvisionVideoClassifier(name)` for `r2plus1d_18`, `mvit_v2_s` with the
   weights' own transforms (mean/std, 112² or 224², 16 frames), registered as `r2plus1d-18-k400`,
   `mvit-v2-s-k400`; a `features()` method so `LinearProbe` works on it (the probe is refitted on UCF101
   features per backbone by `benchmarks/eval_ucf101.py --backbone`).
2. `vam/models/policy.py`: `ActionPolicy(VideoActionModel)` base (`forward_frames` → `actions`, optional
   `instruction`, `state`), `OpenVLAPolicy(name="openvla/openvla-7b", unnorm_key="bridge_orig")` using
   `AutoModelForVision2Seq` with `trust_remote_code`, and a `FakePolicy` for tests (returns a deterministic
   function of the frame mean), registered as `openvla-7b`.
3. `benchmarks/eval_thumos14.py --backbone r2plus1d-18-k400 | mvit-v2-s-k400` with `--head` from the UCF101
   probe fit; samplers uniform, kf-learned (+refine) at 4 / 8 / 16; the scheduler at 150 / 200 ms.
4. `benchmarks/eval_policy.py`: OpenVLA over `Stream(window=1, hop=1)` on two clips; logs per-step latency,
   actions, and the gripper/translation trajectory; with `FeatureCache` keyed by frame index as a no-op check
   (the policy is single-frame, so the cache cannot help; H3's claim is that nothing breaks, and the cost model
   calibrates).
5. Tests: torchvision adapter on a synthetic clip (output shape, labels, probe features); `FakePolicy` through
   `Stream`, `ActionOutput.actions` shape, scheduler calibration with a policy as the model.

## Metrics
H1: on THUMOS14 val with the new backbone, kf-learned − uniform ≥ +5 points at 4 frames (VideoMAE: +14.5), runtime
select + decode within 10 % of VideoMAE's. H2: OpenVLA produces 7-D actions for every frame of both clips; on the
Bridge clip the gripper channel flips at least once (it is a pick-and-place), on the Charades clip the actions
are reported as-is (no ground truth). H3: actions per second with `Stream(hop=1)`, cost model fitted.

## Risks
- OpenVLA needs `timm`, `tokenizers`, and 15 GB of weights; `flash_attn` is optional. Install into the project
  venv on the server only. If the remote code fails on transformers 4.57, fall back to `openvla/openvla-7b` via
  its own `predict_action` with `attn_implementation="eager"`.
- A policy has no accuracy metric here; the result is "it runs through the same contract and the runtime's cost
  model covers it", stated as such.
- torchvision MViT needs 16 frames at 224²; R(2+1)D wants 112². Both are what `Preprocessor` already handles
  with explicit parameters.

## Results
*(to fill)*

## Decision
*(to fill)*
