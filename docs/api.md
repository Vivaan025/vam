# API reference (v0.1)

Every public name, its signature, one line of purpose, and the experiment that justifies it. Names marked
**core** are the ones a user needs; the rest are the runtime's parts, exposed for benchmarks and extension.
`tests/test_docs.py` checks that every name in each module's `__all__` appears here.

## `vam` (top level)

| name | signature | purpose | since |
|---|---|---|---|
| **`load_video`** | `load_video(path) -> Video` | Open a file lazily; counts frames by grabbing when the container lies (still images) | 001, 013 |
| **`Video`** | fields `path, fps, num_frames, width, height`; `duration`; `read(indices, seq_threshold=64) -> FrameBatch`; `iter_frames(start=0, stop=None, step=1)`; `close()` | A video source; random access seeks, sequential runs read forward | 001 |
| **`FrameBatch`** | fields `frames (T,H,W,3) uint8 or (T,3,H,W) float`, `timestamps (T,)`, `indices (T,)`; `to(device)`; `channels_first()`; `is_channels_first` | What a model receives | 001 |
| **`ActionOutput`** | fields `logits, labels, actions, features`; `top_k(k=5) -> [(label, prob)]` | What a model returns; `logits` for recognition, `actions` for a policy | 001, 014 |
| **`get_model`** | `get_model(name, **kwargs) -> VideoActionModel` | Build a registered model (`device=`, model-specific kwargs) | 001 |
| **`list_models`** | `list_models() -> list[str]` | Registered names | 001 |
| **`UniformSampler`** | `UniformSampler(num_frames)` | Evenly spaced frames; the baseline | 001 |
| **`MotionSampler`** | `MotionSampler(num_frames, stride=1, thumb=(64, 36))` | Frames where pixel motion is high; a negative result kept as the failure example | 001, 002 |
| **`KeyframeSampler`** | `KeyframeSampler(num_frames, scorer=None, window_s=5.0, segments=None, thumb=(112,112), scan_workers=None, keyframe_stride=1, scan_backend=None, window="gap", min_window_s=1.0, alloc="equal", temperature=0.2, refine=True, refine_stride=6, keep_edge=224, decode_backend="thread", decode_workers=None, planner=None, fallback="uniform", flat_eps=1e-3, max_coverage=0.5)` | Scan keyframes, score, pick windows, decode them in parallel, refine inside. `scorer=None` loads the packaged class-free head. After a call: `last_keyframes, last_scores, last_chosen, last_windows, last_fallback, last_refine_ms` | 003, 005, 006, 007, 009, 013, 015 |
| **`BudgetedSampler`** | `BudgetedSampler(budget_ms, model=None, calibrate_on=None, scorer=None, cost=None, margin=1.0, cache=True, repeats=2, accuracy=None, **keyframe_kwargs)` | A latency budget in ms: calibrates a cost model on a few clips (cached per machine), plans per video; `last_plan`, `cost`, `scheduler` | 010, 016 |
| **`Stream`** | `Stream(source, window=16, hop=None)` | Overlapping windows from a file or camera index; `hop > window` skips frames; raises `VideoOpenError` / `DecodeError` | 001, 014 |
| `Sampler` | abstract: `select(video) -> list[int]`, `read(video, indices) -> FrameBatch`, `__call__(video)` | Base class for samplers | 001 |
| `profile` / `Profiler` | `with profile(name) as p:`; `p.span(name)`; `p.report() -> str`; `p.as_dict()` | Wall ms and peak GPU MB per stage | 001 |
| `FeatureCache` | `FeatureCache(budget_mb=256)`; `get(key)`, `put(key, tensor)`, `hit_rate`, `stats()` | LRU cache of per-frame features under a memory budget | 001 |
| `VamError`, `VideoOpenError`, `DecodeError`, `ScanError` | exceptions | `VideoOpenError` is an `IOError`, the other two `ValueError`s; all `VamError` | 013 |
| `__version__` | `"0.1.0"` | | |

## `vam.sampling`

| name | signature | purpose | since |
|---|---|---|---|
| `ClipScorer` | `ClipScorer(class_names=(), model="openai/clip-vit-base-patch32", device=None, template="a photo of {}", use_hf_processor=False, half=True)`; `embed(thumbs) -> (K,512)`; `scores_from_embedding(e)`; `__call__(keyframes) -> (K,)` | Zero-shot CLIP keyframe scorer when given class names; with none, the embedder behind the learned scorer | 003, 015 |
| `LearnedScorer` | `LearnedScorer(embedder, checkpoint, device=None)`; `LearnedScorer.load(path, embedder, device=None)`; `for_video(video_id)` (benchmark: pick the fold head that did not train on it) | Class-free context head on CLIP embeddings, imitates the oracle's window-overlap score | 009 |
| `default_scorer` | `default_scorer(device=None, clip_model="openai/clip-vit-base-patch32") -> LearnedScorer` | The packaged head (`vam/weights/scorer_context_nozs.pt`, fp16 ensemble of the five fold heads) | 015 |
| `OracleKeyframeScorer` | `OracleKeyframeScorer(segments=[(start_s, end_s), ...], window_s=5.0)` | Upper bound: scores a keyframe by its window's overlap with the annotated action | 003 |
| `uniform_scores` | `uniform_scores(keyframes) -> (K,)` | Constant scores (the flat case) | 013 |
| `build_features`, `gap_window_targets` | `build_features(emb, gaps, zs, context)`; `gap_window_targets(times, segments, min_window_s, window_s)` | Feature and target construction for training the scorer (`scripts/train_scorer.py`) | 009 |

## `vam.runtime`

| name | signature | purpose | since |
|---|---|---|---|
| `scan_keyframes` | `scan_keyframes(path, thumb=(112,112), max_keyframes=None, workers=None, stride=1, backend=None) -> Keyframes` | Decode only keyframe packets, in parallel time ranges (`forkserver` pool, 120 s timeout) | 003, 005, 011, 013 |
| `Keyframes` | fields `times (K,), indices (K,), thumbs (K,H,W,3), fps, duration, num_frames` | The scan's result | 003 |
| `read_windows` | `read_windows(path, windows, thumb=None, keep_edge=224, workers=None, backend=None) -> list[WindowFrames]`; `Window`, `WindowFrames` | Decode several windows of one file in parallel (thread pool: OpenCV releases the GIL) | 007 |
| `shutdown_pools` | `shutdown_pools(wait=True)` | Tear down the worker pools (tests, clean exit) | 013 |
| `Preprocessor` | `Preprocessor(shortest_edge, crop, mean, std, device)`; `Preprocessor.from_hf(processor, device)`; `__call__(frames uint8) -> (T,3,H,W) float` | Resize / crop / normalise as batched tensor ops on the device, from a model's own parameters | 004 |
| `Config` | `Config(frames, refine, stride=6, uniform=False)`; `segments` | One pipeline configuration the scheduler can choose | 010, 013 |
| `ACCURACY` | `dict[Config, float]` | Measured top-1 per configuration (THUMOS14 val, VideoMAE); the scheduler's ordering | 010 |
| `CostModel` | `CostModel.fit(rows, machine="")`; `predict(features) -> ms`; `to_json()` / `from_json(d)`; fields `w, resid_p75, typical_K, typical_window, n_fit, machine` | Linear latency model: intercept + per-keyframe + per-window-frame + per-candidate + per-model-frame + refinement; NNLS on medians, p75 residual as margin | 010 |
| `calibrate` | `calibrate(videos, run, configs=None, cache=True, repeats=2) -> CostModel` | Fit on the target machine; `run(cfg, video) -> (Features, ms)`; cached in `~/.cache/vam/` | 010 |
| `Scheduler` | `Scheduler(budget_ms, cost, accuracy=None, margin=1.0, min_config=Config(4, False))`; `pre_plan(duration_s, fps) -> Plan`; `post_plan(K, window_frames, flat=False) -> Plan`; `planner(flat_eps=1e-3, max_coverage=0.5)`; `last_plan` | Pick the most accurate configuration that fits, re-plan after the scan; uniform when the scan is flat or windows would cover > half the clip | 010, 013 |
| `BudgetedSampler` | see top level | | 016 |
| `Profiler`, `profile`, `FeatureCache` | see top level | | |

## `vam.models`

| name | signature | purpose | since |
|---|---|---|---|
| `VideoActionModel` | `default_num_frames`; abstract `forward_frames(batch) -> ActionOutput`; `__call__(video_or_batch, sampler=None)` | The contract. Given a `Video` it applies the sampler (default uniform) first | 001 |
| `register` | `@register(name)` on a factory `fn(**kw) -> VideoActionModel` | Add a model to `get_model` | 001 |
| `HFVideoClassifier` | `HFVideoClassifier(name, device=None, dtype=float16, use_hf_processor=False)` | Any `AutoModelForVideoClassification`; registered: `videomae-base-kinetics`, `videomae-small-kinetics`, `timesformer-base-k400` | 001, 004 |
| `HFFeatureExtractor`, `LinearProbe` | `HFFeatureExtractor(name, ...)`.`features(batch)`; `LinearProbe(extractor, num_classes, class_names=None)`; `LinearProbe.fit_head(features, labels, num_classes, epochs=100, lr=1e-2, weight_decay=1e-4, device="cpu")`; `save_head(path)` / `load_head(path)` | Pooled backbone features and a linear head, how every benchmark measures accuracy | 001 |
| `FrameExitModel` | `FrameExitModel(dataset='activitynet', checkpoint=None, device=None, dtype=float16)`; `forward_frames(batch)` pools K frames and answers with the head trained for K; `exit_forward(batch)` runs FrameExit's own early exit; `frame_features(batch)` | the shared ResNet-50 classifier of the prior-work comparison; registered `frameexit-activitynet`, `frameexit-minikinetics`; checkpoints from the FrameExit release (`VAM_FRAMEEXIT_DIR`) | 017 |
| `TorchvisionVideoClassifier` | `TorchvisionVideoClassifier(arch, device=None, dtype=float16)`; `features(batch)`; `feature_dim` | Kinetics-400 R(2+1)D-18, R3D-18, MViT-v2-S; registered `r2plus1d-18-k400`, `r3d-18-k400`, `mvit-v2-s-k400` | 014 |
| `ActionPolicy` | `ActionPolicy(instruction="", state=None)`; implement `act(frames, instruction, state) -> (T, action_dim)`; `forward_frames(batch, instruction=None, state=None)` | A policy is a `VideoActionModel` with `default_num_frames=1` that fills `actions` | 014 |
| `FakePolicy` | `FakePolicy(action_dim=7, **kw)`; registered `fake-policy` | Deterministic stand-in for tests and calibration | 014 |
| `OpenVLAPolicy` | `OpenVLAPolicy(name="openvla/openvla-7b", unnorm_key="bridge_orig", device=None, dtype=bfloat16, attn_implementation="eager")`; registered `openvla-7b` | One frame + instruction to a 7-D end-effector action; needs transformers 4.40.x | 014 |

## `vam.data`

| name | signature | purpose | since |
|---|---|---|---|
| `UCF101`, `Clip` | `UCF101(root, split="train", fold=1, limit=None, seed=0)`; iterates `Clip(path, label, ...)` with `.video()` | Trimmed clips; the probe's training set | 001 |
| `THUMOS14`, `UntrimmedVideo`, `Segment` | `THUMOS14(root, split="val", require_video=True)`; `UntrimmedVideo(path, video_id, segments)` with `.video()`, `.classes`, `.action_fraction(duration)`; `Segment(class_name, start, end)` | Untrimmed long video with temporal annotations; the main benchmark | 002 |
| `ActivityNet` | `ActivityNet(root, split='validation')`; `.videos`, `.classes`, `.labels(uv)`, `.missing` | ActivityNet-1.3 untrimmed videos with multi-label annotations in FrameExit's class order | 017 |
| `MiniKinetics` | `MiniKinetics(root, split='val')`; same interface | the AR-Net 200-class subset of Kinetics-400 val, trimmed 10 s clips | 017 |
| `Charades` | `Charades(root, split="test", require_video=True, min_segments=1)` | Second domain (the negative result); THUMOS14-shaped | 012 |

Download helpers live in `scripts/` (`download_thumos14.py`, `download_charades.py`); the benchmarks in
`benchmarks/` write JSONL with the GPU name so runs on different machines can be compared.

## `vam.errors`

`VamError`, `VideoOpenError(VamError, IOError)`, `DecodeError(VamError, ValueError)`, `ScanError(VamError, ValueError)`. Since 013.
