# Quickstart

`vam` runs video-action models (recognition models that answer *what is happening*, and policies that answer
*what to do*) through one contract, and decides which frames a model sees so long video stays cheap. The
numbers behind every claim are in [`results/`](results/) and the log of how they were obtained in
[`experiments/`](experiments/README.md).

## Install

```bash
pip install vam-runtime[hf]          # import name: vam
```

`[hf]` adds Hugging Face `transformers`, which the bundled classifiers and the packaged keyframe scorer use.
Other extras: `[torchvision]` (R(2+1)D, R3D, MViT), `[policy]` (OpenVLA; see the note at the end), `[all]`.
CPU is enough for everything on this page; a GPU makes the models fast. Python 3.10 to 3.12, Linux, Windows,
macOS (CPU).

No clip at hand? Make a 30 s synthetic one (still scene, then a moving square in the last third):

```bash
python -m examples.make_sample sample.mp4      # from a checkout, or copy examples/make_sample.py
```

## 1. Run a classifier

```python
import vam

video = vam.load_video("sample.mp4")             # lazy: nothing is decoded yet
model = vam.get_model("videomae-base-kinetics")  # 400 Kinetics classes; downloads 350 MB once
out = model(video)                               # 16 evenly spaced frames by default
print(out.top_k(3))
```

`out` is an `ActionOutput`: `logits` and `labels` for recognition, `actions` for a policy. `vam.list_models()`
shows what is registered.

## 2. Choose the frames

The sampler decides which frames the model sees. Uniform is the baseline every paper uses. The keyframe sampler
scans the file's keyframes (cheap: no inter-frame decoding), scores them with a small class-free head on CLIP
embeddings, decodes short windows around the best ones in parallel, and refines inside them.

```python
out = model(video, sampler=vam.UniformSampler(8))
out = model(video, sampler=vam.KeyframeSampler(8))    # packaged scorer; CLIP ViT-B/32 downloads once (600 MB)
```

On 200 untrimmed THUMOS14 sports videos (3 minutes on average, the action covers 29 % of each) with a linear
probe on VideoMAE, on an RTX 5090:

| sampler | 4 frames | 8 frames | 16 frames |
|---|---|---|---|
| `UniformSampler` | 0.720 @ 147 ms | 0.795 @ 277 ms | 0.880 @ 486 ms |
| `KeyframeSampler` | **0.865 @ 143 ms** | **0.925 @ 153 ms** | **0.940 @ 152 ms** |
| oracle (reads the annotations) | 0.890 | 0.925 | 0.955 |

The same gain holds on a held-out split, on a second backbone family (R(2+1)D), and across H.264, HEVC and VP9
at 180p to 1080p. It does **not** help when the action fills the whole clip (Charades: the oracle itself is no
better than uniform); the sampler detects that from the scan and falls back to uniform frames, and
`sampler.last_fallback` tells you when it did. Details: [`results/thumos14_learned.md`](results/thumos14_learned.md),
[`results/charades.md`](results/charades.md).

Where did the time go?

```python
with vam.profile("keyframe") as p:
    out = model(video, sampler=vam.KeyframeSampler(8))
print(p.report())                                     # wall ms and peak GPU MB per stage
```

## 3. Give a budget in milliseconds instead of frames

```python
sampler = vam.BudgetedSampler(150, model, calibrate_on=["a.mp4", "b.mp4", "c.mp4"])
out = model(video, sampler=sampler)
print(sampler.last_plan)      # Plan(config=Config(frames=8, refine=True, stride=6), predicted_ms=131.2, stage='post', ...)
```

Calibration runs the full pipeline on the three clips under four configurations (about 6 s on a GPU, under a
minute on CPU), fits a linear cost model and caches it per machine in `~/.cache/vam/`. For each video the
scheduler picks the most accurate configuration whose predicted cost fits, and re-plans once after the keyframe
scan when it knows the real keyframe count. On the 5090 it chooses 4 frames at 100 ms, 8 at 150 and 16 with
refinement from 200 ms up: 0.855 / 0.960 / 0.945 top-1, where uniform sampling is stuck at 0.720. On a CPU-only
host every choice differs. Budgets are honoured on the median; on a shared machine the 90th percentile sits
50 to 60 ms above it, whatever the policy ([`results/thumos14_scheduler.md`](results/thumos14_scheduler.md)).

## 4. Stream

```python
for batch in vam.Stream("sample.mp4", window=16, hop=8):   # overlapping 16-frame windows; a camera index works too
    print(batch.timestamps[0].item(), model(batch).top_k(1)[0])
```

## 5. Run a policy

A policy is a `VideoActionModel` that fills `actions` instead of `logits`. Everything above applies unchanged.

```python
policy = vam.get_model("fake-policy")                      # deterministic stand-in, 7-D actions, no weights
for batch in vam.Stream("sample.mp4", window=1, hop=1):    # act on every frame
    a = policy(batch).actions                              # (1, 7)
```

OpenVLA-7B (`vam.get_model("openvla-7b", unnorm_key="bridge_orig")`) is the real one: one camera frame plus an
instruction to a 7-D end-effector action, 113 ms a step on a 5090, 15 GB of weights. Its remote code is written
for transformers 4.40, so run it from an environment with `transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10
"numpy<2"`; everything else in `vam` is version-agnostic. What it does on robot data and on the wrong domain:
[`results/model_coverage.md`](results/model_coverage.md).

## 6. Errors you can catch

```python
from vam import VamError, VideoOpenError, DecodeError, ScanError
```

Missing or unreadable files raise `VideoOpenError` (an `IOError`); a source that opens but decodes nothing (a
codec your OpenCV build lacks, a truncated file) raises `DecodeError`; a keyframe scan that fails or hangs past
120 s raises `ScanError`. All subclass `VamError`.

## Next

- [`api.md`](api.md): every public name with its signature.
- [`roadmap.md`](roadmap.md): what is in v0.1 and what is deferred (hardware decode, feature reuse).
- [`experiments/README.md`](experiments/README.md): the sixteen experiments, including the negative one.
