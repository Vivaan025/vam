# 015 · Packaging and CI (roadmap R6)

**Date:** 2026-11-05 · **Status:** planned · **Results:** *(to fill)*

## Motivation
Experiments 001–014 make the runtime research-ready. A stranger still cannot `pip install` it: there is no wheel,
the dependency ranges are open-ended, the learned scorer lives in `results/cache/` on one server and needs the
dataset's class names to construct, and nothing runs the 70 tests on a machine that is not ours. This entry
turns the repository into an installable 0.1.0.

## Hypotheses (acceptance criteria, not research claims)
- **H1, a wheel installs clean.** `python -m build` produces a wheel; in a fresh venv on Windows and on the
  Linux server, `pip install vam_runtime-0.1.0-py3-none-any.whl` then `python examples/quickstart.py` runs end
  to end on CPU with no file from the repository present.
- **H2, the learned scorer ships.** A single release head (one head trained on all 200 THUMOS14 val videos, no
  fold map, ~2 MB) is package data, and `vam.KeyframeSampler(8)` with no arguments loads it plus CLIP ViT-B/32
  and needs no class names. Its scorer-level metrics on the 212 held-out test videos match the 5-fold ensemble
  within noise (Spearman, top-2 precision, chosen overlap).
- **H3, CI is green on both platforms.** GitHub Actions runs ruff and `pytest -m "not slow"` on ubuntu and
  windows with CPU torch; both pass. The `slow` tests (torchvision weights download) stay opt-in.

## Plan
1. Name: `vam` is taken on PyPI (HTTP 200 on the project page). Distribution name `vam-runtime`, import name
   stays `vam`. (One line in `pyproject.toml`; Vivaan can rename before upload.)
2. `pyproject.toml`: bounded ranges (`torch>=2.2,<3`, `numpy>=1.24,<3`, `opencv-python>=4.8,<5`, `av>=12,<17`),
   extras `hf` (transformers 4.40–5.x, pillow, needed for the CLIP scorer and HF models), `torchvision`,
   `policy` (timm, pyarrow), `dev` (pytest, ruff, build); classifiers, URLs, `[tool.setuptools.package-data]`
   for `vam/weights/*.pt`, registered `slow` marker, ruff config.
3. Release scorer: `scripts/train_scorer.py --release` trains one head on every video for the mean best epoch of
   the 5-fold run and writes `vam/weights/scorer_context_nozs.pt` with `folds = {}`. `ClipScorer` accepts an
   empty class list (embeddings only). `KeyframeSampler(num_frames, scorer=None)` builds `LearnedScorer` from the
   packaged head.
4. ruff: fix the 12 findings; `ruff check` is part of CI.
5. `.github/workflows/ci.yml`: matrix {ubuntu-latest, windows-latest} × {3.11, 3.12}, CPU torch from the PyTorch
   index, `pip install -e .[hf,dev]`, ruff, pytest with the `slow` marker deselected; a second job builds the
   wheel and installs it into a clean venv, then imports `vam` and runs the quickstart on `sample.mp4`.
6. Measure H2 on the server (`benchmarks/eval_thumos14.py --split test --samplers kf-learned` with the release
   head vs the fold checkpoint) and H1 on both machines.

## Metrics
H1: exit code 0 of the quickstart from a wheel-only venv on both OSes. H2: held-out test accuracy with the release
head within 1 point of the fold ensemble at 4 / 8 / 16 frames. H3: two green CI runs.

## Risks
- The CLIP download (600 MB) in CI: the quickstart job uses `UniformSampler`, so only VideoMAE (350 MB) is
  pulled; the unit tests pull a 1 KB image-processor config. If CI time is a problem, cache the HF hub directory.
- OpenVLA is not tested in CI (15 GB, transformers 4.40 pin); the `policy` extra installs its deps only.
- Windows: the process scan backend is skipped (no fork); the thread backend is what the tests cover there.

## Results
*(to fill)*

## Decision
*(to fill)*
