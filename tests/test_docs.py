"""016: the documentation is executable and complete.

- the quickstart's snippets that need no model weights run as written (the ones that download weights run in
  the CI wheel job via examples/quickstart.py);
- every name in each public module's `__all__` appears in docs/api.md.
"""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_quickstart_no_weights_snippets(tmp_path):
    import vam
    from examples.make_sample import make_sample

    clip = make_sample(str(tmp_path / "sample.mp4"), seconds=6)

    # 1 / 2: load, uniform frames, a model through the contract (fake policy: no weights)
    video = vam.load_video(clip)
    assert video.num_frames == 60 and abs(video.duration - 6.0) < 0.2
    policy = vam.get_model("fake-policy")
    out = policy(video, sampler=vam.UniformSampler(8))
    assert out.actions.shape == (8, 7) and out.logits is None
    with pytest.raises(ValueError):
        out.top_k(1)

    # profiler
    with vam.profile("uniform") as p:
        policy(video, sampler=vam.UniformSampler(4))
    assert "uniform" in p.report() and p.as_dict()["uniform"]["wall_ms"] >= 0

    # 4: stream, 5: act on every frame
    windows = list(vam.Stream(clip, window=16, hop=8))
    assert len(windows) >= 6 and windows[0].timestamps[0].item() == 0.0
    steps = [policy(b).actions for b in vam.Stream(clip, window=1, hop=1)]
    assert len(steps) == 60 and steps[0].shape == (1, 7)
    video.close()

    # 6: errors
    with pytest.raises(vam.VideoOpenError):
        vam.load_video(str(tmp_path / "missing.mp4"))
    assert issubclass(vam.VideoOpenError, IOError) and issubclass(vam.DecodeError, vam.VamError)


def test_quickstart_budget_snippet_with_an_oracle_scorer(tmp_path):
    """The budget example, with the oracle scorer in place of the packaged CLIP head (no download)."""
    import vam
    from examples.make_sample import make_sample
    from vam.sampling.keyframe import OracleKeyframeScorer

    clip = make_sample(str(tmp_path / "long.mp4"), seconds=30)
    scorer = OracleKeyframeScorer([(20.0, 30.0)], 2.0)
    policy = vam.get_model("fake-policy")
    sampler = vam.BudgetedSampler(5000, policy, calibrate_on=[clip], scorer=scorer, cache=False, repeats=1,
                                  window_s=2.0, decode_backend="serial")
    video = vam.load_video(clip)
    out = policy(video, sampler=sampler)
    assert out.actions.shape[0] == sampler.last_plan.config.frames
    video.close()


@pytest.mark.parametrize("module", ["vam", "vam.sampling", "vam.runtime", "vam.models", "vam.data", "vam.errors"])
def test_api_reference_covers_all(module):
    import importlib

    mod = importlib.import_module(module)
    names = getattr(mod, "__all__", None) or [n for n in dir(mod) if not n.startswith("_") and n[0].isupper()]
    text = (DOCS / "api.md").read_text(encoding="utf-8")
    missing = [n for n in names if not re.search(rf"`{re.escape(n)}[`(]", text)]
    assert not missing, f"{module}: not in docs/api.md: {missing}"
