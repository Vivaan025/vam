"""Experiment 014: a second recognition family and policies through the same contract."""

import pytest
import torch

import vam
from vam.core import FrameBatch
from vam.models import get_model, list_models
from vam.models.policy import ActionPolicy, FakePolicy


def _batch(t=16, h=64, w=96):
    g = torch.Generator().manual_seed(1)
    frames = torch.randint(0, 255, (t, h, w, 3), generator=g, dtype=torch.uint8)
    return FrameBatch(frames, torch.arange(t).float() / 10, torch.arange(t))


def test_registry_lists_new_families():
    names = list_models()
    assert "fake-policy" in names and "openvla-7b" in names
    assert "r2plus1d-18-k400" in names and "mvit-v2-s-k400" in names


def test_fake_policy_fills_actions_not_logits():
    pol = FakePolicy(instruction="pick up the cup")
    out = pol(_batch(4))
    assert out.logits is None and out.actions.shape == (4, 7)
    assert torch.allclose(out.actions, pol(_batch(4)).actions)  # deterministic
    with pytest.raises(ValueError):
        out.top_k()


def test_policy_through_stream_and_sampler(h264_video):
    path, fps, n = h264_video
    pol = get_model("fake-policy", action_dim=3)
    steps = [pol(b) for b in vam.Stream(path, window=1, hop=25)]
    assert len(steps) == n // 25 and all(s.actions.shape == (1, 3) for s in steps)
    v = vam.load_video(path)
    out = pol(v, sampler=vam.UniformSampler(5))  # a policy is a VideoActionModel: samplers apply
    assert out.actions.shape == (5, 3)
    v.close()


def test_policy_instruction_override_and_state():
    class Echo(ActionPolicy):
        def act(self, frames, instruction, state):
            return torch.full((frames.shape[0], 2), float(len(instruction)) + (state.sum() if state is not None else 0))

    pol = Echo(instruction="ab")
    assert pol.forward_frames(_batch(2)).actions[0, 0] == 2
    assert pol.forward_frames(_batch(2), instruction="abcd", state=torch.tensor([1.0, 1.0])).actions[0, 0] == 6


@pytest.mark.slow
def test_torchvision_adapter_runs_and_exposes_features():
    pytest.importorskip("torchvision")
    try:
        m = get_model("r2plus1d-18-k400", device="cpu")
    except Exception as e:  # no network for the 120 MB weights
        pytest.skip(f"weights unavailable: {e}")
    out = m(_batch(16))
    assert out.logits.shape == (400,) and len(out.labels) == 400 and out.top_k(1)
    f = m.features(_batch(16))
    assert f.shape == (m.feature_dim,) and m.feature_dim == 512
    from vam.models.probe import LinearProbe

    probe = LinearProbe(m, 5, list("abcde"))
    assert probe(_batch(8)).logits.shape == (5,)
