"""Experiment 018: prior samplers run as released (OCSampler)."""

import numpy as np
import pytest
import torch

import vam
from vam.core import FrameBatch
from vam.sampling import OCSampler, uniform_sample_frames
from vam.models.arnet import arnet_val_indices
from vam.sampling.prior import MobileNetV2TSMPolicy, temporal_shift


def _mmaction_uniform(num_frames, clip_len=10, seed=255):
    """Verbatim copy of mmaction's UniformSampleFrames._get_test_clips (num_clips=1)."""
    np.random.seed(seed)
    if num_frames < clip_len:
        inds = np.arange(0, clip_len)
    elif clip_len <= num_frames < clip_len * 2:
        basic = np.arange(clip_len)
        inds = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
        offset = np.zeros(clip_len + 1, dtype=np.int64)
        offset[inds] = 1
        offset = np.cumsum(offset)
        inds = basic + offset[:-1]
    else:
        bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
        bsize = np.diff(bids)
        bst = bids[:clip_len]
        inds = bst + np.random.randint(bsize)
    return [int(i) for i in np.mod(inds, num_frames)]


def test_uniform_sample_frames_matches_mmaction():
    for n in (3, 10, 15, 19, 20, 57, 300, 4321):
        assert uniform_sample_frames(n) == _mmaction_uniform(n), n


def test_temporal_shift_moves_folds():
    x = torch.arange(2 * 10 * 10 * 4, dtype=torch.float32).reshape(20, 10, 2, 2)  # 2 videos x 10 segments
    y = temporal_shift(x, 10, 10)
    xv, yv = x.view(2, 10, 10, 4), y.view(2, 10, 10, 4)
    assert torch.equal(yv[:, :-1, 0], xv[:, 1:, 0]) and torch.all(yv[:, -1, 0] == 0)  # fold 0 one step back
    assert torch.equal(yv[:, 1:, 1], xv[:, :-1, 1]) and torch.all(yv[:, 0, 1] == 0)  # fold 1 one step forward
    assert torch.equal(yv[:, :, 2:], xv[:, :, 2:])


def test_policy_shapes_and_checkpoint_keys(tmp_path):
    torch.manual_seed(0)
    pol = MobileNetV2TSMPolicy(10, 10).eval()
    p = pol(torch.randn(10, 3, 128, 128))
    assert p.shape == (10,) and abs(float(p.sum()) - 1) < 1e-5
    keys = set(pol.state_dict())
    # mmaction layout: shifted expand convs carry ".net.", the rest is conv/bn under layerN.i.conv.j
    assert "layer2.1.conv.0.net.conv.weight" in keys and "layer2.0.conv.0.conv.weight" in keys
    assert "layer7.0.conv.2.bn.running_mean" in keys and "logit.weight" in keys
    state = {"sampler." + k: v for k, v in pol.state_dict().items()}
    state["backbone.model.0.weight"] = torch.zeros(1)  # classifier keys are ignored
    torch.save({"state_dict": state}, tmp_path / "oc.pth")
    s = OCSampler(6, checkpoint=tmp_path / "oc.pth", device="cpu")
    assert all(torch.equal(a, b) for a, b in zip(pol.state_dict().values(), s.policy.state_dict().values()))
    assert 0.5 < s.gmacs < 2.0  # MobileNetV2 at 128 px, 10 frames


def test_ocsampler_selects_from_candidates_and_caches(h264_video):
    pytest.importorskip("torchvision")
    from vam.models.frameexit import FrameExitModel

    torch.manual_seed(0)
    fe = FrameExitModel("activitynet", device="cpu", imagenet_init=False, fixed_head=10)
    s = OCSampler(4, preprocessor=fe.pre, device="cpu")
    path, fps, n = h264_video
    v = vam.load_video(path)
    try:
        idx = s.select(v)
        assert len(idx) == 4 and idx == sorted(idx) and set(idx) <= set(s.last_candidates)
        assert len(s.last_candidates) == 10 and s.last_probs.shape == (10,)
        batch = s.read(v, idx)
        assert isinstance(batch, FrameBatch) and batch.indices.tolist() == idx and batch is s._cache
        assert s.read(v, idx[:2]).indices.tolist() == idx[:2]  # other indices decode normally
        out = fe(batch)
        assert out.logits.shape == (200,)
    finally:
        v.close()
    # the fixed head is used whatever the frame count
    feats = fe.frame_features(batch)
    a = fe.logits_from_features(feats)
    fe.fixed_head = None
    assert not torch.allclose(a, fe.logits_from_features(feats))


def test_arnet_val_indices_match_reference():
    for n in (5, 16, 17, 100, 3153):
        if n > 16:
            tick = n / 16.0
            ref = [int(tick / 2.0 + tick * x) for x in range(16)]
        else:
            ref = list(range(n)) + [n - 1] * (16 - n)
        assert arnet_val_indices(n) == ref


@pytest.mark.parametrize("mode", ["scsampler", "ada"])
def test_arnet_modes_run_and_account(mode, tmp_path):
    pytest.importorskip("torchvision")
    from vam.models.arnet import ARNetModel

    torch.manual_seed(0)
    m = ARNetModel(mode, device="cpu")
    g = torch.Generator().manual_seed(1)
    batch = FrameBatch(torch.randint(0, 255, (16, 64, 96, 3), generator=g, dtype=torch.uint8),
                       torch.arange(16).float(), torch.arange(16))
    out = m(batch)
    assert out.logits.shape == (200,) and out.features.shape == (16, 1280)
    assert 1 <= m.last_frames_used <= 16 and m.last_gmacs > 0
    assert len(m.last_decisions) == (10 if mode == "scsampler" else 16)
    if mode == "scsampler":
        assert m.last_frames_used == 10 and m.last_decisions == sorted(m.last_decisions)
        assert abs(m.last_gmacs - (16 * m.gmacs_policy + 10 * m.gmacs_backbone[0])) < 1e-6
        assert 3.9 < m.gmacs_backbone[0] < 4.3  # ResNet-50 at 224
    else:
        assert m.gmacs_backbone[0] > m.gmacs_backbone[1] > m.gmacs_backbone[2] > m.gmacs_policy
    # checkpoint keys in the released layout (module., rnn., lite_backbone.features., base_model_list.N, new_fc_list.N)
    state = {"module." + k: v for k, v in m.state_dict().items()}
    torch.save({"state_dict": state, "epoch": 1}, tmp_path / "ck.pth.tar")
    m2 = ARNetModel(mode, checkpoint=tmp_path / "ck.pth.tar", device="cpu")
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))
