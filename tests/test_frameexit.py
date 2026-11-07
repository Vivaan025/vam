"""Experiment 017: the FrameExit classifier adapter and the ActivityNet / Mini-Kinetics dataset classes."""

import json

import pytest
import torch

from vam.core import FrameBatch
from vam.data import ActivityNet, MiniKinetics
from vam.models import list_models


def _batch(t, h=64, w=96, seed=1):
    g = torch.Generator().manual_seed(seed)
    frames = torch.randint(0, 255, (t, h, w, 3), generator=g, dtype=torch.uint8)
    return FrameBatch(frames, torch.arange(t).float(), torch.arange(t))


@pytest.fixture(scope="module")
def fe():
    pytest.importorskip("torchvision")
    from vam.models.frameexit import FrameExitModel

    torch.manual_seed(0)
    return FrameExitModel("activitynet", device="cpu", imagenet_init=False)


def test_registry_lists_frameexit():
    pytest.importorskip("torchvision")
    assert "frameexit-activitynet" in list_models() and "frameexit-minikinetics" in list_models()


def test_fixed_mode_uses_head_for_frame_count(fe):
    out4 = fe(_batch(4))
    out10 = fe(_batch(10))
    assert out4.logits.shape == (200,) and out10.logits.shape == (200,)
    assert out4.features.shape == (4, 4096)
    # same pooled features through a different head give different logits
    feats = fe.frame_features(_batch(4))
    a = fe.logits_from_features(feats)
    fe_heads = fe.classifiers
    b = fe_heads[9](feats.max(0).values.to(fe_heads[9].weight.dtype)).float()
    assert not torch.allclose(a, b)
    # more than 10 frames falls back to the 10-frame head
    assert fe(_batch(12)).logits.shape == (200,)


def test_exit_forward_returns_logits_and_frames_used(fe):
    logits, used = fe.exit_forward(_batch(10))
    assert logits.dtype == torch.float32
    assert logits.shape == (200,) and 1 <= used <= 10
    # forced order on a shorter batch
    logits, used = fe.exit_forward(_batch(3))
    assert 1 <= used <= 3


def test_checkpoint_key_mapping_roundtrip(fe, tmp_path):
    # FrameExit saves DataParallel keys ("module.") and mlp.layers.N; the loader maps both
    state = {"module." + k.replace("mlp.", "mlp.layers."): v for k, v in fe.state_dict().items()}
    p = tmp_path / "ck.pth"
    torch.save(state, p)
    from vam.models.frameexit import FrameExitModel

    m = FrameExitModel("activitynet", checkpoint=p, device="cpu")
    for (k1, v1), (k2, v2) in zip(fe.state_dict().items(), m.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)


def test_activitynet_and_minik_datasets(tmp_path):
    root = tmp_path / "anet"
    (root / "videos").mkdir(parents=True)
    db = {"abc": {"subset": "validation", "annotations": [{"label": "Archery", "segment": [1.0, 5.0]}]},
          "def": {"subset": "validation", "annotations": [{"label": "Zumba", "segment": [0.0, 2.0]}, {"label": "Archery", "segment": [3.0, 4.0]}]},
          "ghi": {"subset": "training", "annotations": [{"label": "Zumba", "segment": [0.0, 2.0]}]}}
    json.dump({"database": db}, open(root / "activity_net.v1-3.min.json", "w"))
    (root / "classes.txt").write_text("Archery\nZumba\n")
    (root / "videos" / "v_abc.mp4").write_bytes(b"")
    ds = ActivityNet(root)
    assert len(ds) == 1 and ds.missing == ["def"] and ds.labels(ds[0]) == [0]
    ds_all = ActivityNet(root, require_video=False)
    assert len(ds_all) == 2 and ds_all.labels(ds_all[1]) == [0, 1]
    assert len(ActivityNet(root, "train", require_video=False)) == 1

    mk = tmp_path / "minik"
    (mk / "minik_val").mkdir(parents=True)
    (mk / "minik_classInd.txt").write_text("abseiling\nzumba\n")
    (mk / "mini_kinetics_val.txt").write_text("7LSJ2Ryv1a8 299 0\nDcwyDVypfYg 299 1\n")
    (mk / "minik_val" / "7LSJ2Ryv1a8_000299_000309.mp4").write_bytes(b"")
    m = MiniKinetics(mk)
    assert len(m) == 1 and m.missing == ["DcwyDVypfYg"] and m.labels(m[0]) == [0]
    assert m[0].segments[0].class_name == "abseiling"
    (mk / "minik_classInd.txt").write_text("0 abseiling\n1 zumba\n")
    assert MiniKinetics(mk).classes == ["abseiling", "zumba"]
