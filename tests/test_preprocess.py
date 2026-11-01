import numpy as np
import pytest
import torch

from vam.runtime.preprocess import Preprocessor


def _frames(n=4, h=180, w=320, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    # smooth it a little so resampling differences are realistic
    import cv2
    base = cv2.GaussianBlur(base, (9, 9), 0)
    return np.stack([np.roll(base, i * 7, axis=1) for i in range(n)])


def test_shape_and_normalisation_manual():
    pre = Preprocessor(shortest_edge=224, crop=(224, 224), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    out = pre(_frames())
    assert out.shape == (4, 3, 224, 224)
    assert out.dtype == torch.float32
    assert -1.0 <= out.min() and out.max() <= 1.0


def test_single_frame_gets_batch_dim():
    pre = Preprocessor(224, (224, 224), (0, 0, 0), (1, 1, 1))
    assert pre(_frames(1)[0]).shape == (1, 3, 224, 224)


@pytest.mark.parametrize("name", ["MCG-NJU/videomae-base-finetuned-kinetics", "openai/clip-vit-base-patch32"])
def test_matches_hf_processor(name):
    transformers = pytest.importorskip("transformers")
    hf = transformers.AutoImageProcessor.from_pretrained(name)
    pre = Preprocessor.from_hf(hf)
    frames = _frames()
    ref = hf(list(frames), return_tensors="pt")["pixel_values"]
    ref = ref.reshape(-1, *ref.shape[-3:])  # video processors return (1,T,3,H,W)
    out = pre(frames)
    assert out.shape == ref.shape
    diff = (out - ref).abs()
    assert diff.mean().item() < 0.02, diff.mean().item()
    assert diff.max().item() < 0.5, diff.max().item()
