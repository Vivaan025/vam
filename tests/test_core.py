import torch

import vam
from vam.core import ActionOutput


def test_load_video_metadata(synthetic_video):
    path, n, fps = synthetic_video
    v = vam.load_video(path)
    assert v.num_frames == n
    assert abs(v.fps - fps) < 1e-3
    assert v.width == 128 and v.height == 96
    v.close()


def test_read_frames_shape_and_timestamps(synthetic_video):
    path, _, fps = synthetic_video
    v = vam.load_video(path)
    b = v.read([0, 5, 10])
    assert b.frames.shape == (3, 96, 128, 3)
    assert b.frames.dtype == torch.uint8
    assert b.indices.tolist() == [0, 5, 10]
    assert torch.allclose(b.timestamps, torch.tensor([0, 5, 10]) / fps)
    cf = b.channels_first()
    assert cf.frames.shape == (3, 3, 96, 128) and cf.frames.max() <= 1.0
    v.close()


def test_action_output_top_k():
    out = ActionOutput(logits=torch.tensor([0.1, 2.0, 0.5]), labels=["a", "b", "c"])
    top = out.top_k(2)
    assert top[0][0] == "b" and top[1][0] == "c"
    assert 0 < top[0][1] <= 1


def test_stream_windows(synthetic_video):
    path, n, _ = synthetic_video
    windows = list(vam.Stream(path, window=16, hop=8))
    assert len(windows[0]) == 16
    assert windows[1].indices[0].item() == 8
    assert sum(1 for w in windows) >= n // 8
