"""Prior adaptive frame selectors run as released (experiment 018).

OCSampler (Lin et al., CVPR 2022, github.com/MCG-NJU/OCSampler): a MobileNetV2-TSM policy looks at 10
uniformly spaced frames of a video at 128 px and keeps the top-K; the classifier is FrameExit's frozen
ResNet-50 + MLP with the 10-frame head (the released checkpoints carry FrameExit's weights verbatim).
Only the policy lives here; pair it with ``FrameExitModel(fixed_head=10)``.

Released checkpoints (state_dict keys ``sampler.*``): ``anet_10to6``, ``anet_10to5`` (shift_div 10),
``mini_kinetics_10to4``, ``mini_kinetics_10to5`` (shift_div 8).
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vam.core import FrameBatch, Video
from vam.sampling.base import Sampler

_ARCH = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]


def temporal_shift(x: torch.Tensor, num_segments: int, shift_div: int) -> torch.Tensor:
    """TSM shift (Lin et al., 2019): first fold of channels one step back in time, second fold one step forward."""
    n, c, h, w = x.shape
    x = x.view(-1, num_segments, c, h * w)
    fold = c // shift_div
    out = torch.zeros_like(x)
    out[:, :-1, :fold] = x[:, 1:, :fold]
    out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
    return out.view(n, c, h, w)


class _ConvBN(nn.Module):
    """mmcv ConvModule layout (conv, bn, act) so checkpoint keys map one-to-one."""

    def __init__(self, cin: int, cout: int, k: int, stride: int = 1, pad: int = 0, groups: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, pad, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU6(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _ShiftConvBN(nn.Module):
    """TemporalShift(net) as mmaction wraps it: shift the input, then the conv block (keys ``.net.*``)."""

    def __init__(self, net: _ConvBN, num_segments: int, shift_div: int):
        super().__init__()
        self.net = net
        self.num_segments, self.shift_div = num_segments, shift_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(temporal_shift(x, self.num_segments, self.shift_div))


class _InvertedResidual(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, expand: int, num_segments: int, shift_div: int):
        super().__init__()
        hidden = int(round(cin * expand))
        self.use_res_connect = stride == 1 and cin == cout
        layers: list[nn.Module] = []
        if expand != 1:
            first = _ConvBN(cin, hidden, 1)
            # mmaction shifts the expand conv of every residual block with three convs
            layers.append(_ShiftConvBN(first, num_segments, shift_div) if self.use_res_connect else first)
        layers.append(_ConvBN(hidden, hidden, 3, stride, 1, groups=hidden))
        layers.append(_ConvBN(hidden, cout, 1, act=False))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


class MobileNetV2TSMPolicy(nn.Module):
    """OCSampler's sampler: MobileNetV2 with temporal shift, average pool, one logit per frame, softmax over frames."""

    def __init__(self, num_segments: int = 10, shift_div: int = 10):
        super().__init__()
        self.num_segments = num_segments
        self.conv1 = _ConvBN(3, 32, 3, 2, 1)
        cin = 32
        for i, (expand, ch, n, stride) in enumerate(_ARCH):
            blocks = []
            for j in range(n):
                blocks.append(_InvertedResidual(cin, ch, stride if j == 0 else 1, expand, num_segments, shift_div))
                cin = ch
            self.add_module(f"layer{i + 1}", nn.Sequential(*blocks))
        self.conv2 = _ConvBN(cin, 1280, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.logit = nn.Linear(1280, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (num_segments, 3, H, W) one video -> (num_segments,) probabilities."""
        x = self.conv1(x)
        for i in range(1, 8):
            x = getattr(self, f"layer{i}")(x)
        x = self.avg_pool(self.conv2(x)).flatten(1)
        return F.softmax(self.logit(x).squeeze(-1).reshape(-1, self.num_segments), dim=1)[0]


def uniform_sample_frames(num_frames: int, clip_len: int = 10, seed: int = 255) -> list[int]:
    """mmaction ``UniformSampleFrames(test_mode=True, num_clips=1)``: one frame per equal bin, offset drawn
    with the fixed seed, so the positions are deterministic for a given frame count."""
    rng = np.random.RandomState(seed)
    if num_frames < clip_len:
        inds = np.arange(clip_len)
    elif num_frames < 2 * clip_len:
        chosen = rng.choice(clip_len + 1, num_frames - clip_len, replace=False)
        offset = np.zeros(clip_len + 1, dtype=np.int64)
        offset[chosen] = 1
        inds = np.arange(clip_len) + np.cumsum(offset)[:-1]
    else:
        bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
        inds = bids[:clip_len] + rng.randint(np.diff(bids))
    return [int(i) for i in np.mod(inds, num_frames)]


class OCSampler(Sampler):
    """OCSampler as released: 10 candidate frames, policy at 128 px, top-``num_frames`` kept in temporal order.

    ``preprocessor`` must be the classifier's (FrameExit: short side 224, centre crop, ImageNet norm); the
    policy sees the same tensors resized to 128 px with nearest interpolation, as in the authors' code.
    Candidate frames are decoded once and handed to the classifier through :meth:`read`.
    """

    def __init__(self, num_frames: int = 6, checkpoint: str | os.PathLike | None = None, preprocessor=None,
                 device: str | None = None, dtype: torch.dtype = torch.float16, candidates: int = 10,
                 shift_div: int | None = None, resize_px: int = 128, seed: int = 255):
        super().__init__(num_frames)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dev == "cpu":
            dtype = torch.float32
        ck = checkpoint or os.environ.get("VAM_OCSAMPLER_CHECKPOINT")
        if shift_div is None:
            shift_div = 8 if (ck and "kinetics" in str(ck)) else 10
        self.policy = MobileNetV2TSMPolicy(candidates, shift_div)
        if ck:
            self.load_checkpoint(ck)
        self.policy.eval().to(dev, dtype=dtype)
        self._dev, self._dtype = torch.device(dev), dtype
        self.pre = preprocessor
        self.candidates, self.resize_px, self.seed = candidates, resize_px, seed
        self.last_probs: torch.Tensor | None = None
        self.last_candidates: list[int] | None = None
        self._cache: FrameBatch | None = None
        self.gmacs = self._count_gmacs()

    def load_checkpoint(self, path: str | os.PathLike) -> "OCSampler":
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        state = {k[len("sampler."):]: v for k, v in state.items() if k.startswith("sampler.")}
        missing, unexpected = self.policy.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"OCSampler checkpoint mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
        self.checkpoint = str(path)
        return self

    def _count_gmacs(self) -> float:
        """Multiply-accumulates of one policy call (10 frames at 128 px), in the GMACs convention of 017."""
        try:
            from torch.utils.flop_counter import FlopCounterMode
        except ImportError:  # pragma: no cover
            return 0.0
        x = torch.zeros(self.candidates, 3, self.resize_px, self.resize_px, device=self._dev, dtype=self._dtype)
        with torch.inference_mode(), FlopCounterMode(display=False) as fc:
            self.policy(x)
        return fc.get_total_flops() / 2 / 1e9

    @torch.inference_mode()
    def scores(self, batch: FrameBatch) -> torch.Tensor:
        if self.pre is None:
            raise ValueError("OCSampler needs the classifier's preprocessor (FrameExitModel.pre)")
        frames = batch.frames
        if batch.is_channels_first:
            frames = (frames * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        x = self.pre(frames).to(self._dev, self._dtype)  # (T,3,224,224)
        x = F.interpolate(x, size=self.resize_px)  # nearest, as in the released code
        return self.policy(x).float().cpu()

    def select(self, video: Video) -> list[int]:
        cand = uniform_sample_frames(video.num_frames, self.candidates, self.seed)
        order = sorted(range(len(cand)), key=lambda i: cand[i])
        batch = video.read([cand[i] for i in order])  # decoders want ascending indices
        inv = {cand[i]: p for p, i in enumerate(order)}
        batch = FrameBatch(batch.frames[[inv[c] for c in cand]], batch.timestamps[[inv[c] for c in cand]],
                           batch.indices[[inv[c] for c in cand]])
        probs = self.scores(batch)
        k = min(self.num_frames, len(cand))
        top = torch.topk(probs, k).indices.sort().values.tolist()
        self.last_probs, self.last_candidates = probs, cand
        self._cache = FrameBatch(batch.frames[top], batch.timestamps[top], batch.indices[top])
        return [cand[i] for i in top]

    def read(self, video: Video, indices: list[int]) -> FrameBatch:
        c = self._cache
        if c is not None and c.indices.tolist() == list(indices):
            return c
        return video.read(indices)

    def __repr__(self) -> str:
        return f"OCSampler(num_frames={self.num_frames}, candidates={self.candidates}, policy_gmacs={self.gmacs:.2f})"
