"""AR-Net (Meng et al., ECCV 2020, github.com/mengyuest/AR-Net) and its SCSampler re-implementation, run from
the released ActivityNet checkpoints (experiment 018).

Both read 16 frames at AR-Net's validation positions (``int(tick / 2 + tick * x)``, tick = n / 16), short side
256, centre crop 224, ImageNet norm; the lower resolutions are nearest-neighbour rescales of that crop, as
in ``ops/dataset.py``. The classifier is AR-Net's own TSN (torchvision ResNets with a fresh fc), not FrameExit's.

``mode="scsampler"`` (``exp1_res_scsampler.pth.tar``, test script: ``--reso_list 224 84 --policy_input_offset 0
--frame_independent --real_scsampler --top_k 10``): a MobileNetV2 scores every frame (max class logit of its
own 200-way head), the ResNet-50 runs on the top-10, logits averaged.

``mode="ada"`` (``exp0_res_ada.pth.tar``: ``--reso_list 224 168 112 84 --backbone_list resnet50 resnet34
resnet18 --skip_list 1 2 4 --policy_also_backbone --policy_input_offset 3``): a MobileNetV2 at 84 px feeds an
LSTM; per frame the policy picks one of {ResNet-50 @ 224, ResNet-34 @ 168, ResNet-18 @ 112, the policy's own
head, skip 1 / 2 / 4 frames}; the chosen backbones' logits are averaged. The released code samples the test
decision with Gumbel noise at tau = 1e-6; here the decision is the arg max of the policy logits (no noise), the
deterministic limit the authors describe.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vam.core import ActionOutput, FrameBatch
from vam.models.base import VideoActionModel
from vam.runtime.preprocess import Preprocessor

_RESO = {"resnet50": 224, "resnet34": 168, "resnet18": 112}


def arnet_val_indices(num_frames: int, num_segments: int = 16) -> list[int]:
    """AR-Net ``_get_val_indices`` (non-dense): one frame at the centre of each of 16 equal segments."""
    if num_frames > num_segments:
        tick = num_frames / float(num_segments)
        return [int(tick / 2.0 + tick * x) for x in range(num_segments)]
    return list(range(num_frames)) + [num_frames - 1] * (num_segments - num_frames)


def _rescale(x: torch.Tensor, px: int) -> torch.Tensor:
    return x if x.shape[-1] == px else F.interpolate(x, size=(px, px), mode="nearest")


def _gmacs(model: nn.Module, px: int, device, dtype) -> float:
    from torch.utils.flop_counter import FlopCounterMode

    with torch.inference_mode(), FlopCounterMode(display=False) as fc:
        model(torch.zeros(1, 3, px, px, device=device, dtype=dtype))
    return fc.get_total_flops() / 2 / 1e9


class ARNetModel(VideoActionModel):
    def __init__(self, mode: str = "ada", checkpoint: str | os.PathLike | None = None, device: str | None = None,
                 dtype: torch.dtype = torch.float16, num_classes: int = 200, num_segments: int = 16, top_k: int = 10,
                 policy_px: int | None = None, skip_list: tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        import torchvision

        if mode not in ("ada", "scsampler"):
            raise ValueError(mode)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dev == "cpu":
            dtype = torch.float32
        self._dev, self._dtype = torch.device(dev), dtype
        self.mode, self.num_segments, self.top_k, self.skip_list = mode, num_segments, top_k, skip_list
        self.backbones = ["resnet50"] if mode == "scsampler" else ["resnet50", "resnet34", "resnet18"]
        self.policy_px = policy_px or (224 if mode == "scsampler" else 84)

        mnv2 = torchvision.models.mobilenet_v2()
        mnv2.classifier = nn.Identity()  # AR-Net replaces it by Dropout (identity at eval): 1280-d output
        self.lite_backbone = mnv2
        hidden = 1280 if mode == "scsampler" else 512
        self.rnn = nn.LSTMCell(1280, 512) if mode == "ada" else None
        self.action_dim = len(self.backbones) + 1 + len(skip_list)  # resolutions + policy head + skips
        self.linear = nn.Linear(hidden, 1 if mode == "scsampler" else self.action_dim)
        self.lite_fc = nn.Linear(hidden, num_classes)
        self.base_model_list = nn.ModuleList()
        self.new_fc_list = nn.ModuleList()
        for name in self.backbones:
            m = getattr(torchvision.models, name)()
            self.new_fc_list.append(nn.Linear(m.fc.in_features, num_classes))
            m.fc = nn.Identity()
            self.base_model_list.append(m)
        self.labels: list[str] | None = None
        ck = checkpoint or os.environ.get(f"VAM_ARNET_{mode.upper()}_CHECKPOINT")
        if ck:
            self.load_checkpoint(ck)
        self.eval().to(self._dev, dtype=dtype)
        self.pre = Preprocessor(shortest_edge=256, crop=(224, 224), mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225), device=dev)
        # cost per frame of each option, measured in MACs under the same convention as GMACS_PER_FRAME
        self.gmacs_backbone = [_gmacs(b, _RESO[n], self._dev, dtype) for b, n in zip(self.base_model_list, self.backbones)]
        self.gmacs_policy = _gmacs(self.lite_backbone, self.policy_px, self._dev, dtype)
        self.last_frames_used = 0
        self.last_gmacs = 0.0
        self.last_decisions: list[int] | None = None

    def load_checkpoint(self, path: str | os.PathLike) -> "ARNetModel":
        import argparse

        # the .pth.tar files pickle argparse / numpy objects next to the weights; allow exactly those
        try:
            import numpy._core.multiarray as ma  # numpy >= 2
        except ImportError:  # pragma: no cover
            import numpy.core.multiarray as ma
        safe = [argparse.Namespace, np.dtype, np.ndarray, (ma.scalar, "numpy.core.multiarray.scalar"),
                (ma._reconstruct, "numpy.core.multiarray._reconstruct")]
        try:
            import numpy.dtypes as nd

            safe += [getattr(nd, n) for n in dir(nd) if n.endswith("DType")]
        except ImportError:  # pragma: no cover
            pass
        with torch.serialization.safe_globals(safe):
            state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        if self.rnn is None:
            state = {k: v for k, v in state.items() if not k.startswith("rnn.")}  # unused in the SCSampler path
        missing, unexpected = self.load_state_dict(state, strict=False)
        unexpected = [k for k in unexpected if not k.endswith("num_batches_tracked")]
        if missing or unexpected:
            raise RuntimeError(f"AR-Net checkpoint mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
        self.checkpoint = str(path)
        return self

    def _pixel_values(self, batch: FrameBatch) -> torch.Tensor:
        frames = batch.frames
        if batch.is_channels_first:
            frames = (frames * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        return self.pre(frames).to(self._dtype)  # (T,3,224,224)

    @torch.inference_mode()
    def forward_frames(self, batch: FrameBatch) -> ActionOutput:
        x = self._pixel_values(batch)
        t = x.shape[0]
        lite = self.lite_backbone(_rescale(x, self.policy_px))  # (T,1280)
        if self.mode == "scsampler":
            lite_logits = self.lite_fc(lite)  # (T,200)
            k = min(self.top_k, t)
            ind = torch.topk(lite_logits.max(dim=1).values, k).indices.sort().values
            logits = self.new_fc_list[0](self.base_model_list[0](x[ind])).float().mean(0)
            self.last_frames_used, self.last_decisions = k, ind.tolist()
            self.last_gmacs = self.gmacs_policy * t + self.gmacs_backbone[0] * k
            return ActionOutput(logits=logits.cpu(), labels=self.labels, features=lite.float().cpu())
        # adaptive: LSTM policy, one decision per frame, skips freeze the state
        hx = torch.zeros(1, 512, device=self._dev, dtype=self._dtype)
        cx = torch.zeros_like(hx)
        preds, decisions, remain, used, g = [], [], 0, 0, self.gmacs_policy * t
        n_res = len(self.backbones)
        for i in range(t):
            if remain > 0:  # skipping: state and decision carried, frame not processed
                remain -= 1
                decisions.append(-1)
                continue
            hx, cx = self.rnn(lite[i : i + 1], (hx, cx))
            a = int(torch.argmax(self.linear(hx), dim=1))
            decisions.append(a)
            if a < n_res:
                preds.append(self.new_fc_list[a](self.base_model_list[a](_rescale(x[i : i + 1], _RESO[self.backbones[a]]))).float()[0])
                used += 1
                g += self.gmacs_backbone[a]
            elif a == n_res:
                preds.append(self.lite_fc(hx).float()[0])
                used += 1
            else:
                remain = self.skip_list[a - n_res - 1] - 1
        logits = torch.stack(preds).mean(0) if preds else self.lite_fc(hx).float()[0]
        self.last_frames_used, self.last_decisions, self.last_gmacs = used, decisions, g
        return ActionOutput(logits=logits.cpu(), labels=self.labels, features=lite.float().cpu())

    @property
    def feature_dim(self) -> int:
        return 1280
