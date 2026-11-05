"""Policies: the second meaning of "action model" (experiment 014).

An `ActionPolicy` receives frames (what the agent sees) plus an instruction and optional proprioceptive
state, and returns `ActionOutput(actions=(T, action_dim))`. The runtime above it (samplers, Stream, cache,
scheduler cost model) is the same code that serves recognition models; this module only maps a policy's
conventions onto the contract.

`OpenVLAPolicy` wraps `openvla/openvla-7b` (Prismatic VLM, image + instruction -> 7 action tokens decoded to
continuous end-effector deltas with the dataset's un-normalisation statistics). `FakePolicy` is a
deterministic stand-in for tests and for calibrating the scheduler without 15 GB of weights.
"""

from __future__ import annotations

from typing import Optional

import torch

from vam.core import ActionOutput, FrameBatch
from vam.models.base import VideoActionModel
from vam.models.registry import register


class ActionPolicy(VideoActionModel):
    """Base class for policies. Subclasses implement `act(frames, instruction, state)`."""

    default_num_frames: int = 1
    action_dim: int = 7

    def __init__(self, instruction: str = "", state: Optional[torch.Tensor] = None):
        super().__init__()
        self.instruction = instruction
        self.state = state

    def act(self, frames: torch.Tensor, instruction: str, state: Optional[torch.Tensor]) -> torch.Tensor:
        """uint8 frames (T,H,W,3) -> actions (T, action_dim). One action per frame by default."""
        raise NotImplementedError

    def forward_frames(self, batch: FrameBatch, instruction: str | None = None,
                       state: Optional[torch.Tensor] = None) -> ActionOutput:
        frames = batch.frames
        if batch.is_channels_first:
            frames = (frames * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        actions = self.act(frames, instruction if instruction is not None else self.instruction,
                           state if state is not None else self.state)
        return ActionOutput(actions=actions.float().cpu())


class FakePolicy(ActionPolicy):
    """Deterministic: the action is a fixed linear function of the frame's mean colour. For tests and for
    calibrating the runtime on a policy-shaped model."""

    def __init__(self, action_dim: int = 7, **kw):
        super().__init__(**kw)
        self.action_dim = action_dim
        g = torch.Generator().manual_seed(0)
        self.w = torch.randn(3, action_dim, generator=g)

    def act(self, frames, instruction, state):
        mean_rgb = frames.float().mean(dim=(1, 2)) / 255.0  # (T,3)
        return mean_rgb @ self.w


class OpenVLAPolicy(ActionPolicy):
    """`openvla/openvla-7b`: one RGB frame + a language instruction -> a 7-D action
    (dx, dy, dz, droll, dpitch, dyaw, gripper) un-normalised with `unnorm_key`'s dataset statistics."""

    def __init__(self, name: str = "openvla/openvla-7b", unnorm_key: str = "bridge_orig",
                 device: str | None = None, dtype: torch.dtype = torch.bfloat16,
                 attn_implementation: str = "eager", **kw):
        super().__init__(**kw)
        from transformers import AutoModelForVision2Seq, AutoProcessor

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dev == "cpu":
            dtype = torch.float32
        self.processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            name, trust_remote_code=True, dtype=dtype, attn_implementation=attn_implementation,
            low_cpu_mem_usage=True).eval().to(dev)
        self._dev = torch.device(dev)
        self.unnorm_key = unnorm_key
        self.action_dim = 7
        self.name = name

    @torch.inference_mode()
    def act(self, frames, instruction, state):
        from PIL import Image

        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        out = []
        for f in frames:
            img = Image.fromarray(f.cpu().numpy())
            inputs = self.processor(prompt, img).to(self._dev, dtype=self.model.dtype)
            a = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
            out.append(torch.as_tensor(a, dtype=torch.float32).flatten()[: self.action_dim])
        return torch.stack(out)


@register("fake-policy")
def _fake(**kw):
    return FakePolicy(**kw)


@register("openvla-7b")
def _openvla(**kw):
    return OpenVLAPolicy(**kw)
