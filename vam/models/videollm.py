"""Video-LLM adapter (experiment 020): a chat model that takes the selected frames and a question.

`VideoLLM` wraps any Hugging Face image-text-to-text model whose processor accepts a video as an array of
frames (Qwen3-VL, Qwen2.5-VL). The selected frames' timestamps are passed as video metadata so the model's
prompt carries the real times of irregularly spaced frames. Output is text (`ActionOutput.text`); for
multiple-choice questions `answer_letter` extracts the option letter.

    llm = VideoLLM("Qwen/Qwen3-VL-8B-Instruct")
    batch = KeyframeSampler(16)(video)
    out = llm.answer(batch, "What is the person holding?", ["A. a cup", "B. a phone", "C. a book", "D. a key"], duration=video.duration)
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import torch

from vam.core import ActionOutput, FrameBatch
from vam.models.base import VideoActionModel
from vam.models.registry import register

MCQ_PROMPT = ("Select the best answer to the following multiple-choice question based on the video. "
              "Respond with only the letter (A, B, C, or D) of the correct option.\n{question}\n{options}\nThe best answer is:")


def answer_letter(text: str, letters: str = "ABCD") -> Optional[str]:
    """First option letter in the model's reply ("B", "B.", "The answer is (B)"), or None."""
    m = re.search(rf"\b([{letters}])\b", text.strip().upper())
    return m.group(1) if m else None


def _resize_longest(frames: np.ndarray, max_edge: int) -> np.ndarray:
    """uint8 (T,H,W,3) -> longest side <= max_edge, both sides multiples of 32 (one visual token)."""
    import cv2

    t, h, w, _ = frames.shape
    scale = min(1.0, max_edge / max(h, w))
    nh, nw = max(32, int(round(h * scale / 32)) * 32), max(32, int(round(w * scale / 32)) * 32)
    if (nh, nw) == (h, w):
        return frames
    return np.stack([cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA) for f in frames])


class VideoLLM(VideoActionModel):
    def __init__(self, name: str = "Qwen/Qwen3-VL-8B-Instruct", device: str | None = None,
                 dtype: torch.dtype = torch.bfloat16, max_edge: int = 448, max_new_tokens: int = 8,
                 attn_implementation: str | None = None):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dev == "cpu":
            dtype = torch.float32
        self.name, self._dev, self._dtype = name, torch.device(dev), dtype
        self.max_edge, self.max_new_tokens = max_edge, max_new_tokens
        self.processor = AutoProcessor.from_pretrained(name)
        kw = {"dtype": dtype}
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        self.model = AutoModelForImageTextToText.from_pretrained(name, **kw).to(self._dev).eval()
        self.labels = None
        self.last_text: str | None = None
        self.last_tokens: int = 0

    # --- frames + prompt -> text --------------------------------------------------------------------------
    def _frames_u8(self, batch: FrameBatch) -> np.ndarray:
        f = batch.frames
        if batch.is_channels_first:
            f = (f * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        return _resize_longest(f.cpu().numpy(), self.max_edge)

    def _metadata(self, batch: FrameBatch, fps: float | None, num_frames: int | None, duration: float | None):
        """The selected frames' real positions, so the processor's timestamps are right for sparse frames."""
        from transformers.video_utils import VideoMetadata

        idx = [int(i) for i in batch.indices.tolist()]
        ts = batch.timestamps.tolist()
        if fps is None:
            fps = (idx[-1] - idx[0]) / max(ts[-1] - ts[0], 1e-3) if len(idx) > 1 and ts[-1] > ts[0] else 1.0
        total = int(num_frames) if num_frames else (idx[-1] + 1)
        return VideoMetadata(total_num_frames=total, fps=float(fps), duration=float(duration) if duration else total / float(fps),
                             frames_indices=idx, video_backend="vam")

    @torch.inference_mode()
    def generate(self, batch: FrameBatch, prompt: str, fps: float | None = None, num_frames: int | None = None,
                 duration: float | None = None) -> str:
        frames = self._frames_u8(batch)
        meta = self._metadata(batch, fps, num_frames, duration)
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], videos=[frames], video_metadata=[meta], do_sample_frames=False,
                                return_tensors="pt").to(self._dev)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new = out[0, inputs["input_ids"].shape[1]:]
        self.last_tokens = int(inputs["input_ids"].shape[1])
        self.last_text = self.processor.decode(new, skip_special_tokens=True)
        return self.last_text

    def answer(self, batch: FrameBatch, question: str, options: list[str], **video_kw) -> str:
        """Multiple choice: returns the option letter (or the raw reply if none is found)."""
        reply = self.generate(batch, MCQ_PROMPT.format(question=question, options="\n".join(options)), **video_kw)
        return answer_letter(reply) or reply

    def forward_frames(self, batch: FrameBatch, prompt: str = "Describe the video in one sentence.", **video_kw) -> ActionOutput:
        return ActionOutput(text=self.generate(batch, prompt, **video_kw))

    @property
    def feature_dim(self) -> int:
        return 0


@register("qwen3-vl-8b")
def _qwen3_vl_8b(**kw):
    return VideoLLM("Qwen/Qwen3-VL-8B-Instruct", **kw)


@register("qwen3-vl-4b")
def _qwen3_vl_4b(**kw):
    return VideoLLM("Qwen/Qwen3-VL-4B-Instruct", **kw)
