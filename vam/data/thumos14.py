"""THUMOS14 untrimmed videos with temporal action annotations.

Why this dataset: videos are minutes long and the annotated action usually covers a
small fraction of the timeline. That is the regime where *which frames* a model sees
should matter, unlike trimmed UCF101 clips. Its 20 classes are a subset of UCF101,
so a head trained on UCF101 transfers directly.

Expected layout:

    root/
      annotations/<Class>_val.txt      # "video_validation_0000051 72.8 76.4" (start/end seconds)
      videos/video_validation_0000051.mp4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from vam.core import Video, load_video


@dataclass(frozen=True)
class Segment:
    class_name: str
    start: float  # seconds
    end: float


@dataclass
class UntrimmedVideo:
    path: str
    video_id: str
    segments: list[Segment] = field(default_factory=list)

    @property
    def classes(self) -> set[str]:
        return {s.class_name for s in self.segments}

    def action_fraction(self, duration: float) -> float:
        """Fraction of the timeline covered by any annotated action."""
        if duration <= 0 or not self.segments:
            return 0.0
        ivs = sorted((s.start, s.end) for s in self.segments)
        covered, cur_s, cur_e = 0.0, ivs[0][0], ivs[0][1]
        for s, e in ivs[1:]:
            if s > cur_e:
                covered += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        covered += cur_e - cur_s
        return min(1.0, covered / duration)

    def frame_in_action(self, t: float) -> bool:
        return any(s.start <= t <= s.end for s in self.segments)

    def video(self) -> Video:
        return load_video(self.path)


class THUMOS14:
    def __init__(self, root: str | Path, split: str = "val", require_video: bool = True):
        self.root = Path(root)
        ann_dir = self.root / "annotations"
        vid_dir = self.root / "videos"
        if not ann_dir.is_dir():
            raise FileNotFoundError(f"no annotations dir under {self.root}")
        suffix = f"_{split}.txt"
        by_id: dict[str, UntrimmedVideo] = {}
        self.classes: list[str] = []
        for f in sorted(ann_dir.glob(f"*{suffix}")):
            cls = f.name[: -len(suffix)]
            if cls == "Ambiguous":
                continue
            self.classes.append(cls)
            for line in f.read_text().split("\n"):
                parts = line.split()
                if len(parts) < 3:
                    continue
                vid, s, e = parts[0], float(parts[1]), float(parts[2])
                uv = by_id.setdefault(vid, UntrimmedVideo(str(vid_dir / f"{vid}.mp4"), vid))
                uv.segments.append(Segment(cls, s, e))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        vids = sorted(by_id.values(), key=lambda v: v.video_id)
        if require_video:
            vids = [v for v in vids if Path(v.path).exists()]
        self.videos: list[UntrimmedVideo] = vids
        self.split = split

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, i: int) -> UntrimmedVideo:
        return self.videos[i]

    def __iter__(self) -> Iterator[UntrimmedVideo]:
        return iter(self.videos)

    def video_ids(self) -> list[str]:
        return [v.video_id for v in self.videos]

    def __repr__(self) -> str:
        return f"THUMOS14(split={self.split!r}, videos={len(self)}, classes={len(self.classes)})"
