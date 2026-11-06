"""ActivityNet-1.3 (untrimmed, 200 classes, multi-label per video) and Mini-Kinetics (experiment 017).

ActivityNet layout (videos from the public mirror, annotations from `activity_net.v1-3.min.json`):

    root/
      activity_net.v1-3.min.json
      classes.txt                 # FrameExit's class order (antran89/ActivityNet Crawler/classes.txt)
      videos/v_<id>.<ext>

Mini-Kinetics layout (the AR-Net 200-class subset of Kinetics-400 val; lists from the OCSampler repo):

    root/
      mini_kinetics_val.txt       # "<youtube id> <duration> <class index>"
      minik_classInd.txt          # "<index> <class name>"
      minik_val/<youtube id>_<start>_<end>.mp4
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from vam.data.thumos14 import Segment, UntrimmedVideo

_VIDEO_EXT = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def _index_videos(vid_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if vid_dir.exists():
        for p in vid_dir.iterdir():
            if p.suffix.lower() in _VIDEO_EXT:
                out.setdefault(p.stem, p)
    return out


class ActivityNet:
    def __init__(self, root: str | Path, split: str = "validation", classes_file: str = "classes.txt",
                 annotations: str = "activity_net.v1-3.min.json", require_video: bool = True):
        self.root = Path(root)
        split = {"val": "validation", "train": "training"}.get(split, split)
        self.split = split
        db = json.load(open(self.root / annotations))["database"]
        self.classes = [ln.strip() for ln in open(self.root / classes_file) if ln.strip()]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        files = _index_videos(self.root / "videos")
        vids: list[UntrimmedVideo] = []
        self.missing: list[str] = []
        for key, v in sorted(db.items()):
            if v["subset"] != split:
                continue
            path = files.get(f"v_{key}")
            if path is None:
                self.missing.append(key)
                if require_video:
                    continue
            segs = [Segment(a["label"], float(a["segment"][0]), float(a["segment"][1])) for a in v["annotations"]]
            vids.append(UntrimmedVideo(str(path) if path else "", f"v_{key}", segs))
        self.videos = vids

    def labels(self, uv: UntrimmedVideo) -> list[int]:
        """All class indices annotated on the video (ActivityNet is multi-label; 6 of 10k have > 1)."""
        return sorted({self.class_to_idx[c] for c in uv.classes if c in self.class_to_idx})

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, i: int) -> UntrimmedVideo:
        return self.videos[i]

    def __iter__(self) -> Iterator[UntrimmedVideo]:
        return iter(self.videos)

    def __repr__(self) -> str:
        return f"ActivityNet({self.split}, {len(self.videos)} videos with files, {len(self.missing)} missing, {len(self.classes)} classes)"


class MiniKinetics:
    """Trimmed 10 s clips; each video carries one class, stored as a single whole-clip segment."""

    def __init__(self, root: str | Path, split: str = "val", require_video: bool = True):
        self.root = Path(root)
        self.split = split
        self.classes = [ln.split(maxsplit=1)[1].strip() for ln in open(self.root / "minik_classInd.txt") if ln.strip()]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        files = _index_videos(self.root / f"minik_{split}")
        by_yt = {stem[:11]: p for stem, p in files.items()}
        vids: list[UntrimmedVideo] = []
        self.missing: list[str] = []
        for ln in open(self.root / f"mini_kinetics_{split}.txt"):
            parts = ln.split()
            if len(parts) < 3:
                continue
            yt, label = parts[0], int(parts[2])
            path = by_yt.get(yt)
            if path is None:
                self.missing.append(yt)
                if require_video:
                    continue
            vids.append(UntrimmedVideo(str(path) if path else "", path.stem if path else yt,
                                       [Segment(self.classes[label], 0.0, 10.0)]))
        self.videos = vids

    def labels(self, uv: UntrimmedVideo) -> list[int]:
        return sorted({self.class_to_idx[c] for c in uv.classes})

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, i: int) -> UntrimmedVideo:
        return self.videos[i]

    def __iter__(self) -> Iterator[UntrimmedVideo]:
        return iter(self.videos)

    def __repr__(self) -> str:
        return f"MiniKinetics({self.split}, {len(self.videos)} videos with files, {len(self.missing)} missing)"
