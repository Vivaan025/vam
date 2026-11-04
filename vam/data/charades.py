"""Charades: ~30 s indoor videos with per-instance temporal annotations over 157 classes (experiment 012).

Layout (from scripts/download_charades.py):
    <root>/Charades_v1_480/<id>.mp4
    <root>/Charades/Charades_v1_classes.txt        # "c000 Holding some clothes"
    <root>/Charades/Charades_v1_{train,test}.csv   # id,...,actions,length ; actions = "c092 11.90 21.20;c147 0.00 12.60"

Exposes the THUMOS14 interface (videos / classes / class_to_idx / segments per video) so the benchmarks take
either dataset.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from vam.data.thumos14 import Segment, UntrimmedVideo


class Charades:
    def __init__(self, root: str | Path, split: str = "test", require_video: bool = True,
                 min_segments: int = 1):
        self.root = Path(root)
        ann_dir = self.root / "Charades"
        vid_dir = self.root / "Charades_v1_480"
        classes_txt = ann_dir / "Charades_v1_classes.txt"
        if not classes_txt.exists():
            raise FileNotFoundError(f"no {classes_txt}; run scripts/download_charades.py")
        self.code_to_name: dict[str, str] = {}
        for line in classes_txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                code, _, name = line.partition(" ")
                self.code_to_name[code] = name.strip()
        self.classes: list[str] = [self.code_to_name[c] for c in sorted(self.code_to_name)]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.split = split
        self.lengths: dict[str, float] = {}
        vids: list[UntrimmedVideo] = []
        with (ann_dir / f"Charades_v1_{split}.csv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                vid = row["id"]
                uv = UntrimmedVideo(str(vid_dir / f"{vid}.mp4"), vid)
                for item in (row.get("actions") or "").split(";"):
                    parts = item.split()
                    if len(parts) != 3:
                        continue
                    code, s, e = parts
                    if code in self.code_to_name and float(e) > float(s):
                        uv.segments.append(Segment(self.code_to_name[code], float(s), float(e)))
                if len(uv.segments) < min_segments:
                    continue
                try:
                    self.lengths[vid] = float(row.get("length") or 0.0)
                except ValueError:
                    pass
                vids.append(uv)
        if require_video:
            vids = [v for v in vids if Path(v.path).exists()]
        self.videos = sorted(vids, key=lambda v: v.video_id)

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, i: int) -> UntrimmedVideo:
        return self.videos[i]

    def __iter__(self) -> Iterator[UntrimmedVideo]:
        return iter(self.videos)

    def video_ids(self) -> list[str]:
        return [v.video_id for v in self.videos]

    def __repr__(self) -> str:
        return f"Charades(split={self.split!r}, videos={len(self)}, classes={len(self.classes)})"
