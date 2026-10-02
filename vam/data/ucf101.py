"""UCF101 action recognition dataset (13,320 clips, 101 classes, 3 official splits).

Expected layout (what the official archives extract to):

    root/
      UCF-101/<ClassName>/v_<ClassName>_gXX_cYY.avi
      ucfTrainTestlist/trainlist0{1,2,3}.txt  testlist0{1,2,3}.txt  classInd.txt

The dataset yields Clip records (path + label); decoding is left to vam.Video so
every sampler and runtime component sees identical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from vam.core import Video, load_video


@dataclass(frozen=True)
class Clip:
    path: str
    label: int
    class_name: str

    def video(self) -> Video:
        return load_video(self.path)


class UCF101:
    def __init__(self, root: str | Path, split: str = "train", fold: int = 1, limit: int | None = None,
                 seed: int = 0):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        if fold not in (1, 2, 3):
            raise ValueError("fold must be 1, 2 or 3")
        self.root = Path(root)
        self.videos_dir = self.root / "UCF-101"
        lists = self.root / "ucfTrainTestlist"
        if not self.videos_dir.is_dir() or not lists.is_dir():
            raise FileNotFoundError(f"UCF101 not found under {self.root} (need UCF-101/ and ucfTrainTestlist/)")

        self.classes: list[str] = []
        for line in (lists / "classInd.txt").read_text().split("\n"):
            parts = line.split()
            if len(parts) == 2:
                self.classes.append(parts[1])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        list_file = lists / f"{split}list0{fold}.txt"
        rel_paths = [ln.split()[0] for ln in list_file.read_text().split("\n") if ln.strip()]
        clips = []
        for rel in rel_paths:
            cls = rel.split("/")[0]
            clips.append(Clip(str(self.videos_dir / rel), self.class_to_idx[cls], cls))

        if limit is not None and limit < len(clips):
            import random

            rng = random.Random(seed)
            clips = rng.sample(clips, limit)
            clips.sort(key=lambda c: c.path)
        self.clips: list[Clip] = clips
        self.split, self.fold = split, fold

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int) -> Clip:
        return self.clips[i]

    def __iter__(self) -> Iterator[Clip]:
        return iter(self.clips)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def labels(self) -> Sequence[int]:
        return [c.label for c in self.clips]

    def __repr__(self) -> str:
        return f"UCF101(split={self.split!r}, fold={self.fold}, clips={len(self)}, classes={self.num_classes})"
