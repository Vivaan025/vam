"""Video-MME (Fu et al., 2024): 900 YouTube videos x 3 multiple-choice questions, three duration buckets.

Layout (from `lmms-lab/Video-MME` on Hugging Face, experiment 020):

    root/
      raw/videomme/test-00000-of-00001.parquet   # 2,700 rows: video_id, duration, domain, sub_category, url,
                                                 #   videoID, question_id, task_type, question, options, answer
      videos/**/<videoID>.mp4                    # unzipped videos_chunked_*.zip
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

_VIDEO_EXT = (".mp4", ".mkv", ".webm", ".avi", ".mov")


@dataclass
class MMEQuestion:
    question_id: str
    task_type: str
    question: str
    options: list[str]
    answer: str


@dataclass
class MMEVideo:
    video_id: str
    youtube_id: str
    path: str
    duration: str  # "short" | "medium" | "long"
    domain: str
    sub_category: str
    questions: list[MMEQuestion] = field(default_factory=list)


class VideoMME:
    def __init__(self, root: str | Path, parquet: str | None = None, require_video: bool = True):
        self.root = Path(root)
        pq_path = Path(parquet) if parquet else next(iter(sorted((self.root / "raw").rglob("*.parquet"))), None)
        if pq_path is None:
            raise FileNotFoundError(f"no parquet under {self.root / 'raw'}")
        import pyarrow.parquet as pq

        rows = pq.read_table(pq_path).to_pylist()
        files: dict[str, Path] = {}
        vid_dir = self.root / "videos"
        if vid_dir.exists():
            for p in vid_dir.rglob("*"):
                if p.suffix.lower() in _VIDEO_EXT:
                    files.setdefault(p.stem, p)
        by_vid: dict[str, MMEVideo] = {}
        self.missing: list[str] = []
        for r in rows:
            yt = r["videoID"]
            if r["video_id"] not in by_vid:
                path = files.get(yt)
                if path is None:
                    self.missing.append(yt)
                    if require_video:
                        continue
                by_vid[r["video_id"]] = MMEVideo(r["video_id"], yt, str(path) if path else "", r["duration"],
                                                 r["domain"], r["sub_category"])
            if r["video_id"] in by_vid:
                by_vid[r["video_id"]].questions.append(
                    MMEQuestion(r["question_id"], r["task_type"], r["question"], list(r["options"]), r["answer"]))
        self.missing = sorted(set(self.missing))
        self.videos = [by_vid[k] for k in sorted(by_vid)]

    @property
    def questions(self) -> int:
        return sum(len(v.questions) for v in self.videos)

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, i: int) -> MMEVideo:
        return self.videos[i]

    def __iter__(self) -> Iterator[MMEVideo]:
        return iter(self.videos)

    def __repr__(self) -> str:
        return f"VideoMME({len(self.videos)} videos with files, {self.questions} questions, {len(self.missing)} missing)"
