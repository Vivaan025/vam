"""Download only the THUMOS14 videos that carry temporal annotations (~220 val, ~212 test).

The official zips are 84+ GB because they include >1000 background videos. The
per-video URLs are public, so this fetches just the annotated ones in parallel.

    python scripts/download_thumos14.py --root data/thumos14 --split val --workers 6
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ANN = {
    "val": "https://www.crcv.ucf.edu/THUMOS14/Validation_set/TH14_Temporal_annotations_validation.zip",
    "test": "https://www.crcv.ucf.edu/THUMOS14/test_set/TH14_Temporal_Annotations_Test.zip",
}
VID = {
    "val": "https://www.crcv.ucf.edu/THUMOS14/Validation_set/videos/{vid}.mp4",
    "test": "https://www.crcv.ucf.edu/THUMOS14/test_set/videos/{vid}.mp4",
}


def fetch(url: str, dest: Path) -> None:
    """wget -c if available (resumable, tolerant of the site's certificate), else urllib."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("wget"):
        subprocess.run(["wget", "-q", "-c", "--no-check-certificate", "-O", str(dest), url], check=True)
    else:
        import ssl

        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ctx) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)


def ensure_annotations(root: Path, split: str) -> Path:
    ann_dir = root / "annotations"
    if any(ann_dir.glob(f"*_{split}.txt")):
        return ann_dir
    z = root / f"ann_{split}.zip"
    fetch(ANN[split], z)
    with zipfile.ZipFile(z) as zf:
        for m in zf.namelist():
            if m.endswith(".txt"):
                target = ann_dir / Path(m).name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(m))
    z.unlink()
    return ann_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/thumos14")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="only first N videos (for smoke tests)")
    args = ap.parse_args()

    root = Path(args.root)
    ann_dir = ensure_annotations(root, args.split)
    ids = set()
    for f in ann_dir.glob(f"*_{args.split}.txt"):
        if f.name.startswith("Ambiguous"):
            continue
        for line in f.read_text().split("\n"):
            parts = line.split()
            if len(parts) >= 3:
                ids.add(parts[0])
    ids = sorted(ids)[: args.limit]
    vid_dir = root / "videos"
    todo = [v for v in ids if not (vid_dir / f"{v}.mp4").exists()]
    print(f"{len(ids)} annotated videos, {len(todo)} to download -> {vid_dir}", flush=True)

    def one(vid: str) -> str:
        fetch(VID[args.split].format(vid=vid), vid_dir / f"{vid}.mp4")
        return vid

    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one, v): v for v in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:  # keep going; report at the end
                print(f"FAILED {futs[fut]}: {e}", file=sys.stderr, flush=True)
            done += 1
            if done % 10 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
