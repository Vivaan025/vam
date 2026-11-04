"""Download Charades (480p videos + annotations) for experiment 012.

    python scripts/download_charades.py --root data/charades

Fetches Charades.zip (annotations, 3.5 MB) and Charades_v1_480.zip (13 GB) from the Allen AI public bucket
with resumable wget, then extracts to <root>/Charades/ and <root>/Charades_v1_480/. Safe to re-run.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

URLS = {
    "Charades.zip": "https://ai2-public-datasets.s3.amazonaws.com/charades/Charades.zip",
    "Charades_v1_480.zip": "https://ai2-public-datasets.s3.amazonaws.com/charades/Charades_v1_480.zip",
}


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("wget"):
        subprocess.run(["wget", "-q", "-c", "-O", str(dest), url], check=True)
    else:
        import urllib.request

        with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)


def extract(z: Path, root: Path, marker: Path) -> None:
    if marker.exists():
        return
    with zipfile.ZipFile(z) as zf:
        zf.extractall(root)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/charades")
    ap.add_argument("--keep-zips", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    for name, url in URLS.items():
        z = root / name
        marker = root / ("Charades/Charades_v1_classes.txt" if name == "Charades.zip" else "Charades_v1_480")
        if not marker.exists():
            if not z.exists():
                print(f"downloading {name}", flush=True)
                fetch(url, z)
            print(f"extracting {name}", flush=True)
            extract(z, root, marker)
            if not args.keep_zips:
                z.unlink(missing_ok=True)
    n = len(list((root / "Charades_v1_480").glob("*.mp4")))
    print(f"{n} videos under {root / 'Charades_v1_480'}", flush=True)


if __name__ == "__main__":
    main()
