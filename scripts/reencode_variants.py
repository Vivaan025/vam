"""Experiment 011: re-encode THUMOS14 videos into resolution / codec / keyframe-density variants.

Every number before 011 was measured on 320x180 H.264 with the default keyframe interval. Each variant
below changes one thing. The output roots look like a THUMOS14 root (videos/ + annotations/ symlink), so
every benchmark takes them through --root.

    python scripts/reencode_variants.py --root data/thumos14 --out data/thumos14_variants --limit 50 --workers 6

Variants (name: ffmpeg video options):
  h264_1080p   upscale to 1920x1080, libx264 crf 23, default GOP (250)   -> decode cost of HD input
  hevc_720p    1280x720, libx265 crf 28                                   -> another codec, PyAV + OpenCV paths
  vp9_720p     1280x720, libvpx-vp9 crf 33                                -> non-MPEG codec in mp4
  h264_gop30   320x180, libx264, keyframe every 30 frames (1 s)           -> dense keyframes: scan cost up
  h264_gop900  320x180, libx264, keyframe every 900 frames (30 s)         -> sparse keyframes: windows coarse
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

VARIANTS: dict[str, list[str]] = {
    "h264_1080p": ["-vf", "scale=1920:1080:flags=lanczos", "-c:v", "libx264", "-crf", "23", "-preset", "fast"],
    "hevc_720p": ["-vf", "scale=1280:720:flags=lanczos", "-c:v", "libx265", "-crf", "28", "-preset", "fast", "-tag:v", "hvc1"],
    "vp9_720p": ["-vf", "scale=1280:720:flags=lanczos", "-c:v", "libvpx-vp9", "-crf", "33", "-b:v", "0", "-row-mt", "1", "-cpu-used", "4"],
    "h264_gop30": ["-c:v", "libx264", "-crf", "23", "-preset", "fast", "-g", "30", "-keyint_min", "30", "-sc_threshold", "0"],
    "h264_gop900": ["-c:v", "libx264", "-crf", "23", "-preset", "fast", "-g", "900", "-keyint_min", "900", "-sc_threshold", "0"],
}


def encode(src: Path, dst: Path, opts: list[str], threads: int) -> tuple[Path, float, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", str(threads), "-i", str(src),
           *opts, "-an", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        return dst, time.time() - t0, r.stderr.strip()[-200:]
    tmp.replace(dst)
    return dst, time.time() - t0, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/thumos14")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="data/thumos14_variants")
    ap.add_argument("--variants", nargs="+", default=sorted(VARIANTS), choices=sorted(VARIANTS))
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4, help="ffmpeg threads per job")
    args = ap.parse_args()

    from vam.data import THUMOS14
    ds = THUMOS14(args.root, args.split)
    vids = ds.videos[: args.limit]
    out = Path(args.out)
    jobs = []
    for name in args.variants:
        vroot = out / name
        (vroot / "videos").mkdir(parents=True, exist_ok=True)
        link = vroot / "annotations"
        if not link.exists():
            os.symlink(os.path.relpath(Path(args.root).resolve() / "annotations", vroot), link)
        for uv in vids:
            dst = vroot / "videos" / f"{uv.video_id}.mp4"
            if not dst.exists():
                jobs.append((Path(uv.path), dst, VARIANTS[name]))
    print(f"{len(vids)} videos x {len(args.variants)} variants: {len(jobs)} encodes -> {out}", flush=True)
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(encode, s, d, o, args.threads) for s, d, o in jobs]
        for f in as_completed(futs):
            dst, secs, err = f.result()
            done += 1
            if err:
                print(f"FAILED {dst}: {err}", flush=True)
            elif done % 10 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  ({dst.parent.parent.name}/{dst.name} {secs:.0f} s)", flush=True)
    for name in args.variants:
        n = len(list((out / name / "videos").glob("*.mp4")))
        size = sum(p.stat().st_size for p in (out / name / "videos").glob("*.mp4")) / 1e9
        print(f"{name}: {n} videos, {size:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
