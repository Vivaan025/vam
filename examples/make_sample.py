"""Write a small synthetic clip to try vam without downloading a dataset.

    python examples/make_sample.py sample.mp4 [--seconds 30]

30 s at 10 fps, 160x120, MPEG-4 with a keyframe every 2 s: a static scene for the first two thirds, then a
moving square. Frame selection should land on the last third. The same recipe the tests use.
"""

import argparse

import cv2
import numpy as np


def make_sample(path: str, seconds: int = 30, fps: int = 10, w: int = 160, h: int = 120, gop_s: int = 2) -> str:
    import av

    n = seconds * fps
    with av.open(path, "w") as c:
        s = c.add_stream("mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.codec_context.gop_size = gop_s * fps
        still = int(n * 2 / 3)
        for i in range(n):
            f = np.full((h, w, 3), 40, np.uint8)
            cv2.rectangle(f, (10, 10), (w - 10, h - 10), (70, 70, 90), 2)
            x = 20 if i < still else 20 + (i - still) % (w - 60)
            cv2.rectangle(f, (x, 40), (x + 30, 70), (0, 220, 60), -1)
            for pk in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="sample.mp4")
    ap.add_argument("--seconds", type=int, default=30)
    a = ap.parse_args()
    print("wrote", make_sample(a.path, a.seconds))
