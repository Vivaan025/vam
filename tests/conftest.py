import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory):
    """60-frame 10 fps clip: 30 static frames, then a moving square for 30 frames.

    Motion-based sampling should put most of its frames in the second half.
    """
    path = str(tmp_path_factory.mktemp("vid") / "synthetic.avi")
    w, h, fps, n = 128, 96, 10, 60
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x = 10 if i < 30 else 10 + (i - 30) * 3
        cv2.rectangle(frame, (x, 30), (x + 20, 50), (0, 255, 0), -1)
        out.write(frame)
    out.release()
    return path, n, fps


@pytest.fixture(scope="session")
def h264_video(tmp_path_factory):
    """30 s @ 10 fps: static for 20 s, then a moving square for 10 s. Written with PyAV so the
    GOP structure is real (keyframe every 2 s)."""
    import av

    path = str(tmp_path_factory.mktemp("kf") / "clip.mp4")
    fps, n, w, h = 10, 300, 160, 120
    with av.open(path, "w") as c:
        s = c.add_stream("mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.codec_context.gop_size = 20
        for i in range(n):
            f = np.zeros((h, w, 3), np.uint8)
            x = 20 if i < 200 else 20 + (i - 200)
            cv2.rectangle(f, (x, 40), (x + 30, 70), (0, 220, 60), -1)
            for pk in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    return path, fps, n
