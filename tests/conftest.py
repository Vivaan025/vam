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
