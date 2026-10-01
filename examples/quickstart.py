"""Minimal end-to-end: load a video, pick frames, run a pretrained model, print predictions."""

import sys

import vam

path = sys.argv[1] if len(sys.argv) > 1 else "sample.mp4"

video = vam.load_video(path)
model = vam.get_model("videomae-base-kinetics")

with vam.profile("uniform") as p:
    out = model(video, sampler=vam.UniformSampler(16))
print(p.report())
print(out.top_k(3))

with vam.profile("motion") as p:
    out = model(video, sampler=vam.MotionSampler(16))
print(p.report())
print(out.top_k(3))

# streaming: overlapping 16-frame windows
for i, chunk in enumerate(vam.Stream(path, window=16, hop=8)):
    print(i, model(chunk).top_k(1)[0])
    if i == 3:
        break
