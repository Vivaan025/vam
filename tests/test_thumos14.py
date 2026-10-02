import pytest

from vam.data import THUMOS14


def _fake(tmp_path, with_video=True):
    root = tmp_path / "th"
    ann = root / "annotations"
    ann.mkdir(parents=True)
    (ann / "Diving_val.txt").write_text("video_validation_0000001  10.0 12.0\nvideo_validation_0000001  20.0 25.0\n")
    (ann / "GolfSwing_val.txt").write_text("video_validation_0000002  1.0 2.0\n")
    (ann / "Ambiguous_val.txt").write_text("video_validation_0000001  30.0 31.0\n")
    vids = root / "videos"
    vids.mkdir()
    if with_video:
        (vids / "video_validation_0000001.mp4").write_bytes(b"x")
    return root


def test_parses_segments_and_skips_ambiguous(tmp_path):
    ds = THUMOS14(_fake(tmp_path), "val", require_video=False)
    assert ds.classes == ["Diving", "GolfSwing"]
    assert len(ds) == 2
    v = ds[0]
    assert v.video_id == "video_validation_0000001"
    assert len(v.segments) == 2 and v.classes == {"Diving"}


def test_require_video_filters_missing(tmp_path):
    ds = THUMOS14(_fake(tmp_path), "val", require_video=True)
    assert ds.video_ids() == ["video_validation_0000001"]


def test_action_fraction_and_membership(tmp_path):
    v = THUMOS14(_fake(tmp_path), "val", require_video=False)[0]
    assert v.action_fraction(100.0) == pytest.approx(0.07)
    assert v.frame_in_action(11.0) and not v.frame_in_action(15.0)
