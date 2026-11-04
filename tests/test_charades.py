from pathlib import Path

from vam.data import Charades


def test_charades_parses_annotations(tmp_path: Path):
    ann = tmp_path / "Charades"
    ann.mkdir()
    (ann / "Charades_v1_classes.txt").write_text("c000 Holding some clothes\nc001 Putting clothes somewhere\n")
    (ann / "Charades_v1_test.csv").write_text(
        "id,subject,scene,quality,relevance,verified,script,objects,descriptions,actions,length\n"
        "AAAAA,1,Kitchen,5,5,Yes,s,o,d,c000 11.90 21.20;c001 0.00 12.60,30.5\n"
        "BBBBB,2,Bedroom,5,5,Yes,s,o,d,,20.0\n"
        "CCCCC,2,Bedroom,5,5,Yes,s,o,d,c999 1.0 2.0;c001 3.0 1.0,20.0\n"
    )
    ds = Charades(tmp_path, "test", require_video=False)
    assert ds.classes == ["Holding some clothes", "Putting clothes somewhere"]
    assert ds.video_ids() == ["AAAAA"]  # no segments -> dropped; unknown code / inverted interval -> dropped
    v = ds[0]
    assert v.classes == {"Holding some clothes", "Putting clothes somewhere"}
    assert v.frame_in_action(5.0) and not v.frame_in_action(25.0)
    assert abs(v.action_fraction(30.5) - 21.2 / 30.5) < 1e-6
    assert ds.lengths["AAAAA"] == 30.5
    assert v.path.replace("\\", "/").endswith("Charades_v1_480/AAAAA.mp4")
