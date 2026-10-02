import pytest

from vam.data import UCF101


def _fake_ucf(tmp_path):
    root = tmp_path / "ucf"
    (root / "UCF-101" / "Archery").mkdir(parents=True)
    (root / "UCF-101" / "Biking").mkdir(parents=True)
    lists = root / "ucfTrainTestlist"
    lists.mkdir()
    (lists / "classInd.txt").write_text("1 Archery\n2 Biking\n")
    (lists / "trainlist01.txt").write_text("Archery/v_Archery_g01_c01.avi 1\nBiking/v_Biking_g01_c01.avi 2\n")
    (lists / "testlist01.txt").write_text("Archery/v_Archery_g02_c01.avi\n")
    return root


def test_ucf101_parses_splits(tmp_path):
    root = _fake_ucf(tmp_path)
    train = UCF101(root, "train", 1)
    test = UCF101(root, "test", 1)
    assert train.classes == ["Archery", "Biking"]
    assert len(train) == 2 and len(test) == 1
    assert train[1].label == 1 and train[1].class_name == "Biking"
    assert test[0].path.endswith("v_Archery_g02_c01.avi")
    assert train.labels() == [0, 1]


def test_ucf101_limit_is_deterministic(tmp_path):
    root = _fake_ucf(tmp_path)
    a = UCF101(root, "train", 1, limit=1, seed=3)
    b = UCF101(root, "train", 1, limit=1, seed=3)
    assert a.clips == b.clips and len(a) == 1


def test_ucf101_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        UCF101(tmp_path / "nope")
