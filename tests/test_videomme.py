"""Experiment 020: the Video-MME reader, the MCQ prompt and the answer-letter parser."""

import pytest

from vam.core import ActionOutput
from vam.models.videollm import MCQ_PROMPT, answer_letter


def test_answer_letter_parsing():
    assert answer_letter("B") == "B"
    assert answer_letter("b.") == "B"
    assert answer_letter("The best answer is (C).") == "C"
    assert answer_letter("Option D: because") == "D"
    assert answer_letter("I cannot tell from the video") is None
    assert answer_letter("ABOUT") is None  # no stand-alone letter
    assert "Respond with only the letter" in MCQ_PROMPT.format(question="q?", options="A. x\nB. y")


def test_action_output_text_field():
    out = ActionOutput(text="hello")
    assert out.text == "hello" and out.logits is None
    with pytest.raises(ValueError):
        out.top_k()


def test_videomme_reader_groups_questions(tmp_path):
    pq = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pqt

    from vam.data.videomme import VideoMME

    rows = [
        {"video_id": "001", "duration": "short", "domain": "Knowledge", "sub_category": "Humanity", "url": "u", "videoID": "abc",
         "question_id": "001-1", "task_type": "Counting Problem", "question": "q1", "options": ["A. 1", "B. 2", "C. 3", "D. 4"], "answer": "C"},
        {"video_id": "001", "duration": "short", "domain": "Knowledge", "sub_category": "Humanity", "url": "u", "videoID": "abc",
         "question_id": "001-2", "task_type": "OCR Problems", "question": "q2", "options": ["A. 1", "B. 2", "C. 3", "D. 4"], "answer": "A"},
        {"video_id": "002", "duration": "long", "domain": "Sports", "sub_category": "Basketball", "url": "u", "videoID": "zzz",
         "question_id": "002-1", "task_type": "Counting Problem", "question": "q3", "options": ["A. 1", "B. 2", "C. 3", "D. 4"], "answer": "B"},
    ]
    root = tmp_path / "videomme"
    (root / "raw" / "videomme").mkdir(parents=True)
    (root / "videos" / "chunk1").mkdir(parents=True)
    pqt.write_table(pq.Table.from_pylist(rows), root / "raw" / "videomme" / "test-00000-of-00001.parquet")
    (root / "videos" / "chunk1" / "abc.mp4").write_bytes(b"")
    ds = VideoMME(root)
    assert len(ds) == 1 and ds.questions == 2 and ds.missing == ["zzz"]
    assert ds[0].youtube_id == "abc" and ds[0].duration == "short" and [q.question_id for q in ds[0].questions] == ["001-1", "001-2"]
    assert ds[0].questions[0].options[2] == "C. 3"
    ds_all = VideoMME(root, require_video=False)
    assert len(ds_all) == 2 and ds_all[1].path == "" and ds_all[1].duration == "long"
