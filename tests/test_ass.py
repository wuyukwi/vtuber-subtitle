from vtuber_subtitle.ass import ass_time, write_ass
from vtuber_subtitle.models import Segment


def test_ass_time():
    assert ass_time(65.239) == "0:01:05.24"


def test_write_ass(tmp_path):
    target = write_ass([Segment(1, 0, 2.5, "こんにちは{}", "你好")], tmp_path / "x.ass")
    text = target.read_text(encoding="utf-8-sig")
    assert "Dialogue: 0,0:00:00.00,0:00:02.50,Japanese" in text
    assert "こんにちは\\{\\}" in text
    assert "你好" in text
