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


def test_write_ass_with_template(tmp_path):
    template = tmp_path / "template.ass"
    template.write_text("""[Script Info]\nPlayResX: 1280\nPlayResY: 720\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: JP,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,8,10,10,10,1\nStyle: CN,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n""", encoding="utf-8")
    target = write_ass([Segment(1, 3, 4, "日文", "中文")], tmp_path / "templated.ass",
                       template=template, japanese_style="JP", chinese_style="CN")
    text = target.read_text(encoding="utf-8-sig")
    assert "Style: JP" in text
    assert "Style: CN" in text
    assert "Dialogue: 0,0:00:03.00,0:00:04.00,JP" in text
