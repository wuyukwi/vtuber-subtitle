from vtuber_subtitle.pipeline import parse_time


def test_parse_time():
    assert parse_time("00:20:00") == 1200
    assert parse_time("1:02.5") == 62.5
    assert parse_time("12.5") == 12.5
