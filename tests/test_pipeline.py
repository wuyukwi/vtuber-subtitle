from types import SimpleNamespace
from vtuber_subtitle.asr.faster_whisper import _split_by_words, _split_short_response
from vtuber_subtitle.pipeline import parse_time


def test_parse_time():
    assert parse_time("00:20:00") == 1200
    assert parse_time("1:02.5") == 62.5
    assert parse_time("12.5") == 12.5


def test_split_by_word_timestamps():
    words = [SimpleNamespace(word="これは", start=1.0, end=1.4),
             SimpleNamespace(word="テストです", start=1.5, end=2.2),
             SimpleNamespace(word="次の文", start=3.0, end=3.6)]
    chunks = [SimpleNamespace(start=1.0, end=3.6, text="これはテストです次の文", words=words)]
    result = _split_by_words(chunks, max_segment_seconds=7, pause_threshold=0.6)
    assert len(result) == 2
    assert result[0].start == 1.0
    assert result[0].end == 2.2
    assert result[1].japanese == "次の文"


def test_split_short_japanese_response():
    assert _split_short_response("はいで") == ["はい", "で"]
    assert _split_short_response("はいあの誕生日") == ["はい", "あの誕生日"]
    assert _split_short_response("はい想像") == ["はい", "想像"]
    assert _split_short_response("はい") == ["はい"]


def test_split_short_response_without_word_timestamps():
    chunks = [SimpleNamespace(start=1.0, end=4.0, text="はい想像", words=None)]
    result = _split_by_words(chunks, max_segment_seconds=7, pause_threshold=0.8)
    assert [item.japanese for item in result] == ["はい", "想像"]
