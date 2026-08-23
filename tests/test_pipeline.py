from types import SimpleNamespace
from vtuber_subtitle.asr.faster_whisper import _split_by_words, _merge_short_fragments
from vtuber_subtitle.models import Segment
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
    result = _split_by_words(chunks, max_segment_seconds=15, pause_threshold=0.8)
    assert len(result) == 2
    assert result[0].japanese == "これはテストです"
    assert result[1].japanese == "次の文"


def test_split_by_punctuation():
    words = [SimpleNamespace(word="ちょっと待って。", start=1.0, end=1.8),
             SimpleNamespace(word="次いこう", start=1.9, end=2.6)]
    chunks = [SimpleNamespace(start=1.0, end=2.6, text="ちょっと待って。次いこう", words=words)]
    result = _split_by_words(chunks, max_segment_seconds=15, pause_threshold=0.8)
    assert [item.japanese for item in result] == ["ちょっと待って。", "次いこう"]


def _seg(seq_id, start, end, text):
    return Segment(seq_id, start, end, text)


def test_merge_subword_fragments():
    merged = _merge_short_fragments([
        _seg(0, 0.0, 0.3, "ゲ"),
        _seg(1, 0.3, 1.0, "ームは好きだし"),
    ])
    assert len(merged) == 1
    assert merged[0].japanese == "ゲームは好きだし"
    assert merged[0].start == 0.0
    assert merged[0].end == 1.0


def test_do_not_merge_response_or_long_fragments():
    merged = _merge_short_fragments([
        _seg(0, 0.0, 0.5, "はい"),
        _seg(1, 0.6, 2.0, "で誕生日は12月24日です"),
        _seg(2, 2.1, 3.0, "コラボした"),
        _seg(3, 3.1, 4.0, "いです"),
    ])
    assert [item.japanese for item in merged] == [
        "はい", "で誕生日は12月24日です", "コラボした", "いです"
    ]
