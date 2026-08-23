from types import SimpleNamespace
from vtuber_subtitle.asr.faster_whisper import (
    _split_by_words, _split_clause_text, _split_short_response, _merge_short_fragments,
    _build_sentences, _is_sentence_final,
)
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


def test_split_japanese_clauses():
    assert _split_clause_text("クリスマスプレゼントとお誕生日プレゼントは一緒にされましたかっていう質問は") == [
        "クリスマスプレゼントとお誕生日プレゼントは", "一緒にされましたか", "っていう質問は"
    ]
    assert _split_clause_text("ゴールデンタイムに午後寝ることないなぁと思ってからはちょっと今を満喫しようと思って") == [
        "ゴールデンタイムに", "午後寝ることないなぁと思ってからは", "ちょっと今を満喫しようと思って"
    ]
    assert _split_clause_text("起きてるなぁと思って あの") == ["起きてるなぁと思って", "あの"]


def _seg(seq_id, start, end, text):
    return Segment(seq_id, start, end, text)


def test_merge_subword_fragments():
    merged = _merge_short_fragments([
        _seg(0, 0.0, 1.0, "絵とか漫画は結構濃いかもし"),
        _seg(1, 1.0, 1.8, "れないですね"),
    ])
    assert len(merged) == 1
    assert merged[0].japanese == "絵とか漫画は結構濃いかもしれないですね"
    assert merged[0].start == 0.0
    assert merged[0].end == 1.8


def test_merge_cohesive_fragment():
    merged = _merge_short_fragments([
        _seg(0, 5.0, 8.0, "歌もゲームもコラボした"),
        _seg(1, 8.0, 9.0, "いですお願いします"),
    ])
    assert len(merged) == 1
    assert merged[0].japanese == "歌もゲームもコラボしたいですお願いします"


def test_do_not_merge_response_or_complete():
    merged = _merge_short_fragments([
        _seg(0, 0.0, 0.5, "はい"),
        _seg(1, 0.6, 1.2, "で誕生日は12月24日です"),
        _seg(2, 2.0, 3.0, "そんな感じです"),
        _seg(3, 3.1, 4.0, "おすすめの漫画あったら"),
    ])
    assert [item.japanese for item in merged] == [
        "はい", "で誕生日は12月24日です", "そんな感じです", "おすすめの漫画あったら"
    ]


def test_do_not_merge_negative_complete_ending():
    merged = _merge_short_fragments([
        _seg(0, 0.0, 2.0, "ーとはまた違うかもしれない"),
        _seg(1, 2.0, 4.0, "ないんですけど"),
    ])
    assert [item.japanese for item in merged] == ["ーとはまた違うかもしれない", "ないんですけど"]


def test_is_sentence_final():
    assert _is_sentence_final("クリスマスイブです")
    assert _is_sentence_final("好きですよね")
    assert not _is_sentence_final("一緒にされましたかっていう質問は")
    assert not _is_sentence_final("クリスマスプレゼントとお誕生日プレゼントは")
    assert not _is_sentence_final("ちょっと今を満喫しようと思って")


def test_build_sentences_groups_fragments():
    merged = _build_sentences([
        _seg(0, 0.0, 1.0, "はい"),
        _seg(1, 1.0, 2.0, "で誕生日は12月24日です"),
        _seg(2, 2.1, 3.0, "クリスマスイブです"),
        _seg(3, 5.5, 6.5, "次の質問に移ります"),
    ])
    assert [item.japanese for item in merged] == [
        "はい", "で誕生日は12月24日ですクリスマスイブです", "次の質問に移ります"
    ]


def test_build_sentences_splits_at_sentence_end():
    merged = _build_sentences([
        _seg(0, 0.0, 1.0, "今日は楽しいです"),
        _seg(1, 2.5, 3.5, "明日も楽しみです"),
    ])
    assert [item.japanese for item in merged] == ["今日は楽しいです", "明日も楽しみです"]


def test_build_sentences_splits_at_strong_punctuation():
    merged = _build_sentences([
        _seg(0, 0.0, 1.0, "ちょっと待って。"),
        _seg(1, 1.1, 2.0, "次いこう"),
    ])
    assert [item.japanese for item in merged] == ["ちょっと待って。", "次いこう"]
