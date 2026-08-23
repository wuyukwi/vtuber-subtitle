import re
from pathlib import Path
from ..models import Segment


def transcribe(audio: str | Path, model_name: str = "large-v3", device: str = "auto",
               compute_type: str = "auto", beam_size: int = 5, vad_filter: bool = True,
               max_segment_seconds: float = 15.0, pause_threshold: float = 0.8) -> list[Segment]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -e .") from exc
    if device == "auto":
        device = "cuda" if _cuda_available() else "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    vad_parameters = {"min_silence_duration_ms": 300, "speech_pad_ms": 400,
                      "min_speech_duration_ms": 80} if vad_filter else None
    chunks, _ = model.transcribe(
        str(audio), language="ja", beam_size=beam_size, best_of=5,
        temperature=0.0, compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0, no_speech_threshold=0.6,
        vad_filter=vad_filter, vad_parameters=vad_parameters,
        hallucination_silence_threshold=2.0,
        condition_on_previous_text=False, word_timestamps=True)
    raw = _split_by_words(list(chunks), max_segment_seconds, pause_threshold)
    raw = _merge_short_fragments(raw)
    raw = _build_sentences(raw)
    return _remove_repeated_hallucinations(raw)


def _split_by_words(chunks: list, max_segment_seconds: float,
                    pause_threshold: float) -> list[Segment]:
    result: list[Segment] = []
    for chunk in chunks:
        words = [word for word in (getattr(chunk, "words", None) or [])
                 if getattr(word, "word", "").strip()]
        if not words:
            text = chunk.text.strip()
            if text:
                _append_text_with_split(result, float(chunk.start), float(chunk.end), text)
            continue
        group = [words[0]]
        for word in words[1:]:
            previous = group[-1]
            gap = float(word.start) - float(previous.end)
            duration = float(previous.end) - float(group[0].start)
            punctuation_break = previous.word.strip().endswith(("。", "？", "！", ".", "?", "!"))
            safe_pause_break = _safe_pause_boundary(group, word)
            safe_duration_break = _safe_duration_boundary(group, word, max_segment_seconds)
            if (punctuation_break or
                    ((gap >= pause_threshold or safe_duration_break) and safe_pause_break)):
                _append_word_group(result, group)
                group = []
            group.append(word)
        _append_word_group(result, group)
    return result


def _safe_pause_boundary(group: list, next_word) -> bool:
    previous_text = group[-1].word.strip()
    if len(previous_text) >= 2:
        return True
    if previous_text in {"は", "が", "を", "に", "へ", "で", "と", "も", "の", "から", "まで"}:
        return True
    # A one-character BPE token such as ゲ or 一 is usually not a real boundary.
    return False


def _safe_duration_boundary(group: list, next_word, max_segment_seconds: float) -> bool:
    duration = float(group[-1].end) - float(group[0].start)
    return duration >= max_segment_seconds and _safe_pause_boundary(group, next_word)


def _append_word_group(result: list[Segment], words: list) -> None:
    if not words:
        return
    text = "".join(word.word for word in words).strip()
    if not text:
        return
    _append_text_with_split(result, float(words[0].start), float(words[-1].end), text)


def _append_text_with_split(result: list[Segment], start: float, end: float, text: str) -> None:
    pieces: list[str] = []
    for piece in _split_short_response(text):
        pieces.extend(_split_clause_text(piece))
    if len(pieces) == 1:
        result.append(Segment(len(result), start, end, text))
        return
    duration = end - start
    cursor = 0
    for piece in pieces:
        next_cursor = cursor + len(piece)
        piece_start = start + duration * cursor / len(text)
        piece_end = start + duration * next_cursor / len(text)
        result.append(Segment(len(result), piece_start, piece_end, piece))
        cursor = next_cursor


def _split_clause_text(text: str) -> list[str]:
    """Split common Japanese clause boundaries without splitting a short word."""
    if len(text) < 10:
        return [text]
    trailing = re.search(r"(?:\s+)(あの|えっと|その)$", text)
    if trailing and trailing.start() >= 6:
        return [text[:trailing.start()].rstrip(), trailing.group(1)]
    cuts: list[int] = []
    for match in re.finditer(r"か(?=っていう|って)", text):
        cuts.append(match.start() + 1)
    for match in re.finditer(r"は", text):
        if match.start() >= 8 and len(text) - match.end() >= 4:
            cuts.append(match.end())
    for match in re.finditer(r"に", text):
        if 6 <= match.start() < 15 and len(text) - match.end() >= 5:
            cuts.append(match.end())
    if not cuts:
        return [text]
    pieces = []
    previous = 0
    for cut in sorted(set(cuts)):
        pieces.append(text[previous:cut])
        previous = cut
    pieces.append(text[previous:])
    return [piece for piece in pieces if piece]


def _split_short_response(text: str) -> list[str]:
    """Separate short Japanese acknowledgements from the following clause."""
    match = re.match(r"^(はい|ええ|うん|あの|えっと)(?=\S)", text)
    if not match:
        return [text]
    point = match.end()
    return [text[:point], text[point:]]


_KANA_START = "\u3041"
_KANA_END = "\u30ff"


def _is_kana(char: str) -> bool:
    return _KANA_START <= char <= _KANA_END


_COMPLETE_ENDINGS = (
    "ですね", "ですか", "ましたね", "ません", "ました", "です", "ます",
    "でしょう", "なかった", "ない", "なぁ", "な", "ね", "よ", "ぞ", "わ",
    "か", "けど", "から", "ので", "って", "んだ", "ん", "の", "だ", "と",
    "も", "へ", "に", "で", "が", "は", "を", "から", "まで", "では", "ではね",
)
_TRAILING_FILLERS = ("あの", "えっと", "うん", "ええ", "ねえ", "ああ")


def _should_merge_fragments(previous: Segment, current: Segment) -> bool:
    prev_text = previous.japanese.strip()
    curr_text = current.japanese.strip()
    if not prev_text or not curr_text:
        return False
    if prev_text in {"はい", "ええ", "うん", "あの", "えっと", "そう", "じゃ", "じゃあ"}:
        return False
    if current.start - previous.end > 0.8:
        return False
    if prev_text.endswith(("。", "？", "！", ".", "?", "!")):
        return False
    if not _is_kana(prev_text[-1]) or not _is_kana(curr_text[0]):
        return False
    if prev_text.endswith(_TRAILING_FILLERS):
        return False
    if prev_text.endswith(_COMPLETE_ENDINGS):
        return False
    return True


def _merge_short_fragments(segments: list[Segment]) -> list[Segment]:
    """Join sub-word fragments such as コラボした / いです back into one subtitle."""
    merged: list[Segment] = []
    for segment in segments:
        if merged and _should_merge_fragments(merged[-1], segment):
            previous = merged.pop()
            segment = Segment(previous.id, previous.start, segment.end,
                              previous.japanese + segment.japanese, segment.chinese)
        merged.append(Segment(len(merged), segment.start, segment.end,
                              segment.japanese, segment.chinese))
    return merged


def _remove_repeated_hallucinations(segments: list[Segment]) -> list[Segment]:
    """Drop long identical phrases repeated through a silent tail."""
    cleaned: list[Segment] = []
    for segment in segments:
        if (cleaned and len(segment.japanese) >= 8 and
                segment.japanese == cleaned[-1].japanese and
                segment.start <= cleaned[-1].end + 0.15):
            continue
        cleaned.append(Segment(len(cleaned), segment.start, segment.end, segment.japanese, segment.chinese))
    return cleaned


_SENTENCE_ENDERS = (
    "でした", "ましたね", "ですね", "ですか", "ましょう", "ません", "ました",
    "です", "ます", "ですよ", "ですよね", "ますか", "んです", "んだ",
    "だよ", "だね", "だな", "だぞ", "だわ", "かな", "かしら", "よね",
    "。", "！", "？", ".", "?", "!", "～", "〜",
    "よ", "ね", "ぞ", "わ", "なぁ",
)
_LONG_PAUSE_SECONDS = 2.5
_POLITE_PAUSE_SECONDS = 0.35
_CLAUSE_PAUSE_SECONDS = 0.6
_RESPONSE_WORDS = {"はい", "ええ", "うん", "あの", "えっと", "そう", "じゃ", "じゃあ"}
_CLAUSE_ENDERS = ("かっていう", "っていう", "という", "けど", "から", "ので", "んで")


def _is_sentence_final(text: str) -> bool:
    text = text.rstrip()
    for ender in _SENTENCE_ENDERS:
        if text.endswith(ender):
            return True
    return False


def _build_sentences(segments: list[Segment], max_duration: float = 12.0) -> list[Segment]:
    """Group consecutive fragments into complete sentences, splitting only at
    strong punctuation, a long pause, a polite/clause ending with a pause, so
    subtitles are not cut mid-sentence and stay within a readable length."""
    sentences: list[Segment] = []
    buffer_start: float | None = None
    buffer_end: float | None = None
    buffer_text: list[str] = []

    def flush() -> None:
        nonlocal buffer_start, buffer_end, buffer_text
        if buffer_text and buffer_start is not None and buffer_end is not None:
            sentences.append(Segment(len(sentences), buffer_start, buffer_end,
                                     "".join(buffer_text)))
        buffer_start = None
        buffer_end = None
        buffer_text = []

    for segment in segments:
        text = segment.japanese.strip()
        if not text:
            continue
        if not buffer_text and text in _RESPONSE_WORDS:
            sentences.append(Segment(len(sentences), segment.start, segment.end, text))
            continue
        if buffer_text:
            gap = segment.start - buffer_end
            joined = "".join(buffer_text)
            strong_end = joined.endswith(("。", "！", "？", ".", "?", "!"))
            polite_end = _is_sentence_final(joined)
            clause_end = joined.endswith(_CLAUSE_ENDERS)
            if text in _RESPONSE_WORDS:
                flush()
            elif gap >= _LONG_PAUSE_SECONDS or strong_end:
                flush()
            elif polite_end and gap >= _POLITE_PAUSE_SECONDS:
                flush()
            elif clause_end and gap >= _CLAUSE_PAUSE_SECONDS:
                flush()
            elif segment.end - buffer_start >= max_duration and gap >= _POLITE_PAUSE_SECONDS:
                flush()
        if not buffer_text and text in _RESPONSE_WORDS:
            sentences.append(Segment(len(sentences), segment.start, segment.end, text))
            continue
        if buffer_start is None:
            buffer_start = segment.start
        buffer_text.append(text)
        buffer_end = segment.end
    flush()
    return sentences


def _cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False
