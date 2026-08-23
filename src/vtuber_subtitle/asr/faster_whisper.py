from pathlib import Path
from ..models import Segment

_RESPONSE_WORDS = {"はい", "ええ", "うん", "あの", "えっと", "そう", "じゃ", "じゃあ"}


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
    raw = _merge_isolated_fragments(raw)
    return _remove_repeated_hallucinations(raw)


def _split_by_words(chunks: list, max_segment_seconds: float,
                    pause_threshold: float) -> list[Segment]:
    """Split a Whisper chunk at punctuation or word-level pauses only."""
    result: list[Segment] = []
    for chunk in chunks:
        words = [word for word in (getattr(chunk, "words", None) or [])
                 if getattr(word, "word", "").strip()]
        if not words:
            text = chunk.text.strip()
            if text:
                result.append(Segment(len(result), float(chunk.start), float(chunk.end), text))
            continue
        group = [words[0]]
        for word in words[1:]:
            previous = group[-1]
            gap = round(float(word.start) - float(previous.end), 3)
            duration = round(float(previous.end) - float(group[0].start), 3)
            punctuation_break = previous.word.strip().endswith(("。", "？", "！", ".", "?", "!"))
            if (punctuation_break or
                    (gap >= pause_threshold and _safe_pause_boundary(group, word)) or
                    (duration >= max_segment_seconds and _safe_pause_boundary(group, word))):
                _append_word_group(result, group)
                group = []
            group.append(word)
        _append_word_group(result, group)
    return result


def _safe_pause_boundary(group: list, next_word) -> bool:
    """Avoid breaking inside a one-character BPE token such as ゲ or 一."""
    previous_text = group[-1].word.strip()
    if len(previous_text) >= 2:
        return True
    if previous_text in {"は", "が", "を", "に", "へ", "で", "と", "も", "の", "から", "まで"}:
        return True
    return False


def _append_word_group(result: list[Segment], words: list) -> None:
    if not words:
        return
    text = "".join(word.word for word in words).strip()
    if text:
        result.append(Segment(len(result), float(words[0].start), float(words[-1].end), text))


def _merge_short_fragments(segments: list[Segment]) -> list[Segment]:
    """Join tiny sub-word fragments such as ゲ / ーム or 一 / 緒."""
    merged: list[Segment] = []
    for segment in segments:
        if (merged and len(merged[-1].japanese) <= 3 and
                merged[-1].japanese not in _RESPONSE_WORDS and
                segment.start - merged[-1].end <= 0.2):
            previous = merged.pop()
            segment = Segment(previous.id, previous.start, segment.end,
                              previous.japanese + segment.japanese, segment.chinese)
        merged.append(Segment(len(merged), segment.start, segment.end,
                              segment.japanese, segment.chinese))
    return merged


_STANDALONE_WORDS: set[str] = set()
_FRAGMENT_MERGE_GAP = 2.5


def _merge_isolated_fragments(segments: list[Segment]) -> list[Segment]:
    """Merge short filler fragments such as なんか / あと / もう into the closer
    neighbor instead of leaving them as one-word subtitle lines."""
    result: list[Segment] = []
    index = 0
    total = len(segments)
    while index < total:
        segment = segments[index]
        text = segment.japanese.strip()
        if len(text) <= 3 and text not in _STANDALONE_WORDS:
            candidates: list[tuple[str, float]] = []
            if result and segment.start - result[-1].end <= _FRAGMENT_MERGE_GAP:
                candidates.append(("previous", segment.start - result[-1].end))
            if index + 1 < total:
                next_gap = segments[index + 1].start - segment.end
                if next_gap <= _FRAGMENT_MERGE_GAP:
                    candidates.append(("next", next_gap))
            if candidates:
                where, _ = min(candidates, key=lambda item: item[1])
                if where == "previous":
                    previous = result.pop()
                    result.append(Segment(previous.id, previous.start, segment.end,
                                          previous.japanese + text, previous.chinese))
                    index += 1
                    continue
                next_seg = segments[index + 1]
                result.append(Segment(segment.id, segment.start, next_seg.end,
                                      text + next_seg.japanese, segment.chinese))
                index += 2
                continue
        result.append(Segment(len(result), segment.start, segment.end,
                              segment.japanese, segment.chinese))
        index += 1
    return result


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


def _cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False
