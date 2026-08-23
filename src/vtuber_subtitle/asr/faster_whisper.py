from pathlib import Path
from ..models import Segment


def transcribe(audio: str | Path, model_name: str = "large-v3", device: str = "auto",
               compute_type: str = "auto", beam_size: int = 5, vad_filter: bool = True,
               max_segment_seconds: float = 7.0, pause_threshold: float = 0.6) -> list[Segment]:
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
                result.append(Segment(len(result), float(chunk.start), float(chunk.end), text))
            continue
        group = [words[0]]
        for word in words[1:]:
            previous = group[-1]
            gap = float(word.start) - float(previous.end)
            duration = float(previous.end) - float(group[0].start)
            punctuation_break = previous.word.strip().endswith(("。", "？", "！", ".", "?", "!"))
            if gap >= pause_threshold or duration >= max_segment_seconds or punctuation_break:
                _append_word_group(result, group)
                group = []
            group.append(word)
        _append_word_group(result, group)
    return result


def _append_word_group(result: list[Segment], words: list) -> None:
    if not words:
        return
    text = "".join(word.word for word in words).strip()
    if text:
        result.append(Segment(len(result), float(words[0].start), float(words[-1].end), text))


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
