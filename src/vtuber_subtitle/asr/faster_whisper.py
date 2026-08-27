from pathlib import Path
from ..models import Segment

_RESPONSE_WORDS = {"はい", "ええ", "うん", "あの", "えっと", "そう", "じゃ", "じゃあ"}


# Sentence-final forms: a boundary right after one of these is treated as a hard
# sentence break and is NOT merged away by the readability merge below.
_SENTENCE_ENDINGS = ("です", "ます", "た", "だ", "よ", "ね", "か", "わ", "ぞ", "ぜ", "な", "っ",
                     "よう", "まい", "でしょう", "ください", "ございます", "ました", "でした")


def _is_sentence_end(text: str) -> bool:
    text = text.strip()
    if text.endswith(("。", "？", "！", ".", "?", "!")):
        return True
    return any(text.endswith(suf) for suf in _SENTENCE_ENDINGS)


def merge_to_target(raw: list[Segment], target_mean: float = 3.6, max_gap: float = 1.0,
                    hard_max: float = 10.0) -> list[Segment]:
    """Readability merge: pull the average subtitle duration toward ``target_mean`` by
    iteratively folding the shortest segment into the closer neighbour. Never merges
    across a sentence boundary (detected by ``_is_sentence_end``), never creates a
    line longer than ``hard_max``, and only joins segments whose inter-gap is <= ``max_gap``.

    This is adaptive: segments that are already longer than ``target_mean`` on average
    (e.g. a calm stretch) are left untouched, while over-segmented stretches are grouped
    into readable lines — mimicking how human subtitles group short clauses."""
    if not raw or target_mean <= 0:
        return raw
    segs = [Segment(s.id, s.start, s.end, s.japanese, s.chinese) for s in raw]
    while True:
        if not segs:
            break
        cur_mean = sum(s.end - s.start for s in segs) / len(segs)
        if cur_mean >= target_mean:
            break
        best_i, best_dur = -1, float("inf")
        for i, s in enumerate(segs):
            dur = s.end - s.start
            if dur >= best_dur:
                continue
            can_prev = (i > 0 and (s.start - segs[i - 1].end) <= max_gap and
                        (s.end - segs[i - 1].start) <= hard_max and
                        not _is_sentence_end(segs[i - 1].japanese) and
                        not _is_sentence_end(s.japanese))
            can_next = (i + 1 < len(segs) and (segs[i + 1].start - s.end) <= max_gap and
                        (segs[i + 1].end - s.start) <= hard_max and
                        not _is_sentence_end(s.japanese) and
                        not _is_sentence_end(segs[i + 1].japanese))
            if can_prev or can_next:
                best_i, best_dur = i, dur
        if best_i < 0:
            break
        i = best_i
        s = segs[i]
        gp = (s.start - segs[i - 1].end) if i > 0 else float("inf")
        gn = (segs[i + 1].start - s.end) if i + 1 < len(segs) else float("inf")
        can_prev = (i > 0 and gp <= max_gap and (s.end - segs[i - 1].start) <= hard_max and
                    not _is_sentence_end(segs[i - 1].japanese) and not _is_sentence_end(s.japanese))
        can_next = (i + 1 < len(segs) and gn <= max_gap and (segs[i + 1].end - s.start) <= hard_max and
                    not _is_sentence_end(s.japanese) and not _is_sentence_end(segs[i + 1].japanese))
        if can_prev and (not can_next or gp <= gn):
            prev = segs.pop(i - 1)
            segs[i - 1] = Segment(prev.id, prev.start, s.end, prev.japanese + s.japanese, prev.chinese)
        elif can_next:
            nxt = segs.pop(i + 1)
            segs[i] = Segment(s.id, s.start, nxt.end, s.japanese + nxt.japanese, s.chinese)
        else:
            break
    return [Segment(i, s.start, s.end, s.japanese, s.chinese) for i, s in enumerate(segs)]


def transcribe(audio: str | Path, model_name: str = "large-v3", device: str = "auto",
               compute_type: str = "auto", beam_size: int = 10, vad_filter: bool = True,
               max_segment_seconds: float = 10.0, pause_threshold: float = 0.6,
               initial_prompt: str | None = None, merge_target_mean: float = 3.6,
               merge_max_gap: float = 1.0, merge_hard_max: float = 10.0) -> list[Segment]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -e .") from exc
    
    # Add NVIDIA CUDA libraries to PATH if installed via pip
    _add_nvidia_cuda_to_path()
    
    if device == "auto":
        device = "cuda" if _cuda_available() else "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    vad_parameters = {"min_silence_duration_ms": 1000, "speech_pad_ms": 400,
                      "min_speech_duration_ms": 250} if vad_filter else None
    chunks, _ = model.transcribe(
        str(audio), language="ja", beam_size=beam_size, best_of=5,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0, no_speech_threshold=0.6,
        vad_filter=vad_filter, vad_parameters=vad_parameters,
        hallucination_silence_threshold=2.0,
        condition_on_previous_text=True, word_timestamps=True,
        chunk_length=30, initial_prompt=initial_prompt)
    raw = _split_by_words(list(chunks), max_segment_seconds, pause_threshold)
    raw = _merge_short_fragments(raw)
    raw = _merge_isolated_fragments(raw)
    raw = _remove_repeated_hallucinations(raw)
    if merge_target_mean and merge_target_mean > 0:
        raw = merge_to_target(raw, merge_target_mean, merge_max_gap, merge_hard_max)
    return raw


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
            # 句长超限时放宽切分条件，避免超长句
            max_exceeded = duration >= max_segment_seconds
            if (punctuation_break or
                    (gap >= pause_threshold and _safe_pause_boundary(group, word)) or
                    (max_exceeded and _safe_max_boundary(group, word))):
                _append_word_group(result, group)
                group = []
            group.append(word)
        _append_word_group(result, group)
    return result


def _safe_max_boundary(group: list, next_word) -> bool:
    """max 超限时的宽松切分：仅避免单字 BPE 词中切断，其余均可"""
    previous_text = group[-1].word.strip()
    if len(previous_text) == 1:
        return False
    return True


def _safe_pause_boundary(group: list, next_word) -> bool:
    """仅在句末附近允许停顿切分，避免句中/词中切断"""
    previous_text = group[-1].word.strip()
    # 句末标点已在外层判断，这里只处理无标点的停顿
    # 允许的句末形态：です/ます/た/だ/よ/ね/か/わ/ぞ/な/っ 等，或长度>=4的完整词
    sentence_endings = ("です", "ます", "た", "だ", "よ", "ね", "か", "わ", "ぞ", "ぜ", "な", "っ", "よう", "まい", "でしょう", "ください", "ございます", "ました", "でした")
    if any(previous_text.endswith(suf) for suf in sentence_endings):
        return True
    # 助词 は/が/を/に/へ/で/と/も/の 单独时禁止切（句中）
    if previous_text in {"は", "が", "を", "に", "へ", "で", "と", "も", "の", "から", "まで", "や", "か", "ね", "よ"}:
        return False
    # 单字 BPE 如 ゲ/一 禁止
    if len(previous_text) == 1:
        return False
    # 2-3字词需看整体：仅当组内已含动词/形容词结尾才允许，否则视为句中
    if len(previous_text) >= 4:
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


def _add_nvidia_cuda_to_path() -> None:
    """Add NVIDIA CUDA pip packages' bin directories to PATH/OS environment so
    ctranslate2 can find cublas64_12.dll and cudnn64_9.dll."""
    import os
    import site

    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        cuda_base = Path(site_dir) / "nvidia"
        if not cuda_base.is_dir():
            continue
        for sub in cuda_base.iterdir():
            for folder in ("bin", "lib"):
                target = sub / folder
                if target.is_dir():
                    path_str = str(target)
                    if path_str not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")
                    # On Windows, also set CUDA_PATH-like vars that some libs use
                    if folder == "bin":
                        os.add_dll_directory(path_str)
