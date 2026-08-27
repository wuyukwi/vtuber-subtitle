import json
from pathlib import Path
from typing import Callable
from .audio import extract_audio, get_duration
from .asr.faster_whisper import transcribe
from .ass import write_ass
from .glossary import load_glossary
from .models import Segment
from .translation.client import TranslationClient
from .youtube import is_youtube_url, extract_audio_from_youtube


def parse_time(value: str | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    parts = value.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Invalid time: {value}")


def _read_segments(path: Path) -> list[Segment]:
    return [Segment.from_dict(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def _write_segments(path: Path, segments: list[Segment]) -> None:
    path.write_text(json.dumps([s.to_dict() for s in segments], ensure_ascii=False, indent=2), encoding="utf-8")


def _format_time(seconds: float) -> str:
    return f"{seconds:g}"


def _build_windows(start: float, end: float, window_minutes: int) -> list[tuple[float, float]]:
    if window_minutes and window_minutes > 0:
        windows: list[tuple[float, float]] = []
        cursor = start
        step = window_minutes * 60
        while cursor < end:
            windows.append((cursor, min(cursor + step, end)))
            cursor += step
        return windows
    return [(start, end)]


def _merge_windowed_segments(segments: list[Segment]) -> list[Segment]:
    """Drop window-boundary duplicates, keeping the more complete copy."""
    if not segments:
        return segments
    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    merged: list[Segment] = []
    for segment in ordered:
        if not merged:
            merged.append(segment)
            continue
        last = merged[-1]
        if last.end - segment.start > 0.5:
            if (segment.end - segment.start) > (last.end - last.start):
                merged[-1] = segment
        else:
            merged.append(segment)
    return merged


def run(video: str, output: str | None, *, glossary: str | None = None, provider: str = "openai",
        model: str | None = None, base_url: str | None = None, asr_model: str = "large-v3",
        device: str = "auto", compute_type: str = "auto", batch_size: int = 20,
        temperature: float = 0.2, work_dir: str | None = None, skip_translation: bool = False,
        subtitle_mode: str = "bilingual", vad_filter: bool = True,
        start_time: str | float | None = None, end_time: str | float | None = None,
        template: str | None = None, japanese_style: str = "Japanese",
        chinese_style: str = "Chinese", max_segment_seconds: float = 10.0,
        pause_threshold: float = 0.6, window_minutes: int = 15, window_overlap: float = 3.0,
        beam_size: int = 10, merge_target_mean: float = 3.6, merge_max_gap: float = 1.0,
        merge_hard_max: float = 10.0,
        log: Callable[[str], None] = print) -> Path:
    # Check if input is a YouTube URL
    is_youtube = is_youtube_url(video)
    if is_youtube:
        log("Detected YouTube URL, extracting audio...")
        # Create a temporary directory for YouTube downloads
        temp_dir = Path(work_dir) if work_dir else Path("youtube_downloads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        # Extract audio from YouTube
        audio_path = extract_audio_from_youtube(video, temp_dir, log=log)
        video_path = audio_path
        # Use audio filename for work directory name
        work = Path(work_dir) if work_dir else audio_path.parent / f".{audio_path.stem}.vtuber-subtitle"
        work.mkdir(parents=True, exist_ok=True)
        # Auto-generate output path if not provided
        if not output:
            output = str(audio_path.with_suffix(".ass"))
    else:
        # Local file path
        video_path = Path(video).resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
        work = Path(work_dir) if work_dir else video_path.parent / f".{video_path.stem}.vtuber-subtitle"
        work.mkdir(parents=True, exist_ok=True)
        # Auto-generate output path if not provided
        if not output:
            output = str(video_path.with_suffix(".ass"))
    start = parse_time(start_time) or 0.0
    end = parse_time(end_time)
    if end is not None and end <= start:
        raise ValueError("end_time must be greater than start_time")

    if subtitle_mode not in ("bilingual", "japanese", "chinese"):
        raise ValueError("subtitle_mode must be bilingual, japanese or chinese")

    if end is None:
        log("Reading video duration...")
        end = get_duration(video_path)

    windows = _build_windows(start, end, window_minutes)

    # 通用：从术语表构建 initial_prompt，提升固有名词召回（非硬编码）
    initial_prompt = None
    if glossary:
        try:
            _entries_for_prompt = load_glossary(glossary)
            if _entries_for_prompt:
                # 取前 30 条，避免 prompt 过长
                initial_prompt = " ".join(e["source"] for e in _entries_for_prompt[:30])
        except Exception:
            pass

    segments: list[Segment] = []
    for i, (w_start, w_end) in enumerate(windows):
        extract_start = max(w_start - (window_overlap if i > 0 else 0.0), start)
        extract_end = min(w_end + window_overlap, end)
        tag = f"{_format_time(extract_start)}_{_format_time(extract_end)}"
        audio = work / f"audio_{tag}.wav"
        asr_json = work / f"segments_{tag}_v17.json"
        if asr_json.exists():
            segs = _read_segments(asr_json)
            try:
                from .asr.correction import correct_segments
                segs = correct_segments(segs)
            except Exception:
                pass
            log(f"Using cached transcription: {asr_json}")
        else:
            if audio.exists():
                log(f"Using cached audio: {audio}")
            else:
                log(f"Extracting audio {_format_time(extract_start)}s-{_format_time(extract_end)}s...")
                extract_audio(video_path, audio, extract_start, extract_end)
            log(f"Transcribing {_format_time(extract_start)}s-{_format_time(extract_end)}s ({asr_model})...")
            segs = transcribe(audio, asr_model, device, compute_type, vad_filter=vad_filter,
                              max_segment_seconds=max_segment_seconds,
                              pause_threshold=pause_threshold,
                              initial_prompt=initial_prompt,
                              beam_size=beam_size,
                              merge_target_mean=merge_target_mean,
                              merge_max_gap=merge_max_gap,
                              merge_hard_max=merge_hard_max)
            # 应用日文固有名词校正（基于 glossary 的 ASR 纠错）
            try:
                from .asr.correction import correct_segments
                segs = correct_segments(segs)
            except Exception:
                pass
            if extract_start:
                segs = [Segment(s.id, s.start + extract_start, s.end + extract_start,
                                s.japanese, s.chinese) for s in segs]
            _write_segments(asr_json, segs)
        segments.extend(segs)
    segments = _merge_windowed_segments(segments)
    segments = [Segment(i, s.start, s.end, s.japanese, s.chinese) for i, s in enumerate(segments)]

    asr_json = work / "segments_v17.json"
    _write_segments(asr_json, segments)

    translated_json = work / "translated_v17.json"
    if skip_translation:
        translated = segments
    elif subtitle_mode == "japanese":
        translated = segments
    elif translated_json.exists():
        translated = _read_segments(translated_json)
        log(f"Using cached translation: {translated_json}")
    else:
        entries = load_glossary(glossary)
        client = TranslationClient(provider, model, base_url, temperature=temperature)
        translated = []
        for batch_start in range(0, len(segments), batch_size):
            batch = segments[batch_start:batch_start + batch_size]
            log(f"Translating {batch_start + 1}-{batch_start + len(batch)} / {len(segments)}...")
            translated.extend(client.translate(batch, entries))
        _write_segments(translated_json, translated)
    # 中文单语：只保留中文，清空日文，避免 write_ass 输出日文行
    if subtitle_mode == "chinese":
        translated = [Segment(s.id, s.start, s.end, "", s.chinese) for s in translated]
    result = write_ass(translated, output, template=template,
                       japanese_style=japanese_style, chinese_style=chinese_style)
    log(f"Wrote: {result}")
    return result
