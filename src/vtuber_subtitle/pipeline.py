import json
from pathlib import Path
from .audio import extract_audio
from .asr.faster_whisper import transcribe
from .ass import write_ass
from .glossary import load_glossary
from .models import Segment
from .translation.client import TranslationClient


def _read_segments(path: Path) -> list[Segment]:
    return [Segment.from_dict(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def _write_segments(path: Path, segments: list[Segment]) -> None:
    path.write_text(json.dumps([s.to_dict() for s in segments], ensure_ascii=False, indent=2), encoding="utf-8")


def run(video: str, output: str, *, glossary: str | None = None, provider: str = "openai",
        model: str | None = None, base_url: str | None = None, asr_model: str = "medium",
        device: str = "auto", compute_type: str = "auto", batch_size: int = 20,
        temperature: float = 0.2, work_dir: str | None = None, skip_translation: bool = False,
        subtitle_mode: str = "bilingual") -> Path:
    video_path = Path(video).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    work = Path(work_dir) if work_dir else video_path.parent / f".{video_path.stem}.vtuber-subtitle"
    work.mkdir(parents=True, exist_ok=True)
    audio = work / "audio.wav"
    asr_json = work / "segments.json"
    translated_json = work / "translated.json"
    if asr_json.exists():
        segments = _read_segments(asr_json)
        print(f"Using cached transcription: {asr_json}")
    else:
        print("Extracting audio...")
        extract_audio(video_path, audio)
        print(f"Transcribing with faster-whisper ({asr_model})...")
        segments = transcribe(audio, asr_model, device, compute_type)
        _write_segments(asr_json, segments)
    if subtitle_mode not in ("bilingual", "japanese"):
        raise ValueError("subtitle_mode must be bilingual or japanese")
    if skip_translation or subtitle_mode == "japanese":
        translated = segments
    elif translated_json.exists():
        translated = _read_segments(translated_json)
        print(f"Using cached translation: {translated_json}")
    else:
        entries = load_glossary(glossary)
        client = TranslationClient(provider, model, base_url, temperature=temperature)
        translated = []
        for start in range(0, len(segments), batch_size):
            batch = segments[start:start + batch_size]
            print(f"Translating {start + 1}-{start + len(batch)} / {len(segments)}...")
            translated.extend(client.translate(batch, entries))
        _write_segments(translated_json, translated)
    result = write_ass(translated, output)
    print(f"Wrote: {result}")
    return result
