from pathlib import Path
from ..models import Segment


def transcribe(audio: str | Path, model_name: str = "medium", device: str = "auto",
               compute_type: str = "auto", beam_size: int = 5, vad_filter: bool = True) -> list[Segment]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -e .") from exc
    if device == "auto":
        device = "cuda" if _cuda_available() else "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    chunks, _ = model.transcribe(str(audio), language="ja", beam_size=beam_size,
                                  vad_filter=vad_filter, condition_on_previous_text=True)
    return [Segment(i, float(s.start), float(s.end), s.text.strip())
            for i, s in enumerate(chunks) if s.text.strip()]


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False
