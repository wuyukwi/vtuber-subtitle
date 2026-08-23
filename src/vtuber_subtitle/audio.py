import subprocess
from pathlib import Path


def extract_audio(video: str | Path, output: str | Path, start: float | None = None,
                  end: float | None = None) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y"]
    if start is not None:
        command.extend(["-ss", str(start)])
    command.extend(["-i", str(video)])
    if start is not None and end is not None:
        command.extend(["-t", str(end - start)])
    command.extend(["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output_path)])
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"FFmpeg failed: {exc.stderr[-1000:]}") from exc
    return output_path
