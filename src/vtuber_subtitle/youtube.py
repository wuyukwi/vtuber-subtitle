"""YouTube video audio extraction using yt-dlp."""

import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse


def is_youtube_url(url: str) -> bool:
    """Check if the input is a YouTube URL."""
    if not url:
        return False
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        netloc = parsed.netloc.lower()
        # Handle www.youtube.com, youtube.com, youtu.be, m.youtube.com, etc.
        if "youtube.com" in netloc or netloc == "youtu.be":
            return True
        return False
    except Exception:
        return False


def extract_audio_from_youtube(url: str, output_dir: Path, log=None) -> Path:
    """Download audio from YouTube URL and return the path to the audio file.

    Uses yt-dlp to download the best audio and convert to WAV format suitable for Whisper.
    """
    if log is None:
        log = print

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output template: use video ID or title
    output_template = str(output_dir / "%(id)s.%(ext)s")
    wav_path = output_dir / "audio.wav"

    log(f"Downloading audio from YouTube: {url}")

    # Download audio using yt-dlp
    # -x: extract audio
    # --audio-format wav: convert to wav
    # --audio-quality 0: best quality
    # -o: output template
    # --no-playlist: don't download playlist
    # --no-check-certificates: skip SSL verification (some regions need this)
    command = [
        "yt-dlp",
        "-x",  # extract audio
        "--audio-format", "wav",
        "--audio-quality", "0",  # best quality
        "-o", str(output_template),
        "--no-playlist",
        "--no-check-certificates",
        "--progress",
        url,
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        log("YouTube audio download completed")
    except FileNotFoundError as exc:
        raise RuntimeError("yt-dlp is not installed. Install it with: pip install yt-dlp") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr[-1000:] if exc.stderr else "No error details"
        raise RuntimeError(f"yt-dlp failed: {stderr}") from exc

    # Find the downloaded wav file
    wav_files = list(output_dir.glob("*.wav"))
    if not wav_files:
        raise RuntimeError("No audio file was downloaded")

    # If there are multiple wav files, pick the most recent one
    downloaded_wav = max(wav_files, key=lambda p: p.stat().st_mtime)

    # Rename to a standard name for consistency (handle existing file)
    if downloaded_wav != wav_path:
        if wav_path.exists():
            wav_path.unlink()
        downloaded_wav.rename(wav_path)
        downloaded_wav = wav_path

    log(f"Audio saved to: {downloaded_wav}")
    return downloaded_wav


def get_youtube_video_id(url: str) -> str | None:
    """Extract video ID from YouTube URL."""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()

        # Handle youtu.be/VIDEO_ID
        if netloc == "youtu.be":
            return parsed.path.lstrip("/")

        # Handle youtube.com/watch?v=VIDEO_ID
        if "youtube.com" in netloc:
            query = parsed.query
            if "v=" in query:
                # Extract v parameter
                for part in query.split("&"):
                    if part.startswith("v="):
                        return part[2:]

            # Handle youtube.com/shorts/VIDEO_ID or /embed/VIDEO_ID
            path_parts = parsed.path.strip("/").split("/")
            if len(path_parts) >= 2 and path_parts[0] in ("shorts", "embed", "v"):
                return path_parts[1]

        return None
    except Exception:
        return None
