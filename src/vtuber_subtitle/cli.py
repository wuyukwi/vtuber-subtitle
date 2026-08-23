import argparse
import sys
from .env import load_dotenv
from .pipeline import run


def main() -> None:
    load_dotenv()
    # Keep Windows consoles from failing when paths contain Japanese characters.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Convert a Japanese VTuber recording to bilingual ASS subtitles")
    parser.add_argument("video", help="Input video/audio file")
    parser.add_argument("-o", "--output", required=True, help="Output .ass path")
    parser.add_argument("--glossary", help="Glossary .yaml/.yml/.json")
    parser.add_argument("--provider", choices=["openai", "deepseek", "gemini", "opencode-go"], default="openai")
    parser.add_argument("--model", help="LLM model name")
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL")
    parser.add_argument("--asr-model", default="large-v3", help="Whisper model, e.g. medium, large-v3")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--compute-type", default="auto", help="Whisper compute type")
    parser.add_argument("--enable-vad", action="store_true",
                        help="Enable VAD for noisy audio; disabled by default to avoid losing short speech")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--work-dir", help="Cache directory")
    parser.add_argument("--subtitle-mode", choices=["bilingual", "japanese"], default="bilingual",
                        help="Subtitle type: bilingual (default) or japanese")
    parser.add_argument("--skip-translation", action="store_true", help="Export Japanese-only ASS")
    args = parser.parse_args()
    try:
        run(args.video, args.output, glossary=args.glossary, provider=args.provider,
            model=args.model, base_url=args.base_url, asr_model=args.asr_model,
            device=args.device, compute_type=args.compute_type, batch_size=args.batch_size,
            temperature=args.temperature, work_dir=args.work_dir, skip_translation=args.skip_translation,
            subtitle_mode=args.subtitle_mode, vad_filter=args.enable_vad)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
