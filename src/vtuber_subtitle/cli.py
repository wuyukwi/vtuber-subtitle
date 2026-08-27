import argparse
import sys
from .env import ensure_environment
from .pipeline import run


def main() -> None:
    ensure_environment()
    # Keep Windows consoles from failing when paths contain Japanese characters.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Convert a Japanese VTuber recording to bilingual ASS subtitles")
    parser.add_argument("video", help="Input video/audio file path or YouTube URL")
    parser.add_argument("-o", "--output", help="Output .ass path (auto-generated if not specified)")
    parser.add_argument("--glossary", help="Glossary .yaml/.yml/.json")
    parser.add_argument("--provider", choices=["openai", "deepseek", "gemini", "opencode-go"], default="openai")
    parser.add_argument("--model", help="LLM model name")
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL")
    parser.add_argument("--asr-model", default="large-v3", help="Whisper model, e.g. medium, large-v3")
    parser.add_argument("--beam-size", type=int, default=10, help="Beam size for Whisper decoding (default: 10)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--compute-type", default="auto", help="Whisper compute type")
    parser.add_argument("--enable-vad", action="store_true", dest="enable_vad", default=True,
                        help="Enable VAD (default; tuned to preserve short speech)")
    parser.add_argument("--no-vad", action="store_false", dest="enable_vad",
                        help="Disable VAD; may cause repeated hallucinated subtitles during silence")
    parser.add_argument("--max-segment-seconds", type=float, default=10.0,
                        help="Maximum subtitle duration before splitting (default: 15)")
    parser.add_argument("--pause-threshold", type=float, default=0.6,
                        help="Split when the word-level pause reaches this many seconds")
    parser.add_argument("--merge-target-mean", type=float, default=3.6,
                        help="Target average subtitle duration (s); short adjacent clauses are "
                             "merged toward this. 0 disables the readability merge (default: 3.6)")
    parser.add_argument("--merge-max-gap", type=float, default=1.0,
                        help="Max inter-segment gap (s) allowed when merging (default: 1.0)")
    parser.add_argument("--merge-hard-max", type=float, default=10.0,
                        help="Max merged subtitle duration (s) (default: 10.0)")
    parser.add_argument("--window-minutes", type=int, default=15,
                        help="Process long videos in windows of this many minutes to save memory")
    parser.add_argument("--start-time", help="Only process from this source-video time, e.g. 00:20:00")
    parser.add_argument("--end-time", help="Only process until this source-video time, e.g. 00:25:00")
    parser.add_argument("--ass-template", help="Use styles and header from an existing ASS file")
    parser.add_argument("--japanese-style", default="Japanese", help="Japanese style name in ASS template")
    parser.add_argument("--chinese-style", default="Chinese", help="Chinese style name in ASS template")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--work-dir", help="Cache directory")
    parser.add_argument("--subtitle-mode", choices=["bilingual", "japanese", "chinese"], default="bilingual",
                        help="Subtitle type: bilingual (default), japanese or chinese (中文单语)")
    parser.add_argument("--skip-translation", action="store_true", help="Export Japanese-only ASS")
    args = parser.parse_args()
    
    # Auto-generate output path if not provided
    if not args.output:
        from .youtube import is_youtube_url
        if is_youtube_url(args.video):
            # For YouTube URLs, output will be generated in pipeline
            args.output = ""
        else:
            # For local files, generate output path based on input
            from pathlib import Path
            video_path = Path(args.video).resolve()
            args.output = str(video_path.with_suffix(".ass"))
    
    try:
        run(args.video, args.output, glossary=args.glossary, provider=args.provider,
            model=args.model, base_url=args.base_url, asr_model=args.asr_model,
            device=args.device, compute_type=args.compute_type, batch_size=args.batch_size,
            temperature=args.temperature, work_dir=args.work_dir, skip_translation=args.skip_translation,
            subtitle_mode=args.subtitle_mode, vad_filter=args.enable_vad,
            beam_size=args.beam_size, merge_target_mean=args.merge_target_mean,
            merge_max_gap=args.merge_max_gap, merge_hard_max=args.merge_hard_max,
            start_time=args.start_time, end_time=args.end_time,
            template=args.ass_template, japanese_style=args.japanese_style,
            chinese_style=args.chinese_style, max_segment_seconds=args.max_segment_seconds,
            pause_threshold=args.pause_threshold, window_minutes=args.window_minutes)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
