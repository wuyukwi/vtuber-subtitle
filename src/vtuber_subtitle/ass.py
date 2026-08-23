from pathlib import Path
from .models import Segment


def ass_time(seconds: float) -> str:
    total_cs = max(0, round(seconds * 100))
    cs = total_cs % 100
    total = total_cs // 100
    sec = total % 60
    minute = (total // 60) % 60
    hour = total // 3600
    return f"{hour}:{minute:02d}:{sec:02d}.{cs:02d}"


def _escape(text: str) -> str:
    return text.replace("{", "\\{").replace("}", "\\}").replace("\r", "").replace("\n", "\\N")


def _template_header(template: Path) -> tuple[str, set[str]]:
    lines = template.read_text(encoding="utf-8-sig").splitlines()
    events_index = next((i for i, line in enumerate(lines) if line.strip().lower() == "[events]"), None)
    if events_index is None:
        raise ValueError("ASS template has no [Events] section")
    header = lines[:events_index + 1]
    styles = set()
    in_styles = False
    for line in header:
        if line.strip().lower() == "[v4+ styles]":
            in_styles = True
        elif line.startswith("[") and in_styles:
            in_styles = False
        elif in_styles and line.startswith("Style:"):
            styles.add(line.split(":", 1)[1].split(",", 1)[0].strip())
    return "\n".join(header) + "\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n", styles


def write_ass(segments: list[Segment], output: str | Path, config: dict | None = None,
              template: str | Path | None = None, japanese_style: str = "Japanese",
              chinese_style: str = "Chinese") -> Path:
    cfg = config or {}
    x, y = cfg.get("play_res_x", 1920), cfg.get("play_res_y", 1080)
    jf, cf = cfg.get("japanese_font", "Noto Sans JP"), cfg.get("chinese_font", "Noto Sans CJK SC")
    js, cs = cfg.get("japanese_size", 44), cfg.get("chinese_size", 42)
    if template:
        header, styles = _template_header(Path(template))
        if japanese_style not in styles or chinese_style not in styles:
            raise ValueError(f"Styles not found in template. Available: {', '.join(sorted(styles))}")
    else:
        header = f'''[Script Info]\nScriptType: v4.00+\nPlayResX: {x}\nPlayResY: {y}\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Japanese,{jf},{js},&H00FFFFFF,&H000000FF,&H00181818,&H90000000,0,0,0,0,100,100,0,0,1,2,1,8,40,40,70,1\nStyle: Chinese,{cf},{cs},&H00A8E8FF,&H000000FF,&H00181818,&H90000000,0,0,0,0,100,100,0,0,1,2,1,2,40,40,45,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n'''
    lines = [header]
    for s in segments:
        if not s.japanese and not s.chinese:
            continue
        lines.append(f"Dialogue: 0,{ass_time(s.start)},{ass_time(s.end)},{japanese_style},,0,0,0,,{_escape(s.japanese)}")
        if s.chinese:
            lines.append(f"Dialogue: 0,{ass_time(s.start)},{ass_time(s.end)},{chinese_style},,0,0,0,,{_escape(s.chinese)}")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path
