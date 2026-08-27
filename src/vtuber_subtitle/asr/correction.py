"""Post-ASR Japanese correction.

Two layers:
1. Optional video-specific correction map loaded from ``corrections.yaml``
   (searched in the project root and next to this module). High-confidence
   (wrong -> right) substitutions for known mis-recognitions, applied
   longest-wrong-first so partial overlaps don't clobber each other.
2. Generic guards: drop zero-duration / non-Japanese hallucination fragments.

This runs on the Japanese text BEFORE translation, so the translator receives
clean proper nouns (which the glossary then instructs it to keep verbatim).
"""
from __future__ import annotations

import re
from pathlib import Path

from ..models import Segment

_CORRECTIONS_PATHS = [
    Path(__file__).resolve().parent.parent.parent.parent / "corrections.yaml",
    Path(__file__).resolve().parent.parent / "corrections.yaml",
]


def _load_corrections() -> list[tuple[str, str]]:
    for p in _CORRECTIONS_PATHS:
        if p.is_file():
            try:
                import yaml
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or []
                pairs = [(d["wrong"], d["right"]) for d in data if d.get("wrong")]
                # apply longest wrong-string first to avoid partial clashes
                pairs.sort(key=lambda kv: len(kv[0]), reverse=True)
                return pairs
            except Exception:
                return []
    return []


_CORRECTIONS = _load_corrections()


def correct_japanese(text: str) -> str:
    for wrong, right in _CORRECTIONS:
        if wrong in text:
            text = text.replace(wrong, right)
    return text


def correct_segments(segments: list[Segment]) -> list[Segment]:
    out: list[Segment] = []
    for s in segments:
        # filter zero-duration hallucination
        if s.end - s.start < 0.05:
            continue
        # filter obvious non-Japanese hallucination leftovers
        if "Korean abroad" in s.japanese:
            continue
        jp = correct_japanese(s.japanese)
        out.append(Segment(s.id, s.start, s.end, jp, s.chinese))
    return out
