from pathlib import Path
from typing import Any


def load_glossary(path: str | Path | None) -> list[dict[str, str]]:
    """Load either {source: translation} or a list of glossary records."""
    if not path:
        return []
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Glossary not found: {file_path}")
    if file_path.suffix.lower() == ".json":
        import json
        data: Any = json.loads(file_path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML glossary requires PyYAML") from exc
        data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if not data:
        return []
    if isinstance(data, dict):
        return [{"source": str(k), "translation": str(v)} for k, v in data.items()]
    if isinstance(data, list):
        result = []
        for item in data:
            if not isinstance(item, dict) or "source" not in item or "translation" not in item:
                raise ValueError("Each glossary item needs source and translation")
            result.append({"source": str(item["source"]), "translation": str(item["translation"]),
                           "note": str(item.get("note", ""))})
        return result
    raise ValueError("Glossary must be a mapping or a list")


def format_glossary(entries: list[dict[str, str]]) -> str:
    if not entries:
        return "（无自定义术语表）"
    return "\n".join(f"- {e['source']} => {e['translation']}" +
                     (f" ({e['note']})" if e.get("note") else "") for e in entries)
