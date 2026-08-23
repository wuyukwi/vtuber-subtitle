import os
from pathlib import Path

# 当默认 C 盘缓存空间不足时，可把 Whisper 模型缓存放到这些目录（按顺序取第一个存在的）。
HF_CACHE_CANDIDATES = ("D:\\hf-cache", "E:\\hf-cache")


def load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs without requiring an extra dotenv package."""
    file_path = Path(path)
    if not file_path.is_file():
        return
    for raw_line in file_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def ensure_environment() -> None:
    """Load .env and make sure HF_HOME points somewhere with space for models."""
    load_dotenv()
    if os.environ.get("HF_HOME"):
        return
    for candidate in HF_CACHE_CANDIDATES:
        if Path(candidate).is_dir():
            os.environ["HF_HOME"] = candidate
            break
