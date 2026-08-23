import os
from pathlib import Path

# 当默认 C 盘缓存空间不足时，可把 Whisper 模型缓存放到这些目录（按顺序取第一个存在的）。
HF_CACHE_CANDIDATES = ("D:\\hf-cache", "E:\\hf-cache")

# 本地安装的 CUDA 运行库（cuBLAS / cuDNN / NVRTC）可能所在的目录，启动时自动加入 PATH。
CUDA_BIN_CANDIDATES = (
    "D:\\vtuber-cuda\\nvidia\\cudnn\\bin",
    "D:\\vtuber-cuda\\nvidia\\cublas\\bin",
    "D:\\vtuber-cuda\\nvidia\\cuda_nvrtc\\bin",
)


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


def _prepend_to_path(candidates: tuple[str, ...]) -> None:
    bins = [c for c in candidates if Path(c).is_dir()]
    if bins:
        os.environ["PATH"] = os.pathsep.join(bins) + os.pathsep + os.environ.get("PATH", "")


def ensure_environment() -> None:
    """Load .env, point HF_HOME somewhere with space, and expose CUDA libraries."""
    load_dotenv()
    _prepend_to_path(CUDA_BIN_CANDIDATES)
    if os.environ.get("HF_HOME"):
        return
    for candidate in HF_CACHE_CANDIDATES:
        if Path(candidate).is_dir():
            os.environ["HF_HOME"] = candidate
            break
