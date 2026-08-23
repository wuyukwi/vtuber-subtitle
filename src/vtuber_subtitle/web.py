import argparse
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from .env import load_dotenv
from .pipeline import run

INDEX_FILE = Path(__file__).parent / "index.html"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "uploads"


def _safe_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name)
    return name or "upload.bin"


def _default_browse_path() -> str:
    for drive in ("D:\\", "E:\\", "C:\\Users"):
        if Path(drive).exists():
            return drive
    return str(Path.home())


def _list_dir(path_str: str) -> dict:
    path = Path(path_str).expanduser()
    if not path.exists():
        path = Path(_default_browse_path())
    if not path.is_dir():
        path = path.parent
    try:
        entries = sorted(
            ({"name": child.name, "is_dir": child.is_dir()} for child in path.iterdir()),
            key=lambda e: (not e["is_dir"], e["name"].lower()),
        )
    except OSError as exc:
        entries = []
        path = Path(_default_browse_path())
    return {
        "path": str(path.resolve()),
        "parent": str(path.parent) if path.parent != path else str(path),
        "entries": entries,
        "is_dir": True,
    }


class _Job:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.done = False
        self.ok = False
        self.error = ""
        self.output = ""
        self.logs: list[str] = []

    def start(self, config: dict) -> None:
        self.logs = []
        self.running = True
        self.done = False
        self.ok = False
        self.error = ""
        self.output = ""
        thread = threading.Thread(target=self._work, args=(config,), daemon=True)
        thread.start()

    def _work(self, config: dict) -> None:
        load_dotenv()

        def log(line: str) -> None:
            with self.lock:
                self.logs.append(line)
                if len(self.logs) > 800:
                    self.logs = self.logs[-800:]

        try:
            log("=== 任务开始 ===")
            output = run(
                config["video"],
                config["output"],
                glossary=config.get("glossary") or None,
                provider=config.get("provider") or "deepseek",
                model=config.get("model") or None,
                base_url=config.get("base_url") or None,
                asr_model=config.get("asr_model") or "large-v3",
                device=config.get("device") or "auto",
                compute_type=config.get("compute_type") or "auto",
                batch_size=int(config.get("batch_size") or 20),
                temperature=float(config.get("temperature") or 0.2),
                work_dir=config.get("work_dir") or None,
                subtitle_mode=config.get("subtitle_mode") or "bilingual",
                vad_filter=bool(config.get("vad_filter", True)),
                start_time=config.get("start_time") or None,
                end_time=config.get("end_time") or None,
                template=config.get("template") or None,
                japanese_style=config.get("japanese_style") or "Japanese",
                chinese_style=config.get("chinese_style") or "Chinese",
                log=log,
            )
            with self.lock:
                self.ok = True
                self.output = str(output)
            log(f"完成: {output}")
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.ok = False
                self.error = str(exc)
            log(f"错误: {exc}")
        finally:
            with self.lock:
                self.running = False
                self.done = True

    def status(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "done": self.done,
                "ok": self.ok,
                "error": self.error,
                "output": self.output,
                "logs": list(self.logs),
            }


JOB = _Job()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence request logs
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            try:
                body = INDEX_FILE.read_bytes()
            except FileNotFoundError:
                body = b"<h1>index.html missing</h1>"
            self._send(200, body, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/status":
            self._send(200, json.dumps(JOB.status(), ensure_ascii=False).encode(), "application/json; charset=utf-8")
            return
        if parsed.path == "/api/browse":
            query = parse_qs(parsed.query)
            target = query.get("path", [""])[0]
            if not target:
                target = _default_browse_path()
            self._send(200, json.dumps(_list_dir(target), ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")
            return
        if parsed.path == "/api/download":
            query = parse_qs(parsed.query)
            target = query.get("path", [""])[0]
            if target:
                path = Path(target)
                if path.is_file():
                    body = path.read_bytes()
                    filename = path.name
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header(
                        "Content-Disposition",
                        f"attachment; filename*=UTF-8''{quote(filename)}",
                    )
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
            self._send(404, b"file not found", "text/plain; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/run":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                config = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._send(400, b"invalid json", "text/plain; charset=utf-8")
                return
            if JOB.running:
                self._send(409, b"job already running", "text/plain; charset=utf-8")
                return
            JOB.start(config)
            self._send(200, b'{"started": true}', "application/json; charset=utf-8")
            return
        if parsed.path == "/api/upload":
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                self._send(400, b"empty upload", "text/plain; charset=utf-8")
                return
            raw = self.rfile.read(length)
            filename = parse_qs(parsed.query).get("name", [""])[0]
            safe = _safe_filename(filename)
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            target = UPLOAD_DIR / f"{int(time.time())}_{safe}"
            target.write_bytes(raw)
            self._send(200, json.dumps({"path": str(target.resolve())}, ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="VTuber Subtitle web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"VTuber Subtitle Web UI: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
