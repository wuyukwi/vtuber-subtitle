import json
import os
import time
import urllib.error
import urllib.request
from typing import Any
from ..glossary import format_glossary
from ..models import Segment


SYSTEM_PROMPT = """你是专业的日语 VTuber 字幕翻译员。将日文翻译成自然、简洁的简体中文。
保留语气和网络用语的风格，严格遵守术语表。人名、VTuber 名称、团体名、社团名、作品名等专有名词，
如果术语表没有明确译法，则保留日文原文，不要自行音译、意译或编造译名；术语表有明确译法时才使用该译法。
只输出 JSON 数组，数组元素必须是
{\"id\": number, \"translation\": string}，不要输出 Markdown、解释或其它文字。"""


class TranslationClient:
    def __init__(self, provider: str, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.2, retries: int = 3):
        self.provider = provider.lower()
        default_model = (os.getenv("GEMINI_MODEL", "gemini-2.0-flash") if self.provider == "gemini"
                         else ("deepseek-v4-flash" if self.provider in ("opencode", "opencode-go")
                               else ("deepseek-chat" if self.provider == "deepseek" else "gpt-4o-mini")))
        default_url = ("https://opencode.ai/zen/go/v1" if self.provider in ("opencode", "opencode-go")
                       else ("https://api.deepseek.com" if self.provider == "deepseek" else "https://api.openai.com/v1"))
        self.model = model or os.getenv("OPENCODE_MODEL" if self.provider in ("opencode", "opencode-go") else "LLM_MODEL", default_model)
        self.base_url = (base_url or os.getenv("OPENCODE_BASE_URL" if self.provider in ("opencode", "opencode-go") else "LLM_BASE_URL", default_url)).rstrip("/")
        self.api_key = api_key or (os.getenv("GEMINI_API_KEY") if self.provider == "gemini"
                                   else (os.getenv("OPENCODE_API_KEY") if self.provider in ("opencode", "opencode-go")
                                         else os.getenv("LLM_API_KEY")))
        self.temperature = temperature
        self.retries = retries
        if not self.api_key:
            variable = ("GEMINI_API_KEY" if self.provider == "gemini" else
                        ("OPENCODE_API_KEY" if self.provider in ("opencode", "opencode-go") else "LLM_API_KEY"))
            raise ValueError(f"Missing API key. Set {variable} in the environment.")

    def translate(self, segments: list[Segment], glossary: list[dict[str, str]]) -> list[Segment]:
        if not segments:
            return segments
        payload = [{"id": s.id, "text": s.japanese} for s in segments]
        user = "术语表：\n" + format_glossary(glossary) + "\n\n待翻译片段：\n" + json.dumps(payload, ensure_ascii=False)
        result = self._request(user)
        by_id = {int(item["id"]): str(item["translation"]).strip() for item in result}
        missing = [s.id for s in segments if s.id not in by_id]
        if missing:
            raise ValueError(f"Translation response missing segment ids: {missing}")
        return [Segment(s.id, s.start, s.end, s.japanese, by_id[s.id]) for s in segments]

    def _request(self, user: str) -> list[dict[str, Any]]:
        if self.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            body = {"contents": [{"parts": [{"text": SYSTEM_PROMPT + "\n" + user}]}],
                    "generationConfig": {"temperature": self.temperature, "responseMimeType": "application/json"}}
        elif self.provider in ("opencode", "opencode-go") and self.model == "gpt-5.6-luna":
            url = self.base_url + "/responses"
            body = {"model": self.model, "instructions": SYSTEM_PROMPT,
                    "input": user}
        else:
            url = self.base_url + "/chat/completions"
            body = {"model": self.model, "temperature": self.temperature,
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]}
        request = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode(),
                                          headers={"Content-Type": "application/json",
                                                   "User-Agent": "vtuber-subtitle/0.1",
                                                   **({} if self.provider == "gemini" else {"Authorization": f"Bearer {self.api_key}"})})
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = json.loads(response.read().decode("utf-8"))
                if self.provider == "gemini":
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                elif self.provider in ("opencode", "opencode-go") and self.model == "gpt-5.6-luna":
                    text = data.get("output_text") or next(
                        block["text"]
                        for item in data.get("output", [])
                        for block in item.get("content", [])
                        if block.get("type") == "output_text" and block.get("text")
                    )
                else:
                    text = data["choices"][0]["message"]["content"]
                parsed = json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
                if not isinstance(parsed, list):
                    raise ValueError("LLM response is not a JSON array")
                return parsed
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
                if attempt == self.retries - 1:
                    raise RuntimeError(f"Translation request failed: {exc}") from exc
                time.sleep(2 ** attempt)
        raise RuntimeError("Translation request failed")
