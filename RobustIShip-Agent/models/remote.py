"""Remote model client (Qwen via HTTP API)."""

import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from utils.text_utils import _preview, _redact


def http_post_json(url: str, payload: dict, api_key: str | None, *, debug: bool = False, debug_max_chars: int = 4000) -> dict:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(url, data=body, headers=headers, method="POST")
    t0 = time.perf_counter()
    if debug:
        msg_count = len(payload.get("messages", [])) if isinstance(payload.get("messages"), list) else None
        print(f"[debug] HTTP POST messages={msg_count}, body_bytes={len(body)}", file=__import__("sys").stderr)
    try:
        with urlopen(req, timeout=600) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if debug:
                dt_ms = int((time.perf_counter() - t0) * 1000)
                print(f"[debug] HTTP {resp.getcode()} in {dt_ms}ms", file=__import__("sys").stderr)
                print(f"[debug] Response: {_preview(_redact(raw, api_key), debug_max_chars)!r}", file=__import__("sys").stderr)
            return json.loads(raw)
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {raw}") from e
    except URLError as e:
        raise RuntimeError(f"Failed to reach server: {e}") from e


def chat_server(base_url: str, model: str, messages: list, api_key: str | None, *,
                temperature: float = 0.3, top_p: float = 0.95, max_tokens: int = 1500,
                debug: bool = False) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model, "messages": messages,
        "temperature": float(temperature), "top_p": float(top_p),
        "max_tokens": int(max_tokens), "stream": False,
    }
    data = http_post_json(url, payload, api_key=api_key, debug=debug)
    try:
        return str(data["choices"][0]["message"]["content"])
    except Exception:
        raise RuntimeError(f"Unexpected response: {data}")