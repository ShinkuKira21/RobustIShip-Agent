"""Event logging for model calls."""

import json
import time
from pathlib import Path


def log_event(args, event: dict) -> None:
    if not getattr(args, "log_output", False):
        return
    try:
        root = Path(getattr(args, "root", ".")).resolve()
        log_dir = root / ".robustIship" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **event}
        with open(log_dir / "model_calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        if getattr(args, "debug", False):
            print(f"[debug] log_event failed: {e}", file=__import__("sys").stderr)