"""Text formatting utilities."""

import select
import sys


def _flush_stdin():
    try:
        while select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.readline()
    except Exception:
        pass


def _redact(text: str, secret: str | None) -> str:
    if not text or not secret:
        return text
    return text.replace(secret, "[REDACTED]")


def _preview(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 12] + "…[truncated]"