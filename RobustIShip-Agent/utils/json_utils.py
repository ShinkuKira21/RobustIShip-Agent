"""JSON extraction and repair utilities."""

import json
import re


def _extract_json_object(text: str) -> str | None:
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r'^```(?:json)?\s*\n?', '', s)
        s = re.sub(r'\n?```\s*$', '', s)
        s = s.strip()
    
    # Pre-parse repair for common LLM JSON mishaps
    # 1. Fix Qwen's invalid \' escapes (single backslash before quote in JSON)
    s = re.sub(r"(?<!\\)\\'", "'", s)
    # 2. Fix unescaped backslashes before characters that don't need escaping in JSON (common in regex)
    # This specifically targets patterns like \s, \w, \d, \b, \. which LLMs often fail to double-escape
    s = re.sub(r'(?<!\\)\\(?=[^"\\\/bfnrtu])', r'\\\\', s)

    try:
        json.loads(s)
        return s
    except Exception:
        pass
    
    starts = [i for i, ch in enumerate(s) if ch == "{"]
    for start in starts:
        for end in range(len(s) - 1, start, -1):
            if s[end] == "}":
                candidate = s[start: end + 1]
                # Apply same repairs to candidate
                candidate = re.sub(r"(?<!\\)\\'", "'", candidate)
                candidate = re.sub(r'(?<!\\)\\(?=[^"\\\/bfnrtu])', r'\\\\', candidate)
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    continue
    return None