"""Gemma code review."""

import re
from pathlib import Path
from config import _REVIEWABLE_EXTS
from prompts.reviewer import GEMMA_CODE_REVIEWER_SYSTEM
from utils.logging import log_event


def gemma_review_code(cpu_model, path: str, content: str, task: str, args=None) -> str:
    ext = Path(path).suffix.lower()
    if ext not in _REVIEWABLE_EXTS or len(content) > 6000:
        return content
    messages = [
        {"role": "system", "content": GEMMA_CODE_REVIEWER_SYSTEM},
        {"role": "user", "content": f"TASK: {task}\nFILE: {path}\n\nCONTENT:\n{content}"},
    ]
    try:
        if cpu_model and cpu_model.model:
            reviewed = cpu_model.generate(messages, max_tokens=2048, temperature=0.0).strip()
        else:
            return content
        log_event(args, {
            "model_role": "gemma", "purpose": "code_review",
            "model": getattr(cpu_model, "model_id", None),
            "path": path, "task": task, "messages": messages, "raw_response": reviewed,
        })
        if reviewed and len(reviewed) > 20 and reviewed != content:
            reviewed = re.sub(r'^```\w*\n?', '', reviewed)
            reviewed = re.sub(r'\n?```$', '', reviewed).strip()
            return reviewed
    except Exception as e:
        import sys
        print(f"[review error] {e}", file=sys.stderr)
    return content