"""Gemma takeover strategy — quality score ≤ takeover threshold, Gemma writes directly."""

import json

from tools.files import write_file
from tools.validate import validate_written_file
from tools.normalize import _fix_absolute_path


def parse_gemma_raw(raw: str) -> tuple[str, str] | None:
    if not raw:
        return None
    import re
    lines = raw.strip().split('\n')
    if not lines:
        return None
    path = None
    content_start = 0
    if lines[0].startswith("PATH:"):
        path = lines[0][5:].strip()
        content_start = 1
        if content_start < len(lines) and lines[content_start].strip() == "":
            content_start += 1
    if not path:
        first_line = lines[0].strip()
        if first_line.endswith(".py") and "/" not in first_line and " " not in first_line:
            path = first_line
            content_start = 1
    if not path:
        return None
    content = '\n'.join(lines[content_start:]).strip()
    if not content:
        return None
    content = re.sub(r'^```\w*\n?', '', content)
    content = re.sub(r'\n?```$', '', content)
    return path, content


def handle(gemma_model, args, root, state, flags, step_index,
           step_prompt, expected_tool, injected_context, observations, path) -> dict:
    print("   🧠 Quality critically low — Gemma taking over...")

    reflection_context = state.assemble_reflection_context(step_index, "", current_tool=expected_tool)
    prompt = (
        f"Qwen has failed to produce acceptable code for this task. Write the COMPLETE correct file.\n\n"
        f"TASK: {step_prompt}\n\n"
        f"CONTEXT:\n{json.dumps(reflection_context, indent=2, default=str)[:2000]}\n\n"
        f"Output:\nPATH: <relative file path>\n\n<complete file content>"
    )
    messages = [
        {"role": "system", "content": "You are taking over from a weaker model. Write flawless, complete code."},
        {"role": "user", "content": prompt},
    ]

    try:
        raw = gemma_model.generate(messages, max_tokens=2048, temperature=0.0)
        parsed = parse_gemma_raw(raw)
        if parsed:
            final_path, final_content = parsed
            write_file(root, final_path, final_content)
            state.store_file_content(final_path, final_content)
            validation = validate_written_file(root, _fix_absolute_path(final_path, root), pre_flight=flags.pre_flight)
            if validation.get("ok"):
                state.mark_done_by_index(step_index)
                return {"ok": True, "summary": f"Gemma takeover: {final_path}"}
            return {"ok": False, "summary": f"Gemma takeover failed validation: {validation.get('stderr', '')}"}
    except Exception as e:
        print(f"   ❌ Gemma takeover error: {e}")

    return {"ok": False, "summary": "Gemma takeover failed"}