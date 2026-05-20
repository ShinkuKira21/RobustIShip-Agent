"""Dual generation strategy — Qwen vs Gemma parallel generation."""

import concurrent.futures
import json

from tools.files import write_file, read_file
from tools.validate import validate_written_file
from tools.normalize import _fix_absolute_path
from gemma.quality import gemma_quality_check
from qwen.actions import request_qwen_action


def parse_gemma_raw(raw: str) -> tuple[str, str] | None:
    """Parse Gemma's raw code output into (path, content)."""
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
           step_prompt, expected_tool, injected_context, observations,
           path, content) -> dict:
    print("   ⚔️  Qwen (GPU) vs Gemma (CPU) — generating simultaneously...")

    def qwen_gen():
        return request_qwen_action(
            args, root, expected_tool, step_prompt,
            injected_context=injected_context, observations=observations,
            step_index=step_index, purpose="dual_qwen",
        )

    def gemma_gen():
        reflection_context = state.assemble_reflection_context(step_index, "", current_tool=expected_tool)
        prompt = (
            f"You are a code generator. Write the COMPLETE file for this task.\n\n"
            f"TASK: {step_prompt}\n\n"
            f"CONTEXT:\n{json.dumps(reflection_context, indent=2, default=str)[:2000]}\n\n"
            f"Output format:\nPATH: <relative file path>\n\n<complete file content>\n\n"
            f"No explanation. No markdown fences."
        )
        messages = [
            {"role": "system", "content": "You are an expert developer. Write complete, correct, tested code."},
            {"role": "user", "content": prompt},
        ]
        try:
            return gemma_model.generate(messages, max_tokens=2048, temperature=0.0)
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        qwen_future = executor.submit(qwen_gen)
        gemma_future = executor.submit(gemma_gen)
        try:
            qwen_action = qwen_future.result(timeout=flags.dual_gen_timeout_qwen)
        except concurrent.futures.TimeoutError:
            qwen_action = None
        try:
            gemma_raw = gemma_future.result(timeout=flags.dual_gen_timeout_gemma)
        except (concurrent.futures.TimeoutError, Exception):
            gemma_raw = None

    gemma_path, gemma_content = None, None
    if gemma_raw:
        parsed = parse_gemma_raw(gemma_raw)
        if parsed:
            gemma_path, gemma_content = parsed

    # Score both
    qwen_score = 0
    gemma_score = 0

    if qwen_action and "final" not in qwen_action:
        qwen_path = qwen_action.get("args", {}).get("path", "")
        qwen_content_val = qwen_action.get("args", {}).get("content", "")
        qwen_quality = gemma_quality_check(gemma_model, step_prompt=step_prompt, path=qwen_path, content=qwen_content_val, args=args)
        qwen_score = qwen_quality.get("score", 0)

    if gemma_path and gemma_content:
        gemma_quality = gemma_quality_check(gemma_model, step_prompt=step_prompt, path=gemma_path, content=gemma_content, args=args)
        gemma_score = gemma_quality.get("score", 0)

    # Pick winner
    if gemma_score > qwen_score and gemma_path and gemma_content:
        print(f"   🏆 Gemma wins! (Gemma: {gemma_score}/10 vs Qwen: {qwen_score}/10)")
        write_file(root, gemma_path, gemma_content)
        state.store_file_content(gemma_path, gemma_content)
        validation = validate_written_file(root, _fix_absolute_path(gemma_path, root), pre_flight=flags.pre_flight)
        if validation.get("ok"):
            state.mark_done_by_index(step_index)
            return {"ok": True, "summary": f"Gemma won dual-gen: {gemma_path} ({gemma_score}/10)"}
        return {"ok": False, "summary": f"Gemma output failed validation: {validation.get('stderr', '')}"}

    elif qwen_score >= gemma_score:
        if qwen_score > gemma_score:
            print(f"   🏆 Qwen wins! (Qwen: {qwen_score}/10 vs Gemma: {gemma_score}/10)")
        else:
            print(f"   🤝 Tie! (Both: {qwen_score}/10) — using Qwen")
        if qwen_score >= flags.fast_path_threshold:
            state.mark_done_by_index(step_index)
            return {"ok": True, "summary": f"Qwen won dual-gen: {path} ({qwen_score}/10)"}
        else:
            return {"ok": False, "summary": f"Qwen won dual-gen but scored {qwen_score}/10 — needs improvement"}

    # Both failed
    return {"ok": False, "summary": "Dual-gen: both models failed to produce valid output"}