"""Qwen fast-path reflection — lightweight step monitor."""

from pathlib import Path
from models.remote import chat_server


def qwen_fast_reflection(args, root: Path, step_prompt: str, tool: str,
                         result_ok: bool, result_summary: str, next_step_prompt: str) -> str:
    system_msg = (
        "You are a lightweight step monitor. After a tool executes, decide if the next step "
        "should proceed normally or if something needs attention.\n\n"
        "Reply EXACTLY one of:\n"
        "  CONTINUE — everything is fine, proceed to next step.\n"
        "  FLAG: <brief reason> — something is wrong, needs escalation.\n\n"
        "Flag when: the step failed, output is incomplete, a file wasn't found, "
        "a command had errors, the result doesn't match the task intent."
    )
    status = "SUCCESS" if result_ok else "FAILED"
    user_msg = (
        f"STEP: {step_prompt[:300]}\nTOOL: {tool}\nSTATUS: {status}\n"
        f"RESULT: {result_summary[:500]}\n"
        f"NEXT STEP: {next_step_prompt[:200] if next_step_prompt else '(last step)'}\n\n"
        "CONTINUE or FLAG?"
    )
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    try:
        response = chat_server(
            args.base_url, args.model, messages,
            api_key=args.api_key, temperature=0.0, top_p=1.0, max_tokens=64, debug=False,
        )
        response = response.strip()
        if response.upper().startswith("CONTINUE"):
            return "CONTINUE"
        return f"FLAG: {response}"
    except Exception:
        return "CONTINUE"