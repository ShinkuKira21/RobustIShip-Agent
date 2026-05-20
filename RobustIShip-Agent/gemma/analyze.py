"""Gemma failure analysis for /fix command."""

from prompts.reflection import GEMMA_REFLECTION_SYSTEM
from utils.logging import log_event


def gemma_analyze_failures(cpu_model, *, user_goal: str, failed_tasks: list[dict],
                           command_history: list[dict], args=None) -> str:
    failure_text = "\n".join(f"- {f.get('task', 'unknown')}: {f.get('error', 'no error')}" for f in failed_tasks)
    history_text = "\n".join(
        f"COMMAND: {c.get('cmd')} | OK: {c.get('ok')} | {c.get('stderr', '')[:200]}"
        for c in command_history[-6:]
    )
    messages = [
        {"role": "system", "content": GEMMA_REFLECTION_SYSTEM},
        {"role": "user", "content": (
            f"USER GOAL: {user_goal}\n\n"
            f"The following steps failed during execution:\n{failure_text}\n\n"
            f"Recent command history:\n{history_text or '(none)'}\n\n"
            "Analyze these failures and suggest concrete repair steps."
        )},
    ]
    try:
        if cpu_model and cpu_model.model:
            response = cpu_model.generate(messages, max_tokens=384, temperature=0.0)
        else:
            return "Could not analyze failures: Gemma model not available."
        log_event(args, {
            "model_role": "gemma", "purpose": "fix_analysis",
            "model": getattr(cpu_model, "model_id", None),
            "messages": messages, "raw_response": response,
        })
        return response.strip()
    except Exception as e:
        return f"Analysis failed: {e}"