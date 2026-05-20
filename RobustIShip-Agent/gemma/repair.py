"""Gemma JSON repair and retry prompt generation."""

from prompts.validator import GEMMA_VALIDATOR_SYSTEM
from prompts.retry import GEMMA_RETRY_SYSTEM
from utils.json_utils import _extract_json_object
from utils.logging import log_event


def gemma_fix_json(cpu_model, bad_json: str, expected_tool: str, original_prompt: str, args=None) -> str | None:
    messages = [
        {"role": "system", "content": GEMMA_VALIDATOR_SYSTEM},
        {"role": "user", "content": f"EXPECTED TOOL: {expected_tool}\nORIGINAL PROMPT: {original_prompt}\nBROKEN RESPONSE:\n{bad_json}"},
    ]
    if cpu_model and cpu_model.model:
        response = cpu_model.generate(messages, max_tokens=512, temperature=0.0)
        fixed = _extract_json_object(response)
        log_event(args, {
            "model_role": "gemma", "purpose": "json_repair",
            "model": getattr(cpu_model, "model_id", None),
            "expected_tool": expected_tool, "messages": messages,
            "raw_response": response, "parsed_json": fixed,
        })
        return fixed
    return None


def gemma_create_retry_prompt(cpu_model, *, task: str, expected_tool: str, failure: str,
                              observations: str, context: str, args=None) -> str:
    messages = [
        {"role": "system", "content": GEMMA_RETRY_SYSTEM},
        {"role": "user", "content": (
            f"TASK:\n{task}\n\nEXPECTED TOOL: {expected_tool}\n\n"
            f"FAILURE:\n{failure}\n\nOBSERVATIONS:\n{observations or '(none)'}\n\n"
            f"AVAILABLE CONTEXT:\n{context or '(none)'}\n\n"
            "Write the next prompt for Qwen to repair this with one tool call."
        )},
    ]
    try:
        if cpu_model and cpu_model.model:
            response = cpu_model.generate(messages, max_tokens=512, temperature=0.0)
        else:
            return task
        log_event(args, {
            "model_role": "gemma", "purpose": "retry_prompt",
            "model": getattr(cpu_model, "model_id", None),
            "expected_tool": expected_tool, "failure": failure,
            "messages": messages, "raw_response": response,
        })
        return response.strip() or task
    except Exception as e:
        import sys
        print(f"[repair error] {e}", file=sys.stderr)
        return task