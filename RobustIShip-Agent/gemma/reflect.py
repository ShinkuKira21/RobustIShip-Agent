"""Gemma step reflection."""

import json
from prompts.reflection import GEMMA_REFLECTION_SYSTEM
from utils.logging import log_event


def gemma_reflect_on_step(cpu_model, *, reflection_context: dict, args=None) -> str:
    context_text = json.dumps(reflection_context, indent=2, default=str)
    
    # Inject TDD instructions if flag is active
    if args and getattr(args, "tdd", False):
        from prompts.tdd import TDD_REFLECTION_ADDITION
        system_msg = GEMMA_REFLECTION_SYSTEM + "\n" + TDD_REFLECTION_ADDITION
    else:
        system_msg = GEMMA_REFLECTION_SYSTEM
    
    messages = [
        {"role": "system", "content": system_msg},  # ← Use system_msg here
        {"role": "user", "content": f"REFLECTION CONTEXT:\n{context_text}\n\nDecide the next action. Output ONE command only."},
    ]
    try:
        if cpu_model and cpu_model.model:
            response = cpu_model.generate(messages, max_tokens=512, temperature=0.0)
        else:
            return "CONTINUE"
        log_event(args, {
            "model_role": "gemma", "purpose": "reflection",
            "model": getattr(cpu_model, "model_id", None),
            "step_prompt": reflection_context.get("current_step", {}).get("prompt", ""),
            "tool": reflection_context.get("current_step", {}).get("tool", ""),
            "messages": messages, "raw_response": response,
        })
        return response.strip()
    except Exception as e:
        import sys
        print(f"[reflect error] {e}", file=sys.stderr)
        return "CONTINUE"