"""Gemma plan creation and parsing."""

import re
from prompts.planner import GEMMA_PLANNER_SYSTEM
from utils.logging import log_event


def gemma_create_plan(cpu_model, goal: str, project_context: str = "", args=None) -> str:
    user_msg = f"Create a structured plan for this goal:\n\n{goal}"
    if project_context:
        user_msg = f"PROJECT CONTEXT:\n{project_context}\n\nGOAL:\n{goal}\n\nCreate a structured plan."
  
    # Inject TDD instructions if flag is active (OUTSIDE the if project_context block)
    if args and getattr(args, "tdd", False):
        from prompts.tdd import TDD_PLANNER_ADDITION
        system_msg = GEMMA_PLANNER_SYSTEM + "\n" + TDD_PLANNER_ADDITION
    else:
        system_msg = GEMMA_PLANNER_SYSTEM
    
    messages = [
        {"role": "system", "content": system_msg},  # ← Use system_msg here
        {"role": "user", "content": user_msg},
    ]
    
    if cpu_model and cpu_model.model:
        response = cpu_model.generate(messages, max_tokens=1024, temperature=0.2).strip()
        log_event(args, {
            "model_role": "gemma", "purpose": "plan",
            "model": getattr(cpu_model, "model_id", None),
            "messages": messages, "raw_response": response,
        })
        return response
    return "Error: No model available for planning."


def parse_structured_plan(text: str) -> list[dict]:
    steps = []
    current_step = None
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            if current_step and current_step.get("qwen_prompt"):
                steps.append(current_step)
                current_step = None
            continue
        if re.match(r'^STEP\s+\d+', line, re.IGNORECASE):
            if current_step and current_step.get("qwen_prompt"):
                steps.append(current_step)
            current_step = {"step": line, "tool": "", "target": "", "qwen_prompt": "", "context_needed": ""}
        elif current_step:
            upper = line.upper()
            if upper.startswith("TOOL:"):
                current_step["tool"] = line[5:].strip().lower()
            elif upper.startswith("TARGET:"):
                current_step["target"] = line[7:].strip()
            elif upper.startswith("QWEN_PROMPT:"):
                current_step["qwen_prompt"] = line[12:].strip()
            elif upper.startswith("CONTEXT_NEEDED:"):
                current_step["context_needed"] = line[15:].strip()
            elif current_step.get("qwen_prompt") and not re.match(r'^STEP\s+\d+', line, re.IGNORECASE):
                current_step["qwen_prompt"] += " " + line
    if current_step and current_step.get("qwen_prompt"):
        steps.append(current_step)
    return steps


def has_meta_steps(steps: list[dict]) -> bool:
    return any(s.get("tool", "").lower() == "meta" for s in steps)
