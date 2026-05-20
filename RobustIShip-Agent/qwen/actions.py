"""Request tool actions from Qwen (remote model)."""

import json
from pathlib import Path

from config import ALLOWED_TOOLS
from models.remote import chat_server
from tools.normalize import normalize_action
from gemma.repair import gemma_create_retry_prompt
from utils.json_utils import _extract_json_object
from utils.logging import log_event


def request_qwen_action(args, root: Path, expected_tool: str | None, prompt: str,
                        injected_context: str = "", observations: str = "",
                        step_index: int | None = None, purpose: str = "tool_call") -> dict | None:
    if expected_tool:
        tool_instruction = f"Required tool: {expected_tool}"
        tool_reminder = f"\n\nRespond with ONE JSON object using tool={expected_tool}."
    else:
        tool_instruction = f"Choose the best tool from: {', '.join(sorted(ALLOWED_TOOLS))}"
        tool_reminder = "\n\nRespond with ONE JSON object using the most appropriate tool."

    system_msg = (
        f"You are a strict JSON API. Output ONE JSON object only.\n"
        f"{tool_instruction}\n\n"
        f"JSON format rules:\n"
        f"1. Root must have a 'tool' key.\n"
        f"2. Double quotes only. Escape internal double quotes with \\.\n"
        f"3. Use \\n for newlines. Do NOT use backticks.\n"
        f"4. Backslashes in regex (grep_search) MUST be double-escaped (e.g., \\\\s, \\\\w).\n"
        f"5. write_file args: 'path' and 'content'.\n"
        f"6. edit_file args: 'path', 'old_str', 'new_str'.\n"
        f"7. run_command args: 'cmd'.\n"
        f"8. read_file args: 'path', optional 'start_line' (int, 1-based), optional 'end_line' (int).\n"
        f"9. grep_search args: 'pattern' and optional 'include' (glob pattern, default '*').\n"
        f"No prose. No markdown. Just the JSON."
    )
    user_msg = f"Workspace: {root}\n\n{prompt}"
    if observations:
        user_msg += f"\n\n{observations}"
    if injected_context:
        user_msg += f"\n\n{injected_context}"
    if expected_tool == "write_file":
        user_msg += "\n\nFor write_file, include the target path and complete file content."
    elif expected_tool == "edit_file":
        user_msg += "\n\nFor edit_file, old_str must appear exactly once in the current file."
    user_msg += tool_reminder

    qwen_messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    qwen_response = chat_server(
        args.base_url, args.model, qwen_messages,
        api_key=args.api_key, temperature=float(args.temperature),
        top_p=float(args.top_p), max_tokens=int(args.max_tokens), debug=args.debug,
    )
    if args.debug:
        print(f"[debug] Qwen: {qwen_response[:300]}", file=__import__("sys").stderr)

    extracted = _extract_json_object(qwen_response)
    parsed_action = None
    rejection = None
    if extracted:
        try:
            parsed = json.loads(extracted)
            action, _ = normalize_action(parsed, root)
            parsed_action = action
            if expected_tool and action and "final" not in action and action.get("tool") != expected_tool:
                rejection = f"wrong_tool: expected {expected_tool}, got {action.get('tool')}"
                if args.debug:
                    print(f"[debug] Rejecting wrong tool", file=__import__("sys").stderr)
                log_event(args, {
                    "model_role": "qwen", "purpose": purpose, "step_index": step_index,
                    "model": args.model, "expected_tool": expected_tool,
                    "messages": qwen_messages, "raw_response": qwen_response,
                    "extracted_json": extracted, "parsed_action": parsed_action, "rejection": rejection,
                })
                return None
            log_event(args, {
                "model_role": "qwen", "purpose": purpose, "step_index": step_index,
                "model": args.model, "expected_tool": expected_tool,
                "messages": qwen_messages, "raw_response": qwen_response,
                "extracted_json": extracted, "parsed_action": parsed_action,
            })
            return action
        except Exception as e:
            rejection = f"parse_error: {e}"
    log_event(args, {
        "model_role": "qwen", "purpose": purpose, "step_index": step_index,
        "model": args.model, "expected_tool": expected_tool,
        "messages": qwen_messages, "raw_response": qwen_response,
        "extracted_json": extracted, "parsed_action": parsed_action, "rejection": rejection or "no_json",
    })
    return None


def request_qwen_action_with_validation(cpu_model, args, root: Path, expected_tool: str | None, prompt: str,
                                        injected_context: str = "", observations: str = "",
                                        step_index: int | None = None, purpose: str = "tool_call") -> dict | None:
    action = request_qwen_action(args, root, expected_tool, prompt, injected_context, observations, step_index, purpose)
    if action is not None:
        return action
    try:
        fixed_prompt = gemma_create_retry_prompt(
            cpu_model, task=prompt, expected_tool=expected_tool or "unknown",
            failure="Qwen returned an unparseable or invalid tool call.",
            observations=observations, context=injected_context, args=args,
        )
        return request_qwen_action(args, root, expected_tool, fixed_prompt, injected_context, observations, step_index, f"{purpose}_json_retry")
    except Exception:
        return None