"""Apply Gemma's reflection decision to the execution state."""

import re
from pathlib import Path

from flags import FeatureFlags
from state import StateManager
from gemma.plan import parse_structured_plan
from qwen.actions import request_qwen_action_with_validation
from orchestrator.agent_patch import execute_agent_patch, handle_offer_patch


def apply_reflection_decision(cpu_model, args, root, state: StateManager, step_index: int,
                              decision: str, qwen_prompt: str, expected_tool: str,
                              injected_context: str, reflection_context: dict | None = None,
                              flags: FeatureFlags = None) -> tuple:
    decision_upper = decision.upper().strip()

    if decision_upper.startswith("CONTINUE"):
        return "continue", True

    elif flags and flags.retry_step and decision_upper.startswith("RETRY_STEP:"):
        match = re.match(r'RETRY_STEP:\s*(\d+)\s*\|\s*(.+)', decision, re.DOTALL | re.IGNORECASE)
        if match:
            target_step = int(match.group(1)) - 1
            new_prompt = match.group(2).strip()
            if 0 <= target_step < len(state.structured_plan) and target_step != step_index:
                print(f"   ↩️  Gemma says: RETRY_STEP {target_step+1}")
                state.checklist[step_index]["done"] = True
                state.mark_not_done_by_index(target_step)
                state.structured_plan[target_step]["qwen_prompt"] = new_prompt
                return ("retry_step_jump", target_step), True
        return "continue", True

    elif decision_upper.startswith("OFFER_PATCH:"):
        analysis = decision[13:].strip()
        print("   🔧 Gemma offers to patch directly.")
        result, should_advance = handle_offer_patch(state, step_index, analysis)
        if result == "agent_patch_chosen":
            if reflection_context:
                success = execute_agent_patch(cpu_model, state, step_index, reflection_context, root)
                if success:
                    return "agent_patched", True
            return "agent_patch_failed", True
        elif result == "retry_chosen":
            new_prompt = f"{qwen_prompt}\n\n[USER CHOSE RETRY. Previous analysis: {analysis[:200]}]"
            retry_action = request_qwen_action_with_validation(
                cpu_model, args, root, None, new_prompt,
                injected_context=injected_context,
                observations=state.get_recent_observations(),
                step_index=step_index, purpose="user_retry",
            )
            if retry_action:
                return ("retry", retry_action), False
            return "retry_failed", True
        return result, should_advance

    elif decision_upper.startswith("RETRY:"):
        full_decision = decision[6:].strip()
        new_prompt = full_decision
        dynamic_ctx = ""
        
        # Support for Gemma-selected context: "RETRY: Fix X. CONTEXT: file.py:class Y"
        if "CONTEXT:" in full_decision.upper():
            parts = re.split(r'CONTEXT:', full_decision, flags=re.IGNORECASE)
            new_prompt = parts[0].strip()
            context_query = parts[1].strip()
            
            # Resolve snippets immediately
            context_parts = []
            for item in [i.strip() for i in context_query.split(",")]:
                if ":" in item:
                    try:
                        path, q = item.split(":", 1)
                        snippet = state.get_snippet(path, q)
                        if snippet:
                            context_parts.append(f"--- SNIPPET: {item} ---\n{snippet}")
                        else:
                            content = state.get_file_content(path)
                            if content:
                                context_parts.append(f"--- FILE: {path} ---\n{content[:2000]}")
                    except ValueError:
                        pass
                else:
                    content = state.get_file_content(item)
                    if content:
                        context_parts.append(f"--- FILE: {item} ---\n{content[:2000]}")
            dynamic_ctx = "\n".join(context_parts)

        if new_prompt:
            print("   🔄 Gemma says: RETRY with revised prompt")
            if dynamic_ctx:
                print(f"   🧠 Gemma selected targeted context ({len(dynamic_ctx)} chars)")
            
            # Inject dynamic context into the step temporarily for this retry
            original_ctx = state.structured_plan[step_index].get("dynamic_context", "")
            state.structured_plan[step_index]["dynamic_context"] = dynamic_ctx
            
            retry_action = request_qwen_action_with_validation(
                cpu_model, args, root, None, new_prompt,
                injected_context=state.get_context_for_step(step_index),
                observations=state.get_recent_observations(),
                step_index=step_index, purpose="gemma_retry",
            )
            
            # Restore
            state.structured_plan[step_index]["dynamic_context"] = original_ctx
            
            if retry_action:
                return ("retry", retry_action), False
            return "retry_failed", True
        return "retry_empty_prompt", True

    elif decision_upper.startswith("EXPAND:"):
        new_step_text = decision[7:].strip()
        print("   🔧 Gemma says: EXPAND — injecting quality sub-step")
        new_steps = parse_structured_plan(new_step_text)
        if new_steps:
            insert_pos = step_index + 1
            for ns in reversed(new_steps):
                state.structured_plan.insert(insert_pos, ns)
                state.checklist.insert(insert_pos, {"task": ns.get("qwen_prompt", ""), "done": False})
            print(f"   ✅ Injected {len(new_steps)} sub-step(s)")
        return "expanded", True

    elif decision_upper.startswith("REPLACE_NEXT:"):
        new_next = decision[13:].strip()
        print("   🔀 Gemma says: REPLACE_NEXT")
        next_idx = step_index + 1
        if next_idx < len(state.structured_plan):
            new_steps = parse_structured_plan(new_next)
            if new_steps:
                del state.structured_plan[next_idx]
                del state.checklist[next_idx]
                for ns in reversed(new_steps):
                    state.structured_plan.insert(next_idx, ns)
                    state.checklist.insert(next_idx, {"task": ns.get("qwen_prompt", ""), "done": False})
                print(f"   ✅ Replaced step {next_idx+1}")
        return "replaced_next", True

    elif decision_upper.startswith("SKIP_NEXT:"):
        reason = decision[10:].strip()
        print(f"   ⏭️  Gemma says: SKIP_NEXT — {reason}")
        next_idx = step_index + 1
        if next_idx < len(state.structured_plan):
            state.checklist[next_idx]["done"] = True
            print(f"   ✅ Skipped step {next_idx+1}")
        return "skipped_next", True

    elif decision_upper.startswith("ASK_USER:"):
        question = decision[9:].strip()
        print(f"\n{'=' * 60}")
        print(f"🤔 Gemma needs your input:\n   {question}")
        print(f"{'=' * 60}")
        try:
            answer = input("   Your answer (or 'skip' to skip this step): ").strip()
        except EOFError:
            answer = "skip"
        if answer.lower() == "skip":
            return "user_skipped", True
        new_prompt = f"{qwen_prompt}\n\n[USER CLARIFICATION: {answer}]"
        retry_action = request_qwen_action_with_validation(
            cpu_model, args, root, None, new_prompt,
            injected_context=injected_context,
            observations=state.get_recent_observations(),
            step_index=step_index, purpose="user_clarified_retry",
        )
        if retry_action:
            return ("retry", retry_action), False
        return "user_retry_failed", True

    elif decision_upper == "DONE":
        print("   🎉 Gemma says: DONE — goal achieved!")
        for i in range(step_index, len(state.checklist)):
            state.checklist[i]["done"] = True
        return "done_early", True

    else:
        print(f"   ⚠️  Unrecognized Gemma decision: {decision[:100]}...")
        print("   ⏩ Defaulting to CONTINUE")
        return "continue", True
