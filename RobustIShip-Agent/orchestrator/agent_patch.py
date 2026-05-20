"""Agent patch — Gemma writes file directly, bypassing Qwen."""

import json
from pathlib import Path

from state import StateManager
from tools.files import write_file


def handle_offer_patch(state: StateManager, step_index: int, analysis: str) -> tuple:
    print(f"\n{'=' * 60}")
    print(f"🔧 Gemma Analysis:\n   {analysis[:500]}")
    print(f"{'=' * 60}")
    print("   [R] Retry — let Qwen try again")
    print("   [A] Agent patch — Gemma writes directly")
    print("   [C] Cancel — skip this step")

    while True:
        try:
            choice = input("   Choice [R/A/C]: ").strip().upper()
        except EOFError:
            print("\n   ⚠️  Input closed, defaulting to Cancel.")
            choice = "C"
        if choice == "R":
            return "retry_chosen", False
        elif choice == "A":
            return "agent_patch_chosen", False
        elif choice == "C":
            state.checklist[step_index]["done"] = True
            print("   ⏭️  Step cancelled by user.")
            return "cancelled", True
        print("   Please enter R, A, or C.")


def execute_agent_patch(cpu_model, state: StateManager, step_index: int,
                        reflection_context: dict, root: Path) -> bool:
    context_text = json.dumps(reflection_context, indent=2, default=str)
    patch_prompt = (
        f"REFLECTION CONTEXT:\n{context_text}\n\n"
        "The user chose Agent Patch. Write the file directly.\n"
        "Format:\nPATH: <relative file path>\n\n<complete file content>"
    )
    messages = [
        {"role": "system", "content": "You are a direct file writer. Output the corrected file content."},
        {"role": "user", "content": patch_prompt},
    ]

    try:
        if cpu_model and cpu_model.model:
            response = cpu_model.generate(messages, max_tokens=2048, temperature=0.0).strip()
        else:
            print("   ❌ Agent model not available.")
            return False

        import re
        lines = response.split('\n')
        path = ""
        content = ""
        if lines and lines[0].startswith("PATH:"):
            path = lines[0][5:].strip()
            content_start = 1
            if content_start < len(lines) and lines[content_start].strip() == "":
                content_start += 1
            content = '\n'.join(lines[content_start:]).strip()
        else:
            plan_step = state.structured_plan[step_index]
            path = plan_step.get("target", "").strip()
            content = response

        if not path:
            print("   ❌ Could not determine target file path.")
            return False
        if not content:
            print("   ❌ No file content in response.")
            return False

        content = re.sub(r'^```\w*\n?', '', content)
        content = re.sub(r'\n?```$', '', content)

        write_file(root, path, content)
        state.store_file_content(path, content)
        state.mark_done_by_index(step_index)
        print(f"   ✅ Agent patched: {path}")
        return True
    except Exception as e:
        print(f"   ❌ Agent patch failed: {e}")
        return False