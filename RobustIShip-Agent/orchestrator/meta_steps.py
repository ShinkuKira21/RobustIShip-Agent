"""Meta-step execution — gather project context for Gemma planning."""

import shlex
from pathlib import Path

from state import StateManager
from tools.files import read_file
from tools.commands import run_command
from gemma.plan import has_meta_steps as _has_meta_steps

# Re-export for convenience
has_meta_steps = _has_meta_steps


def execute_meta_steps(cpu_model, args, root: Path, state: StateManager, meta_steps: list[dict]) -> str:
    print("\n📋 Gemma needs more context. Gathering project information...\n")
    context_parts = []

    for i, step in enumerate(meta_steps):
        prompt = step.get("qwen_prompt", "")
        target = step.get("target", "").strip()
        print(f"  🔍 Meta-step {i+1}: {prompt[:100]}...")
        
        prompt_lower = prompt.lower()
        handled = False

        if "list" in prompt_lower or "structure" in prompt_lower or "tree" in prompt_lower:
            # Use pre-scanned workspace instead of running find
            overview = state.scanner.get_context_block()
            if overview and "empty" not in overview:
                context_parts.append(overview)
                print(f"     ✅ Workspace overview ({state.scanner.file_count} files)")
                handled = True
            else:
                # Fallback to find command if scanner has nothing
                result = run_command(root, "find . -maxdepth 3 -not -path './.git/*' -not -path './__pycache__/*' -not -path './.robustIship/*' -not -path './venv/*' | head -80")
                if result.get("ok"):
                    context_parts.append(f"PROJECT FILES:\n{result['stdout']}")
                    print("     ✅ Listed files")
                    handled = True
                else:
                    context_parts.append(f"FILE LISTING FAILED: {result.get('stderr', 'unknown')}")
                    print("     ❌ Failed to list files")

        if "readme" in prompt_lower:
            for name in ["README.md", "readme.md", "README.txt", "README"]:
                result = read_file(root, name, max_lines=200)
                if result.get("ok"):
                    context_parts.append(f"README ({name}):\n{result['content'][:2000]}")
                    state.store_file_content(name, result["content"])
                    print(f"     ✅ Read {name}")
                    handled = True
                    break
            else:
                if not handled:
                    context_parts.append("README: Not found")
                    print("     ⚠️  No README found")

        if "pyproject" in prompt_lower or "package.json" in prompt_lower or "makefile" in prompt_lower or "config" in prompt_lower:
            for cfg in ["pyproject.toml", "package.json", "Makefile", "makefile", "setup.py", "setup.cfg"]:
                result = read_file(root, cfg, max_lines=150)
                if result.get("ok"):
                    context_parts.append(f"CONFIG ({cfg}):\n{result['content'][:1500]}")
                    state.store_file_content(cfg, result["content"])
                    print(f"     ✅ Read {cfg}")
                    handled = True

        if target == "return_to_planner":
            context_parts.append("[END OF CONTEXT GATHERING — return to Gemma for real planning]")
            print("     📤 Context gathering complete.")
            break

        if not handled:
            # Fallback to target-based reading
            result = read_file(root, target, max_lines=200)
            if result.get("ok"):
                context_parts.append(f"FILE ({target}):\n{result['content'][:1500]}")
                state.store_file_content(target, result["content"])
                print(f"     ✅ Read {target}")
            else:
                result = run_command(root, f"ls -la {shlex.quote(target)} 2>/dev/null || find . -name '{shlex.quote(target)}' 2>/dev/null | head -20")
                if result.get("ok") and result.get("stdout"):
                    context_parts.append(f"SEARCH ({target}):\n{result['stdout']}")
                    print(f"     ✅ Found matches for {target}")

    return "\n\n".join(context_parts)