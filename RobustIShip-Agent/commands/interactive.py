"""Main interactive loop."""

import sys
from pathlib import Path

from config import BANNER, HELP_CMDS
from flags import FeatureFlags
from gemma.plan import gemma_create_plan, parse_structured_plan, has_meta_steps
from orchestrator.execution import structured_execution_loop
from orchestrator.meta_steps import execute_meta_steps
from commands.planning import planning_mode
from commands.fix import fix_command
from memory import fix_memory
from utils.text_utils import _flush_stdin


def interactive_loop(cpu_model, args, root: Path, state, flags: FeatureFlags = None):
    if flags is None:
        flags = FeatureFlags()
    state.init_history()

    print(BANNER)
    print(HELP_CMDS)
    print(f"{'=' * 60}\n")

    while True:
        try:
            _flush_stdin()
            user_input = input("🎯 You: ").strip()
            if not user_input:
                continue
            if user_input.startswith("/"):
                parts = user_input.split()
                cmd = parts[0].lower()
                if cmd == "/exit":
                    print("\n👋 Goodbye!\n")
                    break
                elif cmd == "/plan":
                    if len(parts) > 1:
                        state.set_goal(" ".join(parts[1:]))
                    planning_mode(cpu_model, state, args)
                elif cmd == "/go":
                    if state.structured_plan:
                        structured_execution_loop(cpu_model, args, root, state, flags)
                    elif state.user_goal:
                        print("📝 No structured plan found. Creating one...")
                        print("   💡 Tip: Use /plan first for file discovery and better precision.")
                        plan_text = gemma_create_plan(cpu_model, state.user_goal, args=args)
                        steps = parse_structured_plan(plan_text)
                        for _ in range(3):
                            if not steps or not has_meta_steps(steps):
                                break
                            meta_steps = [s for s in steps if s.get("tool", "").lower() == "meta"]
                            project_context = execute_meta_steps(cpu_model, args, root, state, meta_steps)
                            plan_text = gemma_create_plan(cpu_model, state.user_goal, project_context=project_context, args=args)
                            steps = parse_structured_plan(plan_text)
                        if steps:
                            state.set_structured_plan(steps)
                        else:
                            state.set_structured_plan([{"step": "Execute", "tool": "run_command", "qwen_prompt": state.user_goal, "context_needed": "none"}])
                        structured_execution_loop(cpu_model, args, root, state, flags)
                    else:
                        print("⚠️ No plan or goal. Use /plan first.")
                elif cmd == "/save":
                    state.save()
                elif cmd == "/load":
                    if state.load():
                        state.init_history()
                        print("✅ State loaded. Type /go to continue.")
                elif cmd == "/status":
                    if state.structured_plan:
                        print(f"\n🎯 Goal: {state.user_goal[:80]}...")
                        print(f"\n{state.get_progress_block()}")
                    else:
                        print("📭 No active plan.")
                elif cmd == "/fix":
                    fix_command(cpu_model, state, args)
                elif cmd == "/clear":
                    state.clear()
                    print("🧹 All state cleared.")
                elif cmd == "/fixes":
                    if fix_memory.command_fixes:
                        print("\n📚 Saved command fixes:")
                        for p, r in fix_memory.command_fixes.items():
                            print(f"   {p} -> {r}")
                    else:
                        print("📭 No saved fixes")
                elif cmd == "/help":
                    print(HELP_CMDS)
                else:
                    print(f"   Unknown command: {cmd}")
                continue
            state.set_goal(user_input)
            print("✅ Goal set. Use /plan to create a structured plan.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()