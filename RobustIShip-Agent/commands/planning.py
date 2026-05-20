"""Interactive planning mode."""

import re
from pathlib import Path

from state import StateManager
from gemma.plan import gemma_create_plan, parse_structured_plan, has_meta_steps
from orchestrator.meta_steps import execute_meta_steps


def planning_mode(cpu_model, state: StateManager, args):
    if state.structured_plan:
        print(f"\n📋 Current plan ({len(state.structured_plan)} steps):")
        for i, s in enumerate(state.structured_plan):
            done = "✅ DONE" if i < len(state.checklist) and state.checklist[i].get("done") else "⏳ TODO"
            print(f"  {i+1}. [{done}] [{s.get('tool','?')}] {s.get('qwen_prompt','')[:80]}")
        print("\n📝 Commands: /add n 'task', /remove n, /move x y, /refine, /done, /cancel, /help")
    else:
        if not state.user_goal:
            state.set_goal(input("🎯 What's your goal? ").strip())
        print("\n🤖 Gemma is creating a structured plan...")
        workspace_overview = state.scanner.get_context_block()
        
        # Check for existing tests the user may have written
        existing_tests = [f for f in state.scanner.files if "test_" in Path(f).name]
        if existing_tests:
            workspace_overview += f"\n\n[EXISTING TESTS FOUND: {', '.join(existing_tests)}. These are the SPECIFICATION. Do NOT write new tests. Plan implementation steps to make these tests pass.]"
        
        plan_text = gemma_create_plan(cpu_model, state.user_goal, project_context=workspace_overview, args=args)

        steps = parse_structured_plan(plan_text)
        for _ in range(3):
            if not steps or not has_meta_steps(steps):
                break
            meta_steps = [s for s in steps if s.get("tool", "").lower() == "meta"]
            project_context = execute_meta_steps(cpu_model, args, Path(args.root).resolve(), state, meta_steps)
            print("\n🤖 Gemma is creating the final plan with gathered context...")
            workspace_overview = state.scanner.get_context_block()
            
            # Check for existing tests the user may have written
            existing_tests = [f for f in state.scanner.files if "test_" in Path(f).name]
            if existing_tests:
                workspace_overview += f"\n\n[EXISTING TESTS FOUND: {', '.join(existing_tests)}. These are the SPECIFICATION. Do NOT write new tests. Plan implementation steps to make these tests pass.]"
            
            plan_text = gemma_create_plan(cpu_model, state.user_goal, project_context=workspace_overview, args=args)

            steps = parse_structured_plan(plan_text)
        print(f"\n{plan_text}")
        if steps:
            state.set_structured_plan(steps)
            print(f"\n✅ Plan created with {len(steps)} steps!")
            print("\n📝 Commands: /add n 'task', /remove n, /move x y, /refine, /done, /cancel, /help")
        else:
            print("\n⚠️ Could not parse structured plan.")
            state.set_structured_plan([{"step": "Execute", "tool": "read_file", "qwen_prompt": state.user_goal, "context_needed": "none"}])

    while True:
        cmd = input("\n/plan> ").strip()
        if not cmd:
            continue
        if cmd in ["/done", "/save"]:
            state.save()
            print("💾 Plan saved. Type /go to execute.")
            return
        elif cmd == "/cancel":
            state.clear()
            print("🗑️ Plan discarded.")
            return
        elif cmd.startswith("/add "):
            rest = cmd[5:].strip()
            match = re.match(r'^(\d+)\s+(.+)', rest)
            if match:
                pos = int(match.group(1)) - 1
                new_step = {"step": f"Step {pos+1}", "tool": "write_file", "qwen_prompt": match.group(2), "context_needed": "none"}
                if 0 <= pos <= len(state.structured_plan):
                    state.structured_plan.insert(pos, new_step)
                    state.checklist.insert(pos, {"task": new_step["qwen_prompt"], "done": False})
                    print(f"   ✅ Added at position {pos+1}")
            else:
                new_step = {"step": f"Step {len(state.structured_plan)+1}", "tool": "write_file", "qwen_prompt": rest, "context_needed": "none"}
                state.structured_plan.append(new_step)
                state.checklist.append({"task": new_step["qwen_prompt"], "done": False})
                print(f"   ✅ Added step {len(state.structured_plan)}")
        elif cmd.startswith("/remove"):
            parts = cmd.split()
            if len(parts) >= 2:
                try:
                    idx = int(parts[1]) - 1
                    if 0 <= idx < len(state.structured_plan):
                        state.structured_plan.pop(idx)
                        state.checklist.pop(idx)
                        print(f"   ✅ Removed step {idx+1}")
                except ValueError:
                    pass
        elif cmd.startswith("/move "):
            parts = cmd.split()
            if len(parts) >= 3:
                try:
                    frm, to = int(parts[1]) - 1, int(parts[2]) - 1
                    if 0 <= frm < len(state.structured_plan) and 0 <= to < len(state.structured_plan):
                        s = state.structured_plan.pop(frm)
                        c = state.checklist.pop(frm)
                        state.structured_plan.insert(to, s)
                        state.checklist.insert(to, c)
                        print(f"   ✅ Moved {frm+1} -> {to+1}")
                except ValueError:
                    pass
        elif cmd == "/refine":
            print("🤖 Gemma is refining the plan...")
            context = "Current plan:\n" + "\n".join(f"{i+1}. [{s.get('tool','')}] {s.get('qwen_prompt','')}" for i, s in enumerate(state.structured_plan))
            print(f"\n{gemma_create_plan(cpu_model, f'{state.user_goal}\n\n{context}', args=args)}")
        elif cmd == "/help":
            print("\n📝 Planning Commands: /add n 'task', /remove n, /move x y, /refine, /done, /cancel")
        else:
            print(f"   Unknown command: {cmd}.")