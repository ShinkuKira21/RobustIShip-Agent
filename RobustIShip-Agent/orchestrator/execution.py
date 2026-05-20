"""Execution loop — one micro sprint for every action, macro sprints for organization."""

import shlex
from pathlib import Path

from flags import FeatureFlags
from state import StateManager
from system.system_memory import system_memory
from tools.files import read_file, write_file, edit_file
from tools.search import grep_search
from tools.commands import run_command
from tools.validate import validate_written_file
from tools.normalize import _fix_absolute_path
from gemma.quality import gemma_quality_check
from gemma.reflect import gemma_reflect_on_step
from gemma.review import gemma_review_code
from qwen.actions import request_qwen_action_with_validation
from qwen.fast_reflection import qwen_fast_reflection
from orchestrator.reflection import apply_reflection_decision
from orchestrator.agent_patch import execute_agent_patch
from orchestrator.strategies.fast_path import handle as fast_path_handle
from orchestrator.strategies.warn_path import handle as warn_path_handle
from orchestrator.strategies.dual_gen import handle as dual_gen_handle
from orchestrator.strategies.multi_gen import handle as multi_gen_handle
from orchestrator.strategies.gemma_takeover import handle as gemma_takeover_handle
from orchestrator.strategies.research_path import handle as research_path_handle
from orchestrator.strategies.tdd_assembly import handle as tdd_assembly_handle
from config import _REVIEWABLE_EXTS


# ═══════════════════════════════════════════════════════════════
# UNIVERSAL MICRO SPRINT
# ═══════════════════════════════════════════════════════════════

def _micro_sprint(cpu_model, args, root, state, flags, step_index, step, max_retries=3):
    """Execute one step with full self-healing.
    
    Tier 1: System cache (instant)
    Tier 2: Model known issues (instant)
    Tier 3: Gemma reflection + RETRY (model call)
    
    Records all discovered fixes to system memory.
    """
    expected_tool = step.get("tool", "")
    qwen_prompt = step.get("qwen_prompt", "")
    injected_context = state.get_context_for_step(step_index)
    
    for attempt in range(max_retries):
        # Execute the step
        result = _execute_single_step(cpu_model, args, root, state, flags, step_index, step)
        
        if result.get("ok"):
            return result
        
        # Failed — run full reflection
        error_msg = result.get("error", result.get("stderr", "unknown"))
        print(f"   🔄 Micro sprint attempt {attempt+1}/{max_retries}: {error_msg[:100]}")
        
        reflection_context = state.assemble_reflection_context(
            step_index, f"Step failed: {error_msg}", current_tool=expected_tool
        )
        gemma_decision = gemma_reflect_on_step(cpu_model, reflection_context=reflection_context, args=args)
        print(f"   🧠 Gemma: {gemma_decision[:150]}...")
        
        state.history.record({
            "type": "reflection", "step_index": step_index,
            "gemma_decision": gemma_decision[:300],
        })
        
        if gemma_decision.upper().startswith("RETRY:"):
            new_prompt = gemma_decision[6:].strip()
            
            # If Gemma is telling us to fix a file, switch tool to write_file
            is_file_fix = any(word in new_prompt.lower() for word in [
                "update ", "fix ", "edit ", "correct ", "rewrite ", "modify ", "change ", "patch ",
                "create ", "write ", "replace "
            ])
            looks_like_code = any(pattern in new_prompt for pattern in [
                "def ", "import ", "class ", "from ", "if __name__", "@dataclass",
                "self.", "return ", "print(", "json."
            ])
            
            if is_file_fix or looks_like_code:
                print(f"   🔧 Switching tool to write_file based on Gemma's guidance")
                state.structured_plan[step_index]["tool"] = "write_file"
            
            state.structured_plan[step_index]["qwen_prompt"] = new_prompt
            state.structured_plan[step_index]["dynamic_context"] = _extract_dynamic_context(
                new_prompt, state
            )

            continue
        
        elif gemma_decision.upper().startswith("RETRY_STEP:"):
            result["retry_step"] = gemma_decision
            return result
        
        elif gemma_decision.upper().startswith("OFFER_PATCH:"):
            print(f"   🔧 Gemma offers agent patch")
            reflection_context = state.assemble_reflection_context(
                step_index, error_msg, current_tool=expected_tool
            )
            success = execute_agent_patch(cpu_model, state, step_index, reflection_context, root)
            if success:
                return {"ok": True, "path": step.get("target", "")}
            continue
        
        elif gemma_decision.upper().startswith("CONTINUE") or gemma_decision.upper().startswith("DONE"):
            break
    
    return result


def _extract_dynamic_context(prompt: str, state) -> str:
    """Extract CONTEXT: references from a RETRY prompt."""
    if "CONTEXT:" not in prompt.upper():
        return ""
    parts = prompt.upper().split("CONTEXT:", 1)
    context_files = [f.strip() for f in parts[1].split(",")]
    ctx_parts = []
    for f in context_files:
        content = state.get_file_content(f)
        if content:
            ctx_parts.append(f"--- {f} ---\n{content[:1500]}")
    return "\n".join(ctx_parts)


# ═══════════════════════════════════════════════════════════════
# SINGLE STEP EXECUTION (called by micro sprint)
# ═══════════════════════════════════════════════════════════════

def _execute_single_step(cpu_model, args, root, state, flags, step_index, step):
    """Execute one step. Returns dict with 'ok' and tool-specific fields."""
    expected_tool = step.get("tool", "")
    qwen_prompt = step.get("qwen_prompt", "")
    injected_context = state.get_context_for_step(step_index)

    try:
        action = request_qwen_action_with_validation(
            cpu_model, args, root, expected_tool, qwen_prompt,
            injected_context=injected_context,
            observations=state.get_recent_observations(),
            step_index=step_index,
            purpose="step_execute",
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}

    if action is None:
        return {"ok": False, "error": "Unparseable response"}

    if "final" in action:
        return {"ok": True, "summary": "Agent signaled final"}

    tool = action.get("tool")
    tool_args = action.get("args") or {}

    if tool == "read_file":
        path = tool_args.get("path", "")
        result = read_file(root, path)
        if result.get("ok"):
            state.store_file_content(result['path'], result['content'])
            return {"ok": True, "path": path, "summary": f"Read {path}"}
        return {"ok": False, "error": result.get("error", "read failed")}

    elif tool == "grep_search":
        pattern = tool_args.get("pattern", "")
        include = tool_args.get("include", "*")
        result = grep_search(root, pattern, include=include)
        if result.get("ok"):
            return {"ok": True, "summary": f"Grep found {result['total_matches']} matches"}
        return {"ok": False, "error": result.get("error", "grep failed")}

    elif tool == "write_file":
        path = tool_args.get("path", "")
        content = tool_args.get("content", "")
        
        # Determine if we're in automated mode
        is_automated = getattr(flags, "tdd", False) or getattr(args, "yes", False)
        has_apply = getattr(args, "apply", False)
        
        if is_automated or has_apply:
            # Auto-commit
            write_file(root, path, content)
            state.store_file_content(path, content)
            print(f"   ✍️  Written: {path}")
        else:
            # Interactive — show quality first, then ask
            quality = gemma_quality_check(cpu_model, step_prompt=qwen_prompt, path=path, content=content, args=args)
            score = quality.get("score", 7)
            issues = quality.get("issues", [])
            
            print(f"\n   📝 {path} — Quality: {score}/10")
            if issues:
                print(f"   Issues: {', '.join(issues[:3])}")
            print(f"   Preview: {content[:200]}...")
            print(f"   [Y] Write to workspace  [N] Skip  [B] Let Gemma pick best version")
            
            choice = input("   Choice [Y/n/b]: ").strip().lower()
            
            if choice in ("", "y", "yes"):
                write_file(root, path, content)
                state.store_file_content(path, content)
                print(f"   ✍️  Written: {path}")
            elif choice == "b":
                print("   ℹ️  Gemma review not yet implemented. Writing file directly.")
                print("   💡 Use --tdd flag for Gemma-powered TDD Assembly.")
                write_file(root, path, content)
                state.store_file_content(path, content)
                print(f"   ✍️  Written: {path}")
            else:
                return {"ok": False, "error": "User skipped"}

        # Validation and quality check (for auto-commit path)
        validation = validate_written_file(root, _fix_absolute_path(path, root), pre_flight=flags.pre_flight)
        if not validation.get("ok"):
            error = validation.get("stderr", "") or validation.get("error", "")
            print(f"   ❌ Validation: {error[:200]}")
            return {"ok": False, "error": error, "path": path}

        quality = gemma_quality_check(cpu_model, step_prompt=qwen_prompt, path=path, content=content, args=args)
        score = quality.get("score", 7)
        
        if score >= 8:
            return {"ok": True, "path": path, "quality_score": score}
        else:
            issues = quality.get("issues", [])
            return {"ok": False, "error": f"Quality {score}/10: {issues}", "path": path, "quality_score": score}

    elif tool == "edit_file":
        path = tool_args.get("path", "")
        result = edit_file(root, path, tool_args.get("old_str", ""), tool_args.get("new_str", ""))
        if result.get("ok"):
            refreshed = read_file(root, path)
            if refreshed.get("ok"):
                state.store_file_content(path, refreshed["content"])
            print(f"   ✏️  Edited: {path}")
            return {"ok": True, "path": path}
        return {"ok": False, "error": result.get("error", "edit failed")}

    elif tool == "run_command":
        cmd = step.get("target", "") or tool_args.get("cmd", "")
        python_path = system_memory.data.get("environment", {}).get("python_path")
        
        if "unittest" in cmd or "pytest" in cmd:
            if not python_path:
                return {"ok": False, "error": "Python path not found in system memory"}
            if " -m " in cmd:
                cmd = python_path + " -m " + cmd.split(" -m ", 1)[1]
            else:
                parts = cmd.split(" ", 1)
                cmd = python_path + " " + parts[1] if len(parts) > 1 else python_path
        
        # Safety gate for non-automated modes
        is_automated = getattr(flags, "tdd", False) or getattr(args, "yes", False)
        has_run = getattr(args, "run", False)
        
        if not is_automated and not has_run:
            print(f"\n   💻 Would run: {cmd}")
            choice = input("   Run this command? [Y/n]: ").strip().lower()
            if choice in ("n", "no"):
                return {"ok": False, "error": "User skipped command"}
        
        result = run_command(root, cmd, flags=flags)
        state.store_command_result(cmd, result)
        
        if result["ok"]:
            print("   💻 Command executed")
            if result.get("stdout"):
                print(f"   📤 {result['stdout'][:200]}")
            return {"ok": True, "stdout": result.get("stdout", ""), "stderr": result.get("stderr", "")}
        else:
            print(f"   ❌ Failed: {result.get('stderr', '')[:200]}")
            return {"ok": False, "stderr": result.get("stderr", ""), "stdout": result.get("stdout", "")}

    return {"ok": False, "error": f"Unknown tool: {tool}"}

# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def structured_execution_loop(cpu_model, args, root, state: StateManager, flags: FeatureFlags):
    if not state.structured_plan:
        print("No structured plan to execute.")
        return

    state.structured_plan = [s for s in state.structured_plan if s.get("tool", "").lower() != "meta"]
    state.checklist = [
        {"task": s.get("qwen_prompt", s.get("step", "")), "done": False}
        for s in state.structured_plan
    ]
    state.failed_tasks = []
    state.consecutive_failures = 0
    state.mark_session_start()

    if not state.structured_plan:
        print("No concrete steps to execute.")
        return

    print(f"\nExecuting: {state.user_goal[:80]}...")

    if flags.tdd:
        _execute_tdd_mode(cpu_model, args, root, state, flags)
    else:
        _execute_default_mode(cpu_model, args, root, state, flags)

    done = len(state.get_done())
    total = len(state.checklist)
    print(f"\n{'=' * 60}")
    print(f"📊 Summary: {done}/{total} steps completed")
    if state.failed_tasks:
        print(f"⚠️  {len(state.failed_tasks)} failures — use /fix to analyze")
    print(f"{'=' * 60}")
    state.save()


# ═══════════════════════════════════════════════════════════════
# DEFAULT MODE — Step by step with micro sprints
# ═══════════════════════════════════════════════════════════════

def _execute_default_mode(cpu_model, args, root, state, flags):
    """Original step-by-step execution. Every step gets a micro sprint."""
    max_steps = min(len(state.structured_plan), args.max_steps)
    step_index = 0

    while step_index < max_steps:
        step = state.structured_plan[step_index]
        expected_tool = step.get("tool", "")

        if expected_tool == "meta":
            state.checklist[step_index]["done"] = True
            step_index += 1
            continue

        if state.checklist[step_index].get("done"):
            step_index += 1
            continue

        if state.history.get_retry_count(step_index) >= 3:
            print(f"[Step {step_index+1}] ⚠️  Failed 3+ times. Skipping.")
            step_index += 1
            continue

        print(f"\n[Step {step_index+1}/{len(state.structured_plan)}] {expected_tool}: {step.get('target', '')}")

        result = _micro_sprint(cpu_model, args, root, state, flags, step_index, step)

        if result.get("ok"):
            state.mark_done_by_index(step_index)
            state.save()
        elif result.get("retry_step"):
            # Handle RETRY_STEP signal
            state.mark_done_by_index(step_index)
            # Extract target step from decision — simplified here
            print(f"   ↩️  RETRY_STEP requested")
        else:
            state.record_failure(step_index, result.get("error", "Unknown"))

        pending_dir = root / ".robustIship" / "pending"
        if pending_dir.exists():
            print(f"\n{'=' * 60}")
            print(f"🔍 Reviewing {len(list(pending_dir.glob('*')))} deferred files with Gemma...")
            for pending_file in pending_dir.glob("*"):
                content = pending_file.read_text()
                original_name = pending_file.name.rsplit(".v", 1)[0]
                print(f"   📝 {original_name}: {content[:100]}...")
                choice = input(f"   Accept Gemma's version? [Y/n]: ").strip().lower()
                if choice in ("", "y", "yes"):
                    # Write to workspace
                    dest = root / original_name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content)
                    print(f"   ✅ Written: {original_name}")
                pending_file.unlink()

        step_index += 1
        # Review deferred files




# ═══════════════════════════════════════════════════════════════
# TDD MODE — Test Iteration → Implementation → Verify → Polish
# ═══════════════════════════════════════════════════════════════
def _execute_tdd_mode(cpu_model, args, root, state, flags):
    """TDD two-sprint flow. Every phase uses micro sprints internally."""
    
    # Classify steps
    test_steps, impl_steps, verify_steps = [], [], []
    for i, step in enumerate(state.structured_plan):
        tool = step.get("tool", "").lower()
        target = step.get("target", "").strip()
        if tool == "write_file" and "test_" in Path(target).name:
            test_steps.append((i, step))
        elif tool == "run_command" and any(kw in (step.get("qwen_prompt", "") + target).lower()
                                          for kw in ["unittest", "pytest", "test_"]):
            verify_steps.append((i, step))
        else:
            impl_steps.append((i, step))

    # ── SPRINT 1: Test iteration ──
    if test_steps:
        print(f"\n{'=' * 60}")
        print(f"🧪 TDD SPRINT 1: TEST ITERATION")
        print(f"{'=' * 60}")
        for idx, step in test_steps:
            print(f"\n[Test] {step.get('target', '')}")
            result = _micro_sprint(cpu_model, args, root, state, flags, idx, step, max_retries=5)
            if result.get("ok"):
                state.mark_done_by_index(idx)
            else:
                # Try TDD Assembly as last resort
                print(f"   🔄 Micro sprint exhausted — trying TDD Assembly...")
                assembly_result = tdd_assembly_handle(state, idx, root, cpu_model, args, flags)
                if assembly_result.get("ok"):
                    state.mark_done_by_index(idx)

    # ── SPRINT 2: Implementation ──
    print(f"\n{'=' * 60}")
    print(f"🧪 TDD SPRINT 2: IMPLEMENTATION")
    print(f"{'=' * 60}")
    for idx, step in impl_steps:
        print(f"\n[Impl] {step.get('target', '')}")
        result = _micro_sprint(cpu_model, args, root, state, flags, idx, step)
        if result.get("ok"):
            state.mark_done_by_index(idx)

    # ── VERIFY ──
    if verify_steps:
        print(f"\n{'=' * 60}")
        print(f"🧪 TDD VERIFY — Running tests, fixing failures")
        print(f"{'=' * 60}")
        
        python_path = system_memory.data.get("environment", {}).get("python_path", "python3")
        
        for idx, step in verify_steps:
            target = step.get("target", "")
            cmd = target
            if " -m " in cmd:
                cmd = python_path + " -m " + cmd.split(" -m ", 1)[1]
            
            print(f"\n[Verify] {cmd}")
            result = run_command(root, cmd, flags=flags)
            state.store_command_result(cmd, result)
            
            if result["ok"]:
                print(f"   ✅ Tests passed")
                state.mark_done_by_index(idx)
            else:
                stderr = result.get("stderr", "")
                print(f"   ❌ Tests failed: {stderr[:200]}")
                
                broken_file = None
                if "models.py" in stderr or "Task" in stderr:
                    broken_file = "models.py"
                elif "storage.py" in stderr or "JSONStorage" in stderr:
                    broken_file = "storage.py"
                elif "cli.py" in stderr or "cli" in stderr.lower():
                    broken_file = "cli.py"
                
                if not broken_file:
                    for part in target.split():
                        if part.endswith(".py"):
                            broken_file = part
                            break
                
                if broken_file:
                    print(f"   🔧 Diagnosed: {broken_file} needs fixing")
                    fix_step = {
                        "tool": "write_file",
                        "target": broken_file,
                        "qwen_prompt": f"Fix {broken_file} so tests pass. Error: {stderr[:300]}"
                    }
                    fix_result = _micro_sprint(cpu_model, args, root, state, flags, idx, fix_step, max_retries=3)
                    if fix_result.get("ok"):
                        result2 = run_command(root, cmd, flags=flags)
                        if result2["ok"]:
                            print(f"   ✅ Tests pass after fix!")
                            state.mark_done_by_index(idx)
                            
    # ── POLISH ──
    print(f"\n{'=' * 60}")
    print(f"🧪 TDD POLISH — Gemma reviewing all files")
    print(f"{'=' * 60}")
    for idx, step in impl_steps:
        target = step.get("target", "").strip()
        if not target:
            continue
        current = read_file(root, target)
        if not current.get("ok"):
            continue
        quality = gemma_quality_check(cpu_model, step_prompt=step.get("qwen_prompt", ""),
                                       path=target, content=current["content"], args=args)
        score = quality.get("score", 7)
        if score < 8:
            print(f"   🔍 {target}: {score}/10 — Gemma polishing...")
            reflection_context = state.assemble_reflection_context(
                idx, f"Quality {score}/10: {quality.get('issues', [])}", current_tool="write_file"
            )
            success = execute_agent_patch(cpu_model, state, idx, reflection_context, root)
            if success:
                print(f"   ✅ {target} polished")
        else:
            print(f"   ✅ {target}: {score}/10")

    # ── SIGN OFF ──
    if verify_steps:
        print(f"\n{'=' * 60}")
        print(f"🧪 TDD SIGN OFF")
        print(f"{'=' * 60}")
        for idx, step in verify_steps:
            result = _micro_sprint(cpu_model, args, root, state, flags, idx, step)
            if result.get("ok"):
                print(f"   ✅ Final verification passed!")
                state.mark_done_by_index(idx)
            else:
                print(f"   ❌ Final verification failed: {result.get('stderr', '')[:200]}")