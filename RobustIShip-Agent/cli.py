"""Command-line interface for RobustIShip."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from config import VERSION, BANNER
from flags import FeatureFlags
from memory import fix_memory
from models.local import LocalModel
from state import StateManager
from system.system_memory import system_memory
from orchestrator.execution import structured_execution_loop
from orchestrator.meta_steps import execute_meta_steps, has_meta_steps
from gemma.plan import gemma_create_plan, parse_structured_plan
from commands.interactive import interactive_loop


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"RobustIShip {VERSION} — Modular Multi-Strategy Code Generation Agent"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--cpu-model", default="google/gemma-4-E4B-it") # v0.27 - cpu-model will be deprecated and renamed to --agent-model (am)
    parser.add_argument("--am-device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--am-max-vram-gib", type=float, default=None)
    parser.add_argument("--am-quant", choices=["none", "4bit", "8bit", "fp16"], default="fp16")
    parser.add_argument("--am-max-memory-gib", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--api-key", default=os.getenv("MODEL_API_KEY"))
    parser.add_argument("--prompt", help="One-shot prompt")
    parser.add_argument("--root", default=".")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--review-mode", choices=["off", "failures-only", "all"], default="failures-only")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--clear-fixes", action="store_true")

    # Debug flag toggles
    parser.add_argument("--verbose", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--log-output", action="store_true", help="Write model calls to .robustIship/logs/")
    parser.add_argument("--debug", action="store_true", help="Show raw HTTP responses and quality debug")

    # Feature flag toggles
    parser.add_argument("--minimal", action="store_true", help="Run with minimal overhead (no dual-gen)")
    parser.add_argument("--experimental", action="store_true", help="Enable experimental multi-gen features")
    parser.add_argument("--tdd", action="store_true", help="Test-driven development: tests must pass to advance")
    parser.add_argument("--no-dual-gen", action="store_true", help="Disable dual generation")
    parser.add_argument("--no-pre-flight", action="store_true", help="Disable pre-flight validation")
    parser.add_argument("--no-retry-step", action="store_true", help="Disable RETRY_STEP command")
    parser.add_argument("--no-session-filter", action="store_true", help="Disable session filtering")
    parser.add_argument("--dual-gen-threshold", type=int, default=None, help="Quality score threshold for dual-gen")
    parser.add_argument("--takeover-threshold", type=int, default=None, help="Quality score threshold for Gemma takeover")

    # Explicit toggles to force-enable dual/multi generation regardless of hardware
    parser.add_argument("--dual-gen", action="store_true", help="Explicitly enable dual-generation (overrides GPU check)")
    parser.add_argument("--multi-gen", action="store_true", help="Explicitly enable multi-generation (overrides GPU check)")

    # Version Flag
    parser.add_argument("--version", action="version", version=f"RobustIShip {VERSION}")

    args = parser.parse_args()
    if args.yes:
        args.apply = args.run = True
    if args.clear_fixes:
        fix_memory.command_fixes = {}
        fix_memory.save_to_file()
        print("🧹 Cleared all saved fixes")

    load_dotenv()
    root = Path(args.root).resolve()
    flags = FeatureFlags.from_args(args)

    print(f"\n📦 Loading Gemma (Planner + Validator + Reviewer + Reflector + Generator)...")
    
    # Start workspace scanner in background thread
    from tools.scanner import WorkspaceScanner
    scanner = WorkspaceScanner(root)
    scanner_thread = scanner.start_async()
    
    gemma = LocalModel(
        args.cpu_model, device=args.am_device, quant=args.am_quant,
        max_memory_gib=args.am_max_memory_gib, max_vram_gib=args.am_max_vram_gib
    )
    gemma.load()
    
    # Wait for scanner to finish (it's usually done by now)
    scanner_thread.join(timeout=5)
    print(f"   📂 Workspace pre-scanned: {scanner.file_count} files")
    


    # Gate dual-gen and multi-gen behind GPU availability unless explicitly requested.
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    # Determine whether Gemma/agent is running on GPU (explicit --am-device or auto with CUDA)
    agent_on_gpu = False
    if args.am_device == "cuda" and cuda_available:
        agent_on_gpu = True
    elif args.am_device == "auto" and cuda_available:
        agent_on_gpu = True

    # If user explicitly passed --dual-gen/--multi-gen, respect that; otherwise require GPU
    if getattr(args, "dual_gen", False):
        flags.dual_gen = True
    else:
        # If user did not explicitly enable, only allow if agent is on GPU
        flags.dual_gen = bool(flags.dual_gen and agent_on_gpu)
    
    dual_gen_reason = ""
    if not flags.dual_gen and not agent_on_gpu:
        dual_gen_reason = " (Gemma on CPU — use --dual-gen to force)"

    if getattr(args, "multi_gen", False):
        flags.multi_gen = True
    else:
        flags.multi_gen = bool(flags.multi_gen and agent_on_gpu)

    # Run System Summary
    env_summary = system_memory.get_environment_context().replace('\n', ' | ')

    # Output CLI
    print(f"\n{'=' * 60}")
    print(f"✅ Agent ready!")
    print(f"   🚀 Qwen (Executor): {args.model}")
    print(f"   🧠 Gemma (Orchestrator): {args.cpu_model} ({args.am_quant})")
    print(f"   ⚔️  Dual-Gen: {'On' if flags.dual_gen else 'Off'}{dual_gen_reason} (threshold ≤{flags.dual_gen_threshold})")
    print(f"   🧬 Multi-Gen: {'On' if flags.multi_gen else 'Off'}")
    print(f"   🔍 Pre-flight: {'On' if flags.pre_flight else 'Off'}")
    print(f"   🧪 TDD Mode: {'On' if flags.tdd else 'Off'}")
    print(f"   🌍 Workspace: {root}")
    print(f"   🖥️  System: {env_summary}")
    print(f"{'=' * 60}")

    if args.prompt:
        state = StateManager(root, flags)
        state.scanner = scanner  # Reuse the pre-scanned instance
        state.init_history()
        state.set_goal(args.prompt)
        print("\n🧠 Gemma is planning the one-shot prompt...")
        plan_text = gemma_create_plan(gemma, args.prompt, args=args)
        steps = parse_structured_plan(plan_text)
        max_meta_rounds = 3
        for _ in range(max_meta_rounds):
            if not steps or not has_meta_steps(steps):
                break
            meta_steps = [s for s in steps if s.get("tool", "").lower() == "meta"]
            project_context = execute_meta_steps(gemma, args, root, state, meta_steps)
            plan_text = gemma_create_plan(gemma, args.prompt, project_context=project_context, args=args)
            steps = parse_structured_plan(plan_text)
        if steps:
            state.set_structured_plan(steps)
        else:
            state.set_structured_plan([
                {"step": "Execute", "tool": "run_command", "qwen_prompt": args.prompt, "context_needed": "none"}
            ])
        structured_execution_loop(gemma, args, root, state, flags)
        return 0
    else:
        from state import StateManager as SM
        st = SM(root, flags)
        st.init_history()
        import traceback
        try:
            interactive_loop(gemma, args, root, st, flags)
        except Exception as e:
            traceback.print_exc()
            return 1
        return 0
    
if __name__ == "__main__":
    sys.exit(main())