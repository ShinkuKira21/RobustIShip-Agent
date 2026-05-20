"""/fix command — analyze failures with Gemma."""

from state import StateManager
from gemma.analyze import gemma_analyze_failures


def fix_command(cpu_model, state: StateManager, args):
    if not state.failed_tasks:
        print("✅ No failures to analyze.")
        return
    print(f"\n🔍 Analyzing {len(state.failed_tasks)} failure(s)...")
    analysis = gemma_analyze_failures(
        cpu_model, user_goal=state.user_goal,
        failed_tasks=state.failed_tasks,
        command_history=state.command_history, args=args,
    )
    print(f"{'=' * 60}\n📋 Gemma's Analysis:\n{analysis}\n{'=' * 60}")