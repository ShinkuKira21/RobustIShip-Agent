"""TDD Gate — simplified: verification happens in sprint phase, not as a blocker."""

def handle(state, step_index: int, test_result: dict, expected_tool: str,
           qwen_prompt: str, flags) -> dict:
    """TDD gate: returns test result for the sprint verify phase."""
    return {
        "ok": test_result.get("ok", False),
        "summary": f"Tests {'passed' if test_result.get('ok') else 'failed'}: {test_result.get('stderr', '')[:200]}",
        "state_update": test_result.get("ok", False),
    }