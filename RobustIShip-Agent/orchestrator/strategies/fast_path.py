"""Fast path strategy — quality score ≥ threshold, accept immediately."""

def handle(state, step_index: int, path: str, score: int) -> dict:
    state.mark_done_by_index(step_index)
    return {
        "ok": True,
        "summary": f"Written and validated: {path} (quality: {score}/10)",
        "state_update": True,
    }