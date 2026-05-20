"""Warn path strategy — quality score = warn threshold, accept with warning."""

def handle(state, step_index: int, path: str, score: int, quality: dict) -> dict:
    issues = quality.get("issues", [])
    print(f"   ⚠️  Quality gate: {score}/10 — acceptable but not ideal")
    if issues:
        print(f"   Issues: {', '.join(issues[:3])}")
    state.mark_done_by_index(step_index)
    return {
        "ok": True,
        "summary": f"Written (warning, {score}/10): {path}",
        "state_update": True,
    }