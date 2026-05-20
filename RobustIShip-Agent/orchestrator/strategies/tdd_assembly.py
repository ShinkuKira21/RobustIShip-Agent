"""TDD Assembly — Gemma curates Qwen's accumulated test versions into one best test."""

import re
from pathlib import Path

from tools.files import write_file, read_file
from tools.validate import validate_written_file
from tools.normalize import _fix_absolute_path
from gemma.quality import gemma_quality_check


def handle(state, step_index: int, root: Path, gemma_model, args, flags) -> dict:
    """Activated on loop detection in TDD mode.
    
    Qwen has generated multiple versions of a test file but keeps looping.
    Gemma reads all versions, selects the best parts from each,
    and assembles a final test targeting ≥ 8/10.
    """
    plan_step = state.structured_plan[step_index]
    test_path = plan_step.get("target", "").strip()
    
    if not test_path:
        return {"ok": False, "summary": "TDD Assembly: no target path"}

    # Get all file versions from history
    versions = state.history.get_file_versions(test_path)
    
    if len(versions) < 2:
        # Not enough versions — let Gemma write from scratch using agent_patch
        print("   🧠 TDD Assembly: Only 1 version — Gemma writing directly from context")
        from orchestrator.agent_patch import execute_agent_patch
        reflection_context = state.assemble_reflection_context(step_index, "Loop detected — need fresh test", current_tool="write_file")
        success = execute_agent_patch(gemma_model, state, step_index, reflection_context, root)
        if success:
            return {"ok": True, "summary": f"Gemma wrote fresh test: {test_path}"}
        return {"ok": False, "summary": "Gemma agent_patch failed"}

    print(f"   🧠 TDD Assembly: Merging {len(versions)} versions into one test...")

    # Read current file for context
    current = read_file(root, test_path)
    current_content = current.get("content", "") if current.get("ok") else ""

    # Build assembly prompt
    versions_text = ""
    for i, v in enumerate(versions):
        versions_text += f"\n=== VERSION {i+1} (seq {v['seq']}) ===\n{v['content'][:2000]}\n"

    prompt_text = (
        f"You are assembling the best test file from multiple Qwen attempts.\n\n"
        f"TARGET FILE: {test_path}\n"
        f"TASK: {plan_step.get('qwen_prompt', '')[:500]}\n\n"
        f"Qwen generated {len(versions)} versions. Each has strengths and weaknesses.\n"
        f"{versions_text}\n\n"
        f"CURRENT FILE ON DISK:\n{current_content[:1500]}\n\n"
        f"Your job:\n"
        f"1. Identify the BEST parts from each version\n"
        f"2. Assemble a FINAL version that combines:\n"
        f"   - Correct imports at the top\n"
        f"   - Real assertions with actual values (not MagicMock everywhere)\n"
        f"   - Proper test structure (unittest.TestCase)\n"
        f"   - Self-contained or properly mocked dependencies\n"
        f"3. If implementation classes (Task, JSONStorage) don't exist yet,\n"
        f"   include minimal inline stubs with @dataclass so the test is runnable\n"
        f"4. Output ONLY the complete file content. No explanation.\n"
        f"   Start with imports, end with 'if __name__ == \"__main__\":'\n"
        f"Target quality: 10/10."
    )

    messages = [
        {"role": "system", "content": "You are an expert Python test engineer. Assemble the best test from multiple attempts. Output only the complete file."},
        {"role": "user", "content": prompt_text},
    ]

    try:
        if gemma_model and gemma_model.model:
            response = gemma_model.generate(messages, max_tokens=2048, temperature=0.0).strip()
        else:
            return {"ok": False, "summary": "Gemma model not available"}

        # Clean response
        response = re.sub(r'^```\w*\n?', '', response)
        response = re.sub(r'\n?```$', '', response).strip()

        if not response or len(response) < 50:
            return {"ok": False, "summary": "Gemma returned empty or too-short response"}

        # Write the assembled test
        write_file(root, test_path, response)
        state.store_file_content(test_path, response)

        # Validate
        validation = validate_written_file(root, _fix_absolute_path(test_path, root), pre_flight=flags.pre_flight)
        if not validation.get("ok"):
            return {"ok": False, "summary": f"Assembly validation failed: {validation.get('stderr', '')[:200]}"}

        # Quality check
        quality = gemma_quality_check(
            gemma_model, 
            step_prompt=plan_step.get("qwen_prompt", ""), 
            path=test_path, 
            content=response, 
            args=args
        )
        score = quality.get("score", 0)
        issues = quality.get("issues", [])

        if score >= 8:
            state.mark_done_by_index(step_index)
            print(f"   🏆 TDD Assembly: {score}/10 — test ready!")
            return {"ok": True, "summary": f"TDD Assembly: {score}/10 — {test_path}"}
        else:
            print(f"   ⚠️  TDD Assembly: {score}/10 — needs improvement")
            if issues:
                print(f"   Issues: {', '.join(issues[:3])}")
            return {"ok": False, "summary": f"TDD Assembly scored {score}/10: {issues[:2] if issues else 'unknown'}"}

    except Exception as e:
        return {"ok": False, "summary": f"TDD Assembly error: {e}"}