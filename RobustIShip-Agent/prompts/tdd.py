"""TDD-specific prompt additions for planner and reflection."""

TDD_PLANNER_ADDITION = """
TDD TWO-SPRINT MODE IS ACTIVE:

SPRINT 1: TEST ITERATION — Write a production-quality test file.
  - Qwen generates test attempts.
  - Gemma scores each attempt and guides improvements.
  - Iterates until test quality ≥ 8/10.

SPRINT 2: IMPLEMENTATION — Write code, verify, polish.
  - Qwen writes all implementation files.
  - Run tests. Fix failures until all pass.
  - Gemma reviews every file. Agent-patches anything below 8/10.
  - Final verification confirms everything works.

CRITICAL RULES:
- Plan MUST include a test file step (e.g., write_file tests/test_taskforge.py).
- Plan MUST include a test run step (e.g., run_command python3 -m unittest).
- Tests test real code, not mocks or MagicMock.
- Implementation steps come before the test run step.
- Do NOT include meta steps for empty workspaces.
"""

TDD_REFLECTION_ADDITION = """
TDD TWO-SPRINT MODE IS ACTIVE:
- Tests are the specification. Implementation must make them pass.
- If tests fail, diagnose the ROOT CAUSE and fix the implementation.
- Do not suggest CONTINUE if tests are failing.
- Target one fix per RETRY. Be specific about which file and what change.
"""