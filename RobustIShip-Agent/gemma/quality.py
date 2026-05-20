"""Gemma quality gate — scores file content against task requirements."""

import sys
import json
from utils.json_utils import _extract_json_object


def gemma_quality_check(cpu_model, *, step_prompt: str, path: str, content: str, args=None) -> dict:
    messages = [
        {"role": "system", "content": (
            "You are a code quality inspector. Review a file against its task description.\n"
            "Output JSON: {\"pass\": true/false, \"score\": 1-10, \"issues\": [\"list of problems\"]}\n"
            "Pass if score >= 8. Be strict about functionality matching requirements.\n"
            "IMPORTANT: The orchestrator handles directory creation automatically.\n"
            "Only evaluate the FILE CONTENT — do not penalize for missing setup/creation logic.\n"
            "Focus on: does the code actually do what the task asks for?\n"
            "For test files, do not pass tests that only assert mock call counts such as sleep calls.\n"
            "Tests must verify observable behavior like printed output, return values, or real assertions about results."
        )},
        {"role": "user", "content": f"TASK: {step_prompt[:500]}\nFILE: {path}\nCONTENT:\n{content[:2000]}"},
    ]
    try:
        if cpu_model and cpu_model.model:
            response = cpu_model.generate(messages, max_tokens=512, temperature=0.0)
            if args and getattr(args, "debug", False):
                print(f"[quality debug] raw response: {response[:200]}", file=sys.stderr)
            extracted = _extract_json_object(response)
            if args and getattr(args, "debug", False):
                print(f"[quality debug] extracted: {extracted}", file=sys.stderr)
            if extracted:
                result = json.loads(extracted)
                if args and getattr(args, "debug", False):
                    print(f"[quality debug] result: {result}", file=sys.stderr)
                return result
    except Exception as exc:
        print(f"[quality gate error] {exc}", file=sys.stderr)
        return {"pass": False, "score": 0, "issues": [f"Quality gate error: {exc}"]}
    
    return {"pass": True, "score": 7, "issues": []}