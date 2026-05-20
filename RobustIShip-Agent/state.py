"""State manager — mirror memory, plan tracking, reflection log."""

import json
import re
from pathlib import Path
from typing import Any

from history import HistoryMap
from flags import FeatureFlags

from tools.scanner import WorkspaceScanner


class StateManager:
    def __init__(self, root: Path, flags: FeatureFlags = None):
        self.root = root
        self.flags = flags or FeatureFlags()
        self.state_dir = root / ".robustIship"
        self.state_file = self.state_dir / "state.json"
        self.user_goal: str = ""
        self.structured_plan: list[dict] = []
        self.checklist: list[dict] = []
        self.file_contents: dict[str, str] = {}
        self.files_read: dict[str, str] = {}
        self.failed_tasks: list[dict] = []
        self.command_history: list[dict[str, Any]] = []
        self.consecutive_failures: int = 0
        self._write_order: list[str] = []
        self.reflection_log: list[dict] = []
        self.project_instructions: str = ""
        self.history = HistoryMap(root, log_events=flags.log_events if flags else False)
        self.scanner = WorkspaceScanner(root)

    def init_history(self):
        self.history.init()
        # Load GEMINI.md if it exists
        gemini_md = self.root / "GEMINI.md"
        if gemini_md.exists():
            try:
                self.project_instructions = gemini_md.read_text(encoding="utf-8")
                print(f"📖 Loaded project instructions from GEMINI.md")
            except Exception as e:
                print(f"⚠️  Failed to read GEMINI.md: {e}")

    def mark_session_start(self):
        if self.flags.session_filter:
            self.history.mark_session_start()

    def set_goal(self, goal: str):
        self.user_goal = goal

    def set_structured_plan(self, plan: list[dict]):
        plan = [s for s in plan if s.get("tool", "").lower() != "meta"]
        self.structured_plan = plan
        self.checklist = [{"task": s.get("qwen_prompt", s.get("step", "")), "done": False} for s in plan]
        self.failed_tasks = []
        self.reflection_log = []

    def store_file_content(self, path: str, content: str):
        try:
            rel_path = str(Path(path).resolve().relative_to(self.root.resolve()))
        except Exception:
            rel_path = path
        self.file_contents[rel_path] = content
        if rel_path in self._write_order:
            self._write_order.remove(rel_path)
        self._write_order.append(rel_path)
        if len(self.file_contents) > 30:
            oldest = list(self.file_contents.keys())[0]
            del self.file_contents[oldest]
        self.history.snapshot_file(rel_path, content)
        self.scanner.upsert(rel_path, content)

    def get_file_content(self, path: str) -> str | None:
        return self.file_contents.get(path)

    def get_snippet(self, path: str, query: str) -> str | None:
        """Extract a specific class, function, or block from a file."""
        content = self.file_contents.get(path)
        if not content:
            return None
        
        query = query.lower().strip()
        lines = content.splitlines()
        
        # Try to find class or function
        if query.startswith("class "):
            name = query[6:].strip()
            pattern = rf"^class\s+{re.escape(name)}[\(:]"
        elif query.startswith("func "):
            name = query[5:].strip()
            pattern = rf"^def\s+{re.escape(name)}\("
        else:
            # General substring search if no prefix
            for i, line in enumerate(lines):
                if query in line.lower():
                    start = max(0, i - 10)
                    end = min(len(lines), i + 30)
                    return "\n".join(lines[start:end])
            return None

        start_idx = -1
        for i, line in enumerate(lines):
            if re.match(pattern, line):
                start_idx = i
                break
        
        if start_idx == -1:
            return None
            
        # Extract block based on indentation
        result = [lines[start_idx]]
        base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if not line.strip():
                result.append(line)
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= base_indent:
                if line.strip():
                    break
            result.append(line)
        return "\n".join(result)

    def get_context_for_step(self, step_index: int) -> str:
        if step_index >= len(self.structured_plan):
            return ""
        step = self.structured_plan[step_index]
        context_needed = step.get("context_needed", "").strip()
        
        # Support for dynamic context injected by Gemma during retries
        dynamic_context = step.get("dynamic_context", "")
        
        context_parts = []
        if self.scanner.file_count > 0:
            context_parts.append(self.scanner.get_context_block())
        
        # Always include project-wide instructions if available
        if self.project_instructions:
            context_parts.append(f"\n[PROJECT INSTRUCTIONS (GEMINI.md)]\n{self.project_instructions}")

        if dynamic_context:
            context_parts.append(f"\n[STRATEGIC CONTEXT]\n{dynamic_context}")

        if context_needed.lower() == "all files" or context_needed.lower() == "all":
            for filepath, content in self.file_contents.items():
                context_parts.append(f"\n--- FILE: {filepath} ---\n{content[:3000]}")
        elif context_needed and context_needed.lower() not in ("none", ""):
            # Support comma-separated list of items: "file.py:class X, other.py"
            items = [i.strip() for i in context_needed.split(",")]
            for item in items:
                if ":" in item:
                    try:
                        path, query = item.split(":", 1)
                        snippet = self.get_snippet(path, query)
                        if snippet:
                            context_parts.append(f"\n--- SNIPPET FROM {path} ({query}) ---\n{snippet}")
                        else:
                            content = self.file_contents.get(path)
                            if content:
                                context_parts.append(f"\n--- FILE: {path} ---\n{content[:2000]}")
                    except ValueError:
                        pass
                else:
                    for filepath in self.file_contents:
                        if Path(filepath).name.lower() == item.lower() or filepath.lower() == item.lower():
                            context_parts.append(f"\n--- FILE: {filepath} ---\n{self.file_contents[filepath][:3000]}")
        
        if step.get("tool", "").lower() == "write_file" and not context_parts and self._write_order:
            for filepath in self._write_order[-2:]:
                content = self.file_contents.get(filepath, "")
                if content:
                    context_parts.append(f"\n--- RECENTLY WRITTEN: {filepath} ---\n{content[:2000]}")
        return "\n".join(context_parts)

    def mark_done_by_index(self, step_index: int):
        if 0 <= step_index < len(self.checklist):
            self.checklist[step_index]["done"] = True
            self.consecutive_failures = 0

    def mark_not_done_by_index(self, step_index: int):
        if 0 <= step_index < len(self.checklist):
            self.checklist[step_index]["done"] = False

    def record_failure(self, step_index: int, error: str):
        self.consecutive_failures += 1
        task = self.checklist[step_index]["task"] if step_index < len(self.checklist) else f"step {step_index+1}"
        self.failed_tasks.append({"task": task, "error": error[:300]})

    def record_reflection(self, step_index: int, qwen_fast: str, gemma_decision: str | None, action: str):
        self.reflection_log.append({
            "step": step_index,
            "qwen_fast_reflection": qwen_fast,
            "gemma_escalation": gemma_decision,
            "action_taken": action,
        })

    def store_command_result(self, cmd: str, result: dict[str, Any]):
        self.command_history.append({
            "cmd": cmd, "ok": bool(result.get("ok")), "code": result.get("code"),
            "stdout": (result.get("stdout") or "")[-2000:],
            "stderr": (result.get("stderr") or result.get("error") or "")[-2000:],
        })
        self.command_history = self.command_history[-8:]

    def get_recent_observations(self) -> str:
        if not self.command_history and not self.failed_tasks:
            return ""
        lines = ["\n[RECENT OBSERVATIONS]"]
        for item in self.command_history[-4:]:
            lines.append(f"COMMAND: {item.get('cmd')}")
            lines.append(f"OK: {item.get('ok')} CODE: {item.get('code')}")
            if item.get("stdout"):
                lines.append(f"STDOUT:\n{item['stdout']}")
            if item.get("stderr"):
                lines.append(f"STDERR:\n{item['stderr']}")
        for failure in self.failed_tasks[-3:]:
            lines.append(f"- {failure.get('task')}: {failure.get('error')}")
        return "\n".join(lines)

    def assemble_reflection_context(self, step_index: int, result_summary: str, current_tool: str = "") -> dict:
        """Build structured context for Gemma reflection with session + relevance filtering."""
        plan_step = self.structured_plan[step_index]
        target = plan_step.get("target", "").strip()
        context_needed = plan_step.get("context_needed", "").lower().strip()

        relevant_paths = set()
        if target:
            relevant_paths.add(target)
            for fp in self.file_contents:
                if Path(fp).name == target or fp == target:
                    relevant_paths.add(fp)
        if context_needed and context_needed not in ("none", ""):
            for fp in self.file_contents:
                if Path(fp).name.lower() in context_needed or fp.lower() in context_needed:
                    relevant_paths.add(fp)
        if current_tool == "run_command" and "unittest" in target.lower():
            for fp in self.file_contents:
                if fp.endswith(".py") and "test_" not in fp:
                    relevant_paths.add(fp)
                if "test_" in Path(fp).name:
                    relevant_paths.add(fp)

        prior_events = self.history.get_events_for_files(relevant_paths, session_only=self.flags.session_filter)
        if not prior_events:
            prior_events = self.history.get_events_for_step(step_index, session_only=self.flags.session_filter)

        context = {
            "user_goal": self.user_goal,
            "plan": [
                {"index": i, "tool": s.get("tool"), "prompt": s.get("qwen_prompt", "")[:200]}
                for i, s in enumerate(self.structured_plan)
            ],
            "current_step": {
                "index": step_index,
                "tool": self.structured_plan[step_index].get("tool", ""),
                "prompt": self.structured_plan[step_index].get("qwen_prompt", ""),
            },
            "result_summary": result_summary[:1000],
            "retry_count": self.history.get_retry_count(step_index),
            "prior_reflections": [
                {"decision": r.get("gemma_decision", ""), "action": r.get("action_taken", "")}
                for r in self.reflection_log[-3:]
            ],
            "prior_events": [
                {"type": e.get("type"), "tool": e.get("tool"), "result_ok": e.get("result_ok"),
                 "summary": e.get("result_summary", "")[:300]}
                for e in prior_events[-5:]
            ],
            "relevant_files": {},
            "file_versions": {},
            "adjacent_steps": {
                "prev": self.structured_plan[step_index-1].get("qwen_prompt", "") if step_index > 0 else None,
                "next": self.structured_plan[step_index+1].get("qwen_prompt", "") if step_index+1 < len(self.structured_plan) else None,
            },
        }

        for path in relevant_paths:
            if path in self.file_contents:
                context["relevant_files"][path] = self.file_contents[path][:1500]
            versions = self.history.get_file_versions(path)
            if versions:
                context["file_versions"][path] = [{"seq": v["seq"], "content": v["content"][:1500]} for v in versions[-2:]]

        # ── Scanner augmentation ──
        if target:
            dependents = self.scanner.get_dependents(target)
            if dependents:
                context["dependents"] = dependents
        context["workspace_overview"] = self.scanner.get_context_block()

        return context

    def get_remaining(self) -> list[str]:
        return [c["task"] for c in self.checklist if not c["done"]]

    def get_done(self) -> list[str]:
        return [c["task"] for c in self.checklist if c["done"]]

    def get_progress_block(self) -> str:
        lines = ["[PROGRESS]\n"]
        done = self.get_done()
        remaining = self.get_remaining()
        if done:
            lines.append(f"Done: {len(done)}/{len(self.checklist)}")
        if remaining:
            lines.append(f"Remaining: {len(remaining)}")
        return "\n".join(lines)

    def clear(self):
        self.user_goal = ""
        self.structured_plan = []
        self.checklist = []
        self.file_contents = {}
        self.files_read = {}
        self.failed_tasks = []
        self.command_history = []
        self.consecutive_failures = 0
        self._write_order = []
        self.reflection_log = []

    def save(self):
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "user_goal": self.user_goal,
                "structured_plan": self.structured_plan,
                "checklist": self.checklist,
                "failed_tasks": self.failed_tasks,
                "files_read": self.files_read,
                "command_history": self.command_history,
                "file_contents": self.file_contents,
                "write_order": self._write_order,
                "reflection_log": self.reflection_log,
            }
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
            print("💾 State saved.")
        except Exception as e:
            print(f"❌ Failed to save: {e}")

    def load(self):
        try:
            if self.state_file.exists():
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.user_goal = data.get("user_goal", "")
                    self.structured_plan = data.get("structured_plan", [])
                    self.checklist = data.get("checklist", [])
                    self.failed_tasks = data.get("failed_tasks", [])
                    self.files_read = data.get("files_read", {})
                    self.command_history = data.get("command_history", [])
                    self.file_contents = data.get("file_contents", {})
                    self._write_order = data.get("write_order", [])
                    self.reflection_log = data.get("reflection_log", [])
                    self.consecutive_failures = 0
                print(f"📂 Loaded plan: {self.user_goal[:60]}... ({len(self.get_done())}/{len(self.checklist)} done)")
                return True
        except Exception as e:
            print(f"❌ Failed to load: {e}")
        return False
