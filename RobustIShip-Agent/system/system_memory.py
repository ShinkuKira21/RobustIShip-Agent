"""System-level memory — broken commands and working workarounds.

Stores environment info and command fixes that compound across all projects.
Checked before any model call — cache hits are instant.

Located at ~/.robustIship/system_fixes.json
"""

import json
import os
import platform
import re
import shutil
from pathlib import Path


class SystemMemory:
    """Cross-project memory for system-level command fixes and environment.
    
    Two data sections:
    - environment: OS, shell, Python path, package manager
    - command_fixes: broken commands mapped to working workarounds
    - missing_tools: tools not installed, with alternatives
    """

    def __init__(self):
        self.memory_path = Path.home() / ".robustIship" / "system_fixes.json"
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()
        self._discover_environment()

    def _load(self) -> dict:
        if self.memory_path.exists():
            try:
                with open(self.memory_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "environment": {},
            "command_fixes": {},
            "missing_tools": {},
        }

    def save(self):
        try:
            with open(self.memory_path, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    # ── ENVIRONMENT ──

    def _discover_environment(self):
        env = self.data.setdefault("environment", {})
        if not env:
            env["os"] = platform.system().lower()
            env["shell"] = self._detect_shell()
            env["python_path"] = self._find_python()
            env["python_version"] = platform.python_version()
            env["has_sudo"] = self._check_sudo()
            env["package_manager"] = self._detect_package_manager()
            self.save()

    def _detect_shell(self) -> str:
        if platform.system() == "Windows":
            return os.environ.get("COMSPEC", "cmd.exe")
        return os.environ.get("SHELL", "bash")

    def _find_python(self) -> str:
        import sys
        return sys.executable

    def _check_sudo(self) -> bool:
        return shutil.which("sudo") is not None

    def _detect_package_manager(self) -> str:
        for pm in ["apt", "dnf", "yum", "pacman", "brew", "winget"]:
            if shutil.which(pm):
                return pm
        return "unknown"

    def get_python_path(self) -> str | None:
        return self.data.get("environment", {}).get("python_path")

    # ── COMMAND FIXES ──

    def resolve_command(self, cmd: str) -> tuple[str, bool]:
        """Check if this command has a known fix. Returns (resolved_cmd, was_fixed)."""
        base = cmd.split()[0] if cmd.split() else cmd
        fixes = self.data.get("command_fixes", {})

        if base in fixes:
            fix = fixes[base]["fix"]
            resolved = fix + cmd[len(base):]
            return resolved, True

        return cmd, False

    def record_command_fix(self, broken: str, working: str):
        """Record a broken command and its working alternative."""
        base_broken = broken.split()[0] if broken.split() else broken
        base_working = working.split()[0] if working.split() else working

        if base_broken == base_working:
            return  # Same command, not a fix

        fixes = self.data.setdefault("command_fixes", {})
        if base_broken in fixes:
            fixes[base_broken]["count"] += 1
        else:
            fixes[base_broken] = {
                "fix": base_working,
                "count": 1,
            }
        self.save()

    # ── MISSING TOOLS ──

    def check_tool(self, tool_name: str) -> dict:
        """Check if a tool is available. Returns {available, alternatives}."""
        missing = self.data.get("missing_tools", {})
        if tool_name in missing:
            return {"available": False, "alternatives": missing[tool_name]}

        if shutil.which(tool_name):
            return {"available": True, "alternatives": []}

        return {"available": False, "alternatives": []}

    def record_missing_tool(self, tool_name: str, alternatives: list[str]):
        """Record a missing tool and its alternatives."""
        self.data.setdefault("missing_tools", {})[tool_name] = alternatives
        self.save()

    # ── QUERYING ──

    def get_environment_context(self) -> str:
        env = self.data.get("environment", {})
        lines = [
            f"OS: {env.get('os', 'unknown')}",
            f"Shell: {env.get('shell', 'unknown')}",
            f"Python: {env.get('python_path', 'unknown')} ({env.get('python_version', 'unknown')})",
            f"Package Manager: {env.get('package_manager', 'unknown')}",
            f"Has sudo: {env.get('has_sudo', False)}",
        ]
        return "\n".join(lines)


# Singleton
system_memory = SystemMemory()