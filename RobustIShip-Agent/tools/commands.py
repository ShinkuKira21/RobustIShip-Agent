"""Command execution with system cache, safety checks, and auto-fixes."""

import os
import subprocess
from pathlib import Path

from system.system_memory import system_memory
from config import DANGEROUS_COMMANDS
from flags import FeatureFlags


def run_command(root: Path, cmd: str, timeout_s: int = 1800, flags: FeatureFlags = None) -> dict:
    # Safety check
    if any(x in f" {cmd.strip()} ".lower() for x in DANGEROUS_COMMANDS):
        return {"ok": False, "code": 126, "stdout": "", "stderr": "Refused dangerous command"}

    original_cmd = cmd

    # Tier 1: System cache — check for known command fixes
    cmd, was_cached = system_memory.resolve_command(cmd)

    # Tier 2: Tool availability — use cached alternatives if tool missing
    base_tool = cmd.split()[0] if cmd.split() else cmd
    tool_check = system_memory.check_tool(base_tool)
    if not tool_check.get("available", True):
        alternatives = tool_check.get("alternatives", [])
        if alternatives:
            cmd = cmd.replace(base_tool, alternatives[0], 1)

    # Execute
    proc = subprocess.run(
        cmd, cwd=str(root), shell=True, text=True,
        capture_output=True, timeout=timeout_s, env=os.environ.copy(),
    )

    result = {
        "ok": proc.returncode == 0,
        "code": proc.returncode,
        "stdout": proc.stdout[-20000:],
        "stderr": proc.stderr[-20000:],
    }

    # Record fixes that worked
    if was_cached and result["ok"]:
        system_memory.record_command_fix(broken=original_cmd, working=cmd)

    return result