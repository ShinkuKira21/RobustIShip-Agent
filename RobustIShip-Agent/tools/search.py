"""Search tools for the agent."""

import subprocess
from pathlib import Path
from tools.normalize import is_within_root

def grep_search(root: Path, pattern: str, include: str = "*") -> dict:
    """Search for a pattern in files using grep."""
    try:
        # Using -n for line numbers, -r for recursive, -I to ignore binary
        # --include for filtering
        cmd = ["grep", "-nrI", f"--include={include}", pattern, "."]
        
        proc = subprocess.run(
            cmd, cwd=str(root), text=True, capture_output=True, timeout=30
        )
        
        if proc.returncode == 0 or (proc.returncode == 1 and not proc.stderr):
            stdout = proc.stdout
            lines = stdout.splitlines()
            count = len(lines)
            
            if count > 100:
                stdout = "\n".join(lines[:100]) + f"\n... (truncated, {count} total matches)"
                
            return {
                "ok": True,
                "matches": stdout,
                "total_matches": count
            }
        else:
            return {"error": f"Grep failed: {proc.stderr}"}
            
    except Exception as e:
        return {"error": f"Search failed: {e}"}
