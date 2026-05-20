"""Persistent fix memory for command corrections."""

import json
from pathlib import Path


class FixMemory:
    def __init__(self):
        self.command_fixes = {}
        self.load_from_file()

    def load_from_file(self):
        try:
            fix_file = Path.home() / ".robustIship_fixes.json"
            if fix_file.exists():
                with open(fix_file, "r") as f:
                    self.command_fixes = json.load(f).get("command_fixes", {})
                    if self.command_fixes:
                        print(f"   📚 Loaded {len(self.command_fixes)} saved fixes")
        except Exception:
            pass

    def save_to_file(self):
        try:
            fix_file = Path.home() / ".robustIship_fixes.json"
            with open(fix_file, "w") as f:
                json.dump({"command_fixes": self.command_fixes}, f, indent=2)
        except Exception:
            pass

    def add_fix(self, original: str, fixed: str):
        pattern = original.split()[0] if original else original
        if pattern and pattern != (fixed.split()[0] if fixed else ""):
            self.command_fixes[pattern] = fixed.split()[0]
            print(f"   💾 Remembered: use '{fixed.split()[0]}' instead of '{pattern}'")
            self.save_to_file()

    def apply_fixes(self, cmd: str) -> str:
        result = cmd
        for pattern, replacement in self.command_fixes.items():
            if result.startswith(pattern + " ") or result == pattern:
                result = replacement + result[len(pattern):]
        return result


# Singleton instance
fix_memory = FixMemory()