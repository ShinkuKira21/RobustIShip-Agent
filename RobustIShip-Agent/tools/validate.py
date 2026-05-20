"""File validation — syntax checks, pre-flight anti-pattern detection."""

import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

from tools.normalize import is_within_root


def validate_written_file(root: Path, rel_path: str, pre_flight: bool = True) -> dict:
    path = (root / rel_path).resolve()
    if not is_within_root(root, path):
        return {"ok": False, "error": f"Refusing to validate outside root: {rel_path}"}
    if not path.exists():
        return {"ok": False, "error": f"File missing after write: {rel_path}"}

    ext = path.suffix.lower()
    try:
        if ext == ".py":
            source = path.read_text(encoding="utf-8")
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                compile(source, str(path), "exec")
                if w and any("invalid escape sequence" in str(warning.message) for warning in w):
                    return {
                        "ok": False, "code": 1, "stdout": "",
                        "stderr": f"Invalid escape sequence in {rel_path}: {w[0].message}",
                    }
            smoke = subprocess.run(
                [sys.executable, "-c", f"import py_compile; py_compile.compile({str(path)!r}, doraise=True)"],
                capture_output=True, text=True, timeout=15, cwd=str(root),
            )
            if smoke.returncode != 0:
                return {"ok": False, "code": 1, "stdout": "", "stderr": smoke.stderr[:500]}

            if pre_flight and path.name.startswith("test_"):
                # Anti-pattern: import inside function body
                if re.search(r'^\s+import\s+\w+\s*$', source, re.MULTILINE):
                    return {
                        "ok": False, "code": 1, "stdout": "",
                        "stderr": "Test file imports a module inside a function body. "
                                  "Import modules at the top of the file and call functions directly.",
                    }
                # Anti-pattern: accessing .args on an imported module
                if re.search(r'\b\w+\.args\b', source):
                    return {
                        "ok": False, "code": 1, "stdout": "",
                        "stderr": "Test file accesses '.args' on an imported module. "
                                  "args is local to __main__; test argparse by calling the parsing function directly.",
                    }
                # Anti-pattern: sys.argv manipulation (anywhere in file)
                if re.search(r'sys\.argv\s*=', source):
                    return {
                        "ok": False, "code": 1, "stdout": "",
                        "stderr": "Test file modifies sys.argv. Import the module at the top and call "
                                  "its functions directly with test arguments.",
                    }
                test_count = len(re.findall(r"^\s*def\s+test_", source, flags=re.MULTILINE))
                unittest_assertions = len(re.findall(r"\bself\.assert\w+\s*\(", source))
                pytest_assertions = len(re.findall(r"^\s+assert\s", source, flags=re.MULTILINE))
                if test_count and unittest_assertions == 0 and pytest_assertions == 0:
                    return {"ok": False, "code": 1, "stdout": "", "stderr": f"Test file has {test_count} test methods but no assertions found"}
                if "Add assertions" in source or "TODO" in source:
                    return {"ok": False, "code": 1, "stdout": "", "stderr": "Test file still contains placeholder assertion comments"}
                if unittest_assertions and "import unittest" not in source and "from unittest" not in source:
                    return {"ok": False, "code": 1, "stdout": "", "stderr": "Test file uses self.assert* but missing unittest import"}
            return {"ok": True, "code": 0, "stdout": "", "stderr": ""}
        if ext == ".json":
            json.loads(path.read_text(encoding="utf-8"))
            return {"ok": True, "code": 0, "stdout": "", "stderr": ""}
        if ext == ".md":
            content = path.read_text(encoding="utf-8")
            if content.count("```") % 2 != 0:
                return {"ok": False, "code": 1, "stdout": "", "stderr": "Markdown code fences are unbalanced"}
            return {"ok": True, "code": 0, "stdout": "", "stderr": ""}
    except Exception as e:
        return {"ok": False, "code": 1, "stdout": "", "stderr": str(e)}
    return {"ok": True, "code": 0, "stdout": "", "stderr": ""}