"""Workspace pre-scanner — builds file map while Gemma loads."""

import re
import threading
from pathlib import Path


class WorkspaceScanner:
    """Scans workspace files in a background thread during Gemma load.
    
    Extracts imports, classes, functions, and file metadata.
    Runs once at startup, then incrementally updated on write/edit.
    """

    def __init__(self, root: Path):
        self.root = root
        self.files: dict[str, dict] = {}  # rel_path -> metadata
        self._lock = threading.Lock()
        self._ready = False

    def start_async(self):
        """Launch scan in background thread. Call immediately before Gemma.load()."""
        t = threading.Thread(target=self._scan, daemon=True)
        t.start()
        return t

    def _scan(self):
        """Scan all workspace files. Skips .robustIship and hidden dirs."""
        try:
            for path in self.root.rglob("*"):
                if self._should_skip(path):
                    continue
                if path.is_file():
                    try:
                        rel = str(path.relative_to(self.root))
                        content = path.read_text(encoding="utf-8", errors="replace")
                        self.files[rel] = self._analyze(rel, content)
                    except Exception:
                        pass
        finally:
            self._ready = True

    def _should_skip(self, path: Path) -> bool:
        """Skip hidden dirs, .git, .robustIship, node_modules, __pycache__."""
        skip_dirs = {".git", ".robustIship", "node_modules", "__pycache__", ".venv", "venv", ".robustIship"}
        for part in path.parts:
            if part in skip_dirs or part.startswith("."):
                return True
        return False

    def _analyze(self, rel_path: str, content: str) -> dict:
        """Extract metadata from file content."""
        return {
            "path": rel_path,
            "size": len(content),
            "lines": content.count("\n") + 1,
            "imports": self._extract_imports(content, rel_path),
            "functions": self._extract_functions(content),
            "classes": self._extract_classes(content),
            "summary": content[:500],
        }

    def _extract_imports(self, content: str, rel_path: str) -> list[str]:
        """Extract imports, resolving relative to project."""
        imports = []
        for match in re.finditer(
            r"^(?:from\s+(\S+)\s+import\s+\S+|import\s+(\S+))", 
            content, re.MULTILINE
        ):
            mod = match.group(1) or match.group(2)
            if mod and not mod.startswith("."):
                imports.append(mod)
        return imports

    def _extract_functions(self, content: str) -> list[str]:
        """Extract top-level function names."""
        return re.findall(r"^def\s+(\w+)", content, re.MULTILINE)

    def _extract_classes(self, content: str) -> list[str]:
        """Extract class names."""
        return re.findall(r"^class\s+(\w+)", content, re.MULTILINE)

    def upsert(self, rel_path: str, content: str):
        """Update index after write/edit."""
        with self._lock:
            self.files[rel_path] = self._analyze(rel_path, content)

    def remove(self, rel_path: str):
        """Remove file from index."""
        with self._lock:
            self.files.pop(rel_path, None)

    def query(self, text: str, top_k: int = 10) -> list[dict]:
        """Keyword search across file metadata."""
        results = []
        terms = set(text.lower().split())
        with self._lock:
            for path, meta in self.files.items():
                searchable = f"{path} {meta.get('summary', '')} {' '.join(meta.get('functions', []))} {' '.join(meta.get('classes', []))}"
                score = sum(1 for t in terms if t in searchable.lower()) / len(terms)
                if score > 0:
                    results.append({"score": score, **meta})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def get_importers(self, module_name: str) -> list[str]:
        """Find all files that import a given module."""
        importers = []
        with self._lock:
            for path, meta in self.files.items():
                if module_name in meta.get("imports", []):
                    importers.append(path)
        return importers

    def get_dependents(self, rel_path: str) -> list[str]:
        """Find files that depend on this file (import it)."""
        # Extract module name from path
        module = rel_path.replace("/", ".").replace(".py", "")
        return self.get_importers(module)

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def is_ready(self) -> bool:
        return self._ready

    def get_context_block(self) -> str:
        """Return a formatted context block for Gemma/Qwen."""
        if not self.files:
            return "[WORKSPACE: empty — no files exist yet. This is a NEW project. Plan concrete steps directly.]"
        lines = [f"[WORKSPACE: {len(self.files)} files]"]
        for path, meta in sorted(self.files.items()):
            funcs = ", ".join(meta.get("functions", [])[:5])
            classes = ", ".join(meta.get("classes", [])[:5])
            detail = ""
            if funcs:
                detail += f" functions=[{funcs}]"
            if classes:
                detail += f" classes=[{classes}]"
            lines.append(f"  {path} ({meta['lines']} lines{detail})")
        return "\n".join(lines)