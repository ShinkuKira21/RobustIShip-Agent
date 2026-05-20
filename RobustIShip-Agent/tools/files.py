"""File operations — read, write, edit."""

from pathlib import Path
from tools.normalize import is_within_root


def read_file(root: Path, rel_path: str, max_lines: int = 500, start_line: int = 1, end_line: int = None) -> dict:
    path = (root / rel_path).resolve()
    if not is_within_root(root, path):
        return {"error": f"Refusing to read outside root: {rel_path}"}
    if not path.exists():
        return {"error": f"File not found: {rel_path}"}
    if path.is_dir():
        return {"error": f"Cannot read directory: {rel_path}. Use ls to list contents."}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        total = len(lines)
        
        # Adjust 1-based start_line to 0-based index
        start_idx = max(0, start_line - 1)
        if end_line is None:
            end_idx = min(total, start_idx + max_lines)
        else:
            end_idx = min(total, end_line)
        
        subset = lines[start_idx:end_idx]
        content = "\n".join(subset)
        
        if end_line is None and total > (start_idx + max_lines):
            content += f"\n... (truncated, {total} total lines)"
        
        return {
            "ok": True, 
            "path": rel_path, 
            "content": content, 
            "lines": total,
            "start_line": start_idx + 1,
            "end_line": end_idx
        }
    except Exception as e:
        return {"error": f"Failed to read: {e}"}


def write_file(root: Path, rel_path: str, content: str) -> None:
    path = (root / rel_path).resolve()
    if not is_within_root(root, path):
        raise PermissionError(f"Refusing to write outside root: {rel_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def edit_file(root: Path, rel_path: str, old_str: str, new_str: str) -> dict:
    path = (root / rel_path).resolve()
    if not is_within_root(root, path):
        return {"error": f"Refusing to edit outside root: {rel_path}"}
    if not path.exists():
        return {"error": f"File not found: {rel_path}. Use write_file to create it first."}
    try:
        content = path.read_text(encoding="utf-8")
        count = content.count(old_str)
        if count == 0:
            return {"error": f"edit_file: old_str not found in {rel_path}. Verify the exact snippet."}
        if count > 1:
            return {"error": f"edit_file: old_str matches {count} locations in {rel_path}. Make snippet more specific."}
        new_content = content.replace(old_str, new_str, 1)
        path.write_text(new_content, encoding="utf-8")
        return {"ok": True, "path": rel_path}
    except Exception as e:
        return {"error": f"edit_file failed: {e}"}