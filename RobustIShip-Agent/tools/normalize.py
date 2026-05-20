"""Action normalization — maps Qwen's free-form JSON to canonical tool calls."""

import shlex
from pathlib import Path

from config import ALLOWED_TOOLS
from memory import fix_memory
from utils.path_utils import _fix_absolute_path


TOOL_ALIASES = {
    "create_file": "write_file", "write": "write_file", "save_file": "write_file",
    "save": "write_file", "create": "write_file", "new_file": "write_file",
    "update_file": "write_file", "modify_file": "write_file",
    "edit_file": "edit_file", "patch_file": "edit_file", "patch": "edit_file",
    "str_replace": "edit_file", "replace_in_file": "edit_file",
    "exec": "run_command", "command": "run_command", "shell": "run_command",
    "bash": "run_command", "sh": "run_command", "execute": "run_command",
    "run": "run_command", "cmd": "run_command", "terminal": "run_command",
    "list_files": "run_command", "ls": "run_command", "dir": "run_command",
    "list": "run_command", "find": "run_command", "tree": "run_command",
    "grep": "run_command", "head": "run_command", "tail": "run_command",
    "wc": "run_command", "mkdir": "run_command",
    "cat": "read_file", "view": "read_file", "show": "read_file",
    "open": "read_file", "read": "read_file",
}


def is_within_root(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def normalize_action(action: dict, root: Path | None = None) -> tuple[dict | None, str | None]:
    if not isinstance(action, dict):
        return None, "Action must be a JSON object"
    if "final" in action:
        if not isinstance(action.get("final"), str):
            return None, '"final" must be a string'
        return {"final": action["final"]}, None

    tool = action.get("tool")
    args = action.get("args") or {}

    if tool is None or not isinstance(tool, str):
        for k in args.keys():
            if '/' in k or '.' in k:
                return {"tool": "write_file", "args": {"path": k, "content": args[k]}}, None
        return None, "Tool must be string"

    tool = TOOL_ALIASES.get(tool, tool)

    if tool == "read_file":
        path = args.get("path") or args.get("file") or args.get("file_path")
        if not isinstance(path, str) or not path:
            return None, '"read_file" requires args.path string'
        if root:
            path = _fix_absolute_path(path, root)
        
        normalized_args = {"path": path}
        if "start_line" in args:
            normalized_args["start_line"] = int(args["start_line"])
        if "end_line" in args:
            normalized_args["end_line"] = int(args["end_line"])
            
        return {"tool": "read_file", "args": normalized_args}, None

    if tool == "grep_search":
        pattern = args.get("pattern") or args.get("search") or args.get("query")
        if not isinstance(pattern, str) or not pattern:
            return None, '"grep_search" requires args.pattern string'
        include = args.get("include") or args.get("glob") or "*"
        return {"tool": "grep_search", "args": {"pattern": pattern, "include": include}}, None

    if tool == "edit_file":
        path = args.get("path") or args.get("file") or args.get("file_path")
        old_str = args.get("old_str") or args.get("old") or args.get("search") or ""
        new_str = args.get("new_str") or args.get("new") or args.get("replace") or ""
        if not isinstance(path, str) or not path:
            return None, '"edit_file" requires args.path string'
        if not isinstance(old_str, str) or not old_str:
            return None, '"edit_file" requires args.old_str string'
        if root:
            path = _fix_absolute_path(path, root)
        return {"tool": "edit_file", "args": {"path": path, "old_str": old_str, "new_str": new_str}}, None

    if tool in {"create_folder", "create_dir", "create_directory", "mkdir"}:
        path = args.get("path") or args.get("dir") or args.get("directory")
        if isinstance(path, str) and path:
            return {"tool": "run_command", "args": {"cmd": f"mkdir -p {shlex.quote(path)}"}}, None

    if tool == "write_file":
        path = args.get("path") or args.get("file") or args.get("file_path")
        if not path:
            for k in args.keys():
                if "/" in k or "." in k:
                    path = k
                    break
        content = args.get("content") or args.get("text") or ""
        if not content:
            for k, v in args.items():
                if k != path:
                    content = v
                    break
        if not isinstance(path, str) or not path:
            return None, '"write_file" requires args.path string'
        if root:
            path = _fix_absolute_path(path, root)
        if not isinstance(content, str):
            content = str(content)
        return {"tool": "write_file", "args": {"path": path, "content": content}}, None

    if tool == "run_command":
        cmd = args.get("cmd") or args.get("command")
        if not isinstance(cmd, str) or not cmd.strip():
            return None, '"run_command" requires args.cmd string'
        cmd = fix_memory.apply_fixes(cmd)
        return {"tool": "run_command", "args": {"cmd": cmd}}, None

    return None, f'Unknown tool: {tool} (allowed: {sorted(ALLOWED_TOOLS)})'