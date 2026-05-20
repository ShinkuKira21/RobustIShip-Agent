"""Path manipulation utilities."""

from pathlib import Path


def _fix_absolute_path(path: str, root: Path) -> str:
    if path.startswith("/"):
        try:
            return str(Path(path).resolve().relative_to(root.resolve()))
        except Exception:
            pass
    return path