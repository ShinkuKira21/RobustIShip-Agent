"""Append-only event log + file versioning with session tracking."""

import json
import time
from pathlib import Path


class HistoryMap:    
    def __init__(self, root: Path, log_events: bool = False):
        self.root = root
        self.log_events = log_events
        self.history_dir = root / ".robustIship" / "history"
        self.events_file = self.history_dir / "events.jsonl"
        self.file_versions_dir = self.history_dir / "file_versions"
        self._seq = 0
        self._session_start_seq = 0
    
    @property
    def current_seq(self) -> int:
        return self._seq

    def init(self):
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.file_versions_dir.mkdir(parents=True, exist_ok=True)
        if self.events_file.exists():
            try:
                with open(self.events_file, "r") as f:
                    for line in f:
                        if line.strip():
                            self._seq = max(self._seq, json.loads(line).get("seq", 0))
            except Exception:
                pass
        self._session_start_seq = self._seq

    def mark_session_start(self):
        self._session_start_seq = self._seq

    def record(self, event: dict):
        if not self.log_events:
            return
        self._seq += 1
        event["seq"] = self._seq
        event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            with open(self.events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def snapshot_file(self, rel_path: str, content: str):
        if not self.log_events:
            return
        try:
            file_dir = self.file_versions_dir / rel_path
            file_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = file_dir / f"{self._seq:04d}.txt"
            snapshot_path.write_text(content, encoding="utf-8")
        except Exception:
            pass

    def get_file_versions(self, rel_path: str) -> list[dict]:
        file_dir = self.file_versions_dir / rel_path
        if not file_dir.exists():
            return []
        versions = []
        for snapshot in sorted(file_dir.glob("*.txt")):
            try:
                content = snapshot.read_text(encoding="utf-8")[:1500]
                versions.append({"seq": int(snapshot.stem), "content": content})
            except Exception:
                pass
        return versions

    def get_events_for_files(self, paths: set, session_only: bool = True) -> list[dict]:
        events = []
        if not self.events_file.exists() or not paths:
            return events
        try:
            with open(self.events_file, "r") as f:
                for line in f:
                    if line.strip():
                        evt = json.loads(line)
                        if session_only and evt.get("seq", 0) < self._session_start_seq:
                            continue
                        if set(evt.get("files_changed", [])) & paths:
                            events.append(evt)
        except Exception:
            pass
        return events

    def get_events_for_step(self, step_index: int, session_only: bool = True) -> list[dict]:
        events = []
        if not self.events_file.exists():
            return events
        try:
            with open(self.events_file, "r") as f:
                for line in f:
                    if line.strip():
                        evt = json.loads(line)
                        if session_only and evt.get("seq", 0) < self._session_start_seq:
                            continue
                        if evt.get("step_index") == step_index:
                            events.append(evt)
        except Exception:
            pass
        return events

    def get_retry_count(self, step_index: int) -> int:
        return len([e for e in self.get_events_for_step(step_index) if e.get("type") == "retry"])