"""
Claw Bot — Generation Metadata Tracker

Records each script/storyboard/video generation with full settings + timing.
Last 10 jobs kept in memory; full history persisted to JSON.
Dashboard reads from here to display recent generations.
"""

import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("claw_bot.generation_meta")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
META_FILE = PROJECT_ROOT / "05_Config" / "generation_history.json"
MAX_RECENT = 10
MAX_PERSISTED = 200

# In-memory recent buffer (newest first)
_RECENT: deque = deque(maxlen=MAX_RECENT)


def _load_persisted() -> list[dict]:
    if not META_FILE.exists():
        return []
    try:
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"Could not read generation_history.json: {e}")
        return []


def _save_persisted(data: list[dict]):
    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# Hydrate _RECENT from disk on import (so dashboard shows history across restarts)
_persisted = _load_persisted()
for entry in _persisted[-MAX_RECENT:]:
    _RECENT.appendleft(entry)


# ============================================================
# Public API
# ============================================================

class JobTimer:
    """Context manager: start/stop timing + capture VRAM peak."""

    def __init__(self, kind: str, script_id: str, settings: dict):
        self.kind = kind   # 'script' | 'storyboard' | 'video'
        self.script_id = script_id
        self.settings = settings or {}
        self._start = None
        self._vram_peak_mb = 0
        self._error: Optional[str] = None

    def __enter__(self):
        self._start = time.time()
        return self

    def update_vram_peak(self, used_mb: int):
        if used_mb > self._vram_peak_mb:
            self._vram_peak_mb = used_mb

    def set_error(self, err: str):
        self._error = err[:200]

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - (self._start or time.time())
        if exc_type is not None:
            self._error = f"{exc_type.__name__}: {str(exc)[:150]}"
        record({
            "kind": self.kind,
            "script_id": self.script_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "duration_sec": round(elapsed, 1),
            "vram_peak_mb": self._vram_peak_mb,
            "settings": self.settings,
            "error": self._error,
            "success": self._error is None,
        })
        return False  # don't swallow exceptions


def record(entry: dict):
    """Append a generation record. Front-loads in-memory deque, persists to disk."""
    entry.setdefault("ts", datetime.now(timezone.utc).isoformat())
    _RECENT.appendleft(entry)
    history = _load_persisted()
    history.append(entry)
    history = history[-MAX_PERSISTED:]
    _save_persisted(history)
    kind = entry.get("kind", "?")
    sid = entry.get("script_id", "?")
    dur = entry.get("duration_sec", "?")
    log.info(f"Generation logged: {kind} script={sid} took={dur}s")


def get_recent(n: int = MAX_RECENT) -> list[dict]:
    """Return the last n entries (newest first)."""
    return list(_RECENT)[:n]


def get_last_of_kind(kind: str) -> Optional[dict]:
    for entry in _RECENT:
        if entry.get("kind") == kind:
            return entry
    return None


def clear():
    _RECENT.clear()
    if META_FILE.exists():
        META_FILE.unlink()


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("Recent generations:")
    for r in get_recent():
        print(f"  - {r.get('kind')} {r.get('script_id')} {r.get('duration_sec')}s")
