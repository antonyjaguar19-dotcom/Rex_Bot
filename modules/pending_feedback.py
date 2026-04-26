"""
Claw Bot — Pending Feedback Store
Persists scripts waiting for user revision notes across bot restarts.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("claw_bot.pending_feedback")

# Single JSON file. Simple, atomic enough for our scale.
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
STORE_PATH = PROJECT_ROOT / "05_Config" / "pending_feedback.json"


def _load_all() -> dict:
    if not STORE_PATH.exists():
        return {}
    try:
        return json.loads(STORE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        log.error(f"Could not read pending_feedback.json ({e}); starting fresh.")
        return {}


def _save_all(data: dict):
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def add(script: dict, owner_id: int, channel_id: int, reason: str = "paused") -> None:
    """Register a script as awaiting feedback."""
    data = _load_all()
    data[script["script_id"]] = {
        "script": script,
        "owner_id": owner_id,
        "channel_id": channel_id,
        "paused_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,   # 'paused' | 'timeout'
    }
    _save_all(data)
    log.info(f"Pending feedback stored for {script['script_id']} (reason={reason}).")


def get(script_id: str) -> Optional[dict]:
    """Return the full entry or None."""
    return _load_all().get(script_id)


def remove(script_id: str) -> bool:
    """Remove an entry. Returns True if it existed."""
    data = _load_all()
    if script_id in data:
        del data[script_id]
        _save_all(data)
        log.info(f"Pending feedback removed: {script_id}")
        return True
    return False


def list_all() -> list[dict]:
    """Return all pending entries, most recent first."""
    data = _load_all()
    entries = []
    for sid, entry in data.items():
        entries.append({
            "script_id": sid,
            "title": entry["script"].get("title", "Untitled"),
            "theme": entry["script"].get("theme", ""),
            "owner_id": entry["owner_id"],
            "paused_at": entry["paused_at"],
            "reason": entry["reason"],
        })
    entries.sort(key=lambda e: e["paused_at"], reverse=True)
    return entries


def count() -> int:
    return len(_load_all())