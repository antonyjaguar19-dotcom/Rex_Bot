"""
Claw Bot — Sync Bridge (Web ↔ Discord)

One-way gap closer. Discord → Web already syncs because the dashboard polls
disk every 1.5s (regen writes new files → dashboard re-renders). The missing
direction is Web → Discord: when the dashboard regenerates a clip/frame, the
file on disk changes but Discord already posted the OLD file (cached on its
CDN) and never re-checks.

This module is the shared on-disk event queue that fixes that. The dashboard
calls emit(...) after a successful regen; the bot runs a background poller
(claw_bot._sync_bridge_loop) that reads new events and re-posts the updated
media to the matching Discord channel.

Design (matches the project's "shared on-disk JSON state" pattern):
  - 05_Config/sync_events.json — append-only-ish event log (capped). The
    dashboard WRITES this; the bot only READS it.
  - 05_Config/sync_cursor.json — last event id the bot has handled. The bot
    WRITES this; the dashboard never touches it.
Separate files = no cross-process write contention on one file.
"""

import json
import logging
import threading
import time
from pathlib import Path

log = logging.getLogger("claw_bot.sync_bridge")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
CONFIG_DIR = PROJECT_ROOT / "05_Config"
EVENTS_PATH = CONFIG_DIR / "sync_events.json"
CURSOR_PATH = CONFIG_DIR / "sync_cursor.json"

_LOCK = threading.Lock()
_MAX_EVENTS = 200  # keep the log bounded


def _atomic_write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_store() -> dict:
    if EVENTS_PATH.exists():
        try:
            d = json.loads(EVENTS_PATH.read_text(encoding="utf-8"))
            d.setdefault("events", [])
            d.setdefault("next_id", 1)
            return d
        except Exception as e:
            log.warning(f"sync_events.json unreadable ({e}); resetting")
    return {"events": [], "next_id": 1}


def emit(event_type: str, source: str = "dashboard", **data) -> int:
    """Append an event. Returns its id. Called from the dashboard thread."""
    with _LOCK:
        store = _read_store()
        ev_id = store["next_id"]
        store["events"].append({
            "id": ev_id,
            "type": event_type,
            "source": source,
            "ts": time.time(),
            "data": data,
        })
        store["next_id"] = ev_id + 1
        store["events"] = store["events"][-_MAX_EVENTS:]
        _atomic_write(EVENTS_PATH, store)
    log.info(f"emit {event_type} #{ev_id} {data}")
    return ev_id


def latest_id() -> int:
    """Highest event id currently on disk (0 if none). Used to skip backlog."""
    store = _read_store()
    return max((e["id"] for e in store["events"]), default=0)


def read_new(cursor: int) -> list[dict]:
    """All events with id > cursor, oldest first."""
    store = _read_store()
    return sorted((e for e in store["events"] if e["id"] > cursor),
                  key=lambda e: e["id"])


def get_cursor() -> int:
    if CURSOR_PATH.exists():
        try:
            return int(json.loads(CURSOR_PATH.read_text(encoding="utf-8")).get("cursor", 0))
        except Exception:
            return 0
    return 0


def set_cursor(cursor: int) -> None:
    _atomic_write(CURSOR_PATH, {"cursor": cursor})
