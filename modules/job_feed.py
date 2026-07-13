"""
Claw Bot — Job Feed (Discord → Web)

The other half of the sync bridge. `sync_bridge` carries dashboard-started jobs
INTO Discord; this carries Discord-started jobs into the DASHBOARD, so a render
kicked off from your phone is visible in the browser and vice versa.

Shape of the problem: the dashboard already knows about its own jobs (they run in
its process and push to `State`). A `!facts` render is a different code path in
the same process — the dashboard's State never hears about it, so the browser sat
blank for 30 minutes while the bot was clearly working.

Design follows the project's on-disk-state pattern, and the split that keeps it
contention-free:
  - 05_Config/bot_job.json — the CURRENTLY running Discord job. The bot WRITES it;
    the dashboard only READS. (sync_events.json is the mirror image: web writes,
    bot reads.) Separate files, one writer each, no locking needed across
    front-ends.

It is a live snapshot, not a log: one job at a time (the GPU lock guarantees
that), a bounded tail of progress lines, and a `done` flag. Nothing to replay,
nothing to prune.
"""

import logging
import threading
import time
from pathlib import Path

from modules.file_utils import atomic_write_json

log = logging.getLogger("claw_bot.job_feed")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FEED_PATH = PROJECT_ROOT / "05_Config" / "bot_job.json"

MAX_LINES = 40
# A feed nobody has touched for this long is a crashed bot, not a slow render.
# (A Wan clip can take minutes, so this has to be generous.)
STALE_AFTER_SEC = 30 * 60

_LOCK = threading.Lock()
_STATE: dict = {}


def _write():
    try:
        atomic_write_json(FEED_PATH, _STATE)
    except Exception as e:
        log.debug(f"job feed write failed: {e}")


def begin(job_id: str, mode: str, label: str) -> None:
    """A Discord job started. Replaces whatever was there — one job at a time."""
    global _STATE
    with _LOCK:
        _STATE = {
            "job": job_id,
            "mode": mode,
            "label": label,
            "stage": "",
            "lines": [],
            "started": time.time(),
            "updated": time.time(),
            "done": False,
            "ok": True,
        }
        _write()


def push(line: str, stage: str = "") -> None:
    """A progress line. Cheap enough to call often — the file is small."""
    line = (line or "").strip()
    if not line:
        return
    with _LOCK:
        if not _STATE or _STATE.get("done"):
            return
        _STATE["lines"] = (_STATE["lines"] + [line])[-MAX_LINES:]
        if stage:
            _STATE["stage"] = stage
        _STATE["updated"] = time.time()
        _write()


def end(ok: bool = True, note: str = "") -> None:
    with _LOCK:
        if not _STATE:
            return
        _STATE["done"] = True
        _STATE["ok"] = bool(ok)
        _STATE["updated"] = time.time()
        if note:
            _STATE["lines"] = (_STATE["lines"] + [note])[-MAX_LINES:]
        _write()


def read() -> dict:
    """The live Discord job, or {} when there is none.

    A finished job stays readable for a couple of minutes so the dashboard can
    show how it ended instead of the panel just vanishing.
    """
    if not FEED_PATH.exists():
        return {}
    try:
        import json
        data = json.loads(FEED_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not data.get("job"):
        return {}
    age = time.time() - float(data.get("updated", 0))
    if data.get("done") and age > 120:
        return {}
    if not data.get("done") and age > STALE_AFTER_SEC:
        data["done"] = True
        data["ok"] = False
        data["lines"] = (data.get("lines") or []) + [
            "⚠️ no update in 30 minutes — the bot may have died mid-render"]
    return data


def is_running() -> bool:
    d = read()
    return bool(d) and not d.get("done")
