"""
Claw Bot — shared GPU job lock

One GPU, three job sources: Discord commands, the NiceGUI dashboard
(worker threads in the same process), and the daily scheduler. Before this
lock, each had its own "busy" notion (dashboard S.busy, Discord none), so
two front-ends could start renders simultaneously — VRAM OOM and clobbered
output files. Think render-farm node allocation: one node, one job.

Semantics:
- Reentrant for the SAME thread only. All Discord handlers run on the
  bot's asyncio event-loop thread, so chained pipelines (approve script ->
  storyboard -> video) re-acquire without deadlocking. The dashboard runs
  on its own thread, so Discord vs dashboard exclusion is thread-vs-thread.
- release() may be called from a DIFFERENT thread than acquire() — the
  dashboard claims the lock in a UI click handler and releases in the
  worker thread's finally block. (This is why threading.RLock isn't used:
  it forbids cross-thread release.)
- An on-disk marker (05_Config/job_lock.json) records who holds the lock
  (label, pid, started). It blocks a SECOND bot process from starting GPU
  work, and lets !status / the dashboard show "busy with X". A marker
  from a dead or ancient pid is treated as stale and stolen.

Usage:
    if not job_lock.acquire("discord:storyboard"):
        ...tell the user who holds it: job_lock.holder_label()...
    try:
        ...GPU work...
    finally:
        job_lock.release()
"""

import ctypes
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

from modules.file_utils import atomic_write_json

log = logging.getLogger("claw_bot.job_lock")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
LOCK_PATH = PROJECT_ROOT / "05_Config" / "job_lock.json"

# A pipeline run (script + 10 shots of video + upscale) can take a couple of
# hours; past this the marker is assumed orphaned even if the pid matches.
STALE_SEC = 4 * 3600

_guard = threading.Lock()
_depth = 0
_holder_ident: int | None = None
_holder_label = ""


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _read_marker() -> dict | None:
    if not LOCK_PATH.exists():
        return None
    try:
        return json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _marker_is_stale(marker: dict) -> bool:
    age = time.time() - float(marker.get("started", 0))
    if age > STALE_SEC:
        return True
    return not _pid_alive(int(marker.get("pid", -1)))


def current() -> dict | None:
    """Marker of the active job, or None when idle. Stale markers count as idle."""
    marker = _read_marker()
    if marker is None:
        return None
    if marker.get("pid") != os.getpid() and _marker_is_stale(marker):
        return None
    return marker


def holder_label() -> str:
    with _guard:
        if _depth > 0:
            return _holder_label
    marker = current()
    return marker.get("label", "unknown job") if marker else "idle"


def acquire(label: str) -> bool:
    """Try to claim the GPU. Non-blocking; False means someone else has it."""
    global _depth, _holder_ident, _holder_label
    with _guard:
        if _depth > 0:
            if threading.get_ident() != _holder_ident:
                return False
            _depth += 1   # same thread chaining pipelines
            return True
        # Idle in this process — but another PROCESS may hold the marker
        # (e.g. a second bot instance started by accident).
        marker = _read_marker()
        if marker and marker.get("pid") != os.getpid():
            if not _marker_is_stale(marker):
                return False
            log.warning(f"Stealing stale job lock from pid {marker.get('pid')} "
                        f"({marker.get('label')})")
        _depth = 1
        _holder_ident = threading.get_ident()
        _holder_label = label
        try:
            atomic_write_json(LOCK_PATH, {
                "label": label,
                "pid": os.getpid(),
                "started": time.time(),
            })
        except Exception as e:
            log.warning(f"Could not write job lock marker: {e}")
        log.info(f"Job lock acquired: {label}")
        return True


def release() -> None:
    """Release one level. Safe to call from a different thread than acquire()."""
    global _depth, _holder_ident, _holder_label
    with _guard:
        if _depth == 0:
            log.warning("job_lock.release() called while not held")
            return
        _depth -= 1
        if _depth == 0:
            _holder_ident = None
            _holder_label = ""
            try:
                LOCK_PATH.unlink(missing_ok=True)
            except Exception as e:
                log.warning(f"Could not remove job lock marker: {e}")
            log.info("Job lock released.")
