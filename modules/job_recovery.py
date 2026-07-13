"""
Claw Bot — interrupted-job recovery

A render can run for hours. If the PC is shut down, the bot is killed, or the
power drops mid-job, the work simply vanished: no record that anything was in
flight, no way back to it. You came back to a half-written 04_Outputs and had
to guess what stage it died at.

This module keeps a small on-disk register of jobs that are RUNNING right now.
A job that is registered but whose process is dead was interrupted. On the next
boot each mode's tab shows a banner offering **Resume** or **Discard**.

  begin(mode, label, stage=..., context={...})  -> job_id   (register)
  checkpoint(job_id, stage="video")                         (record progress)
  finish(job_id)                                            (deregister)
  list_interrupted()                                        (dead pids)
  discard(job_id)

Design notes:
- The register is the SOURCE OF TRUTH for "was something running", but never for
  the artifacts themselves — those already live on disk (script/facts/song/horror
  JSON, storyboard PNGs, clips). Resume reloads that state and re-enters the
  pipeline at the recorded STAGE. It restarts that stage, not the exact frame:
  half-written outputs are re-made, finished stages are not.
- "Interrupted" == registered AND its pid is not alive. A record whose pid is
  alive belongs to a job running right now (possibly in another bot process) and
  is left alone.
- finish() is called from the worker's finally block, so a job that merely
  RAISED is deregistered too. Only a dead process leaves a record behind — which
  is exactly the "you shut the PC down" case.
- Records are capped and atomically written, like every other state file here.
"""

import ctypes
import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from modules.file_utils import atomic_write_json

log = logging.getLogger("claw_bot.job_recovery")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
REGISTER_PATH = PROJECT_ROOT / "05_Config" / "active_jobs.json"

MAX_RECORDS = 50

# Modes that own a dashboard tab. `manual` is included: an interrupted animate
# or assemble is worth offering back.
MODES = ("story", "facts", "music", "horror", "manual", "lesson")

_lock = threading.Lock()


# ==============================================================================
# PID LIVENESS (same rule as job_lock — a dead holder is not a holder)
# ==============================================================================

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


# ==============================================================================
# STORE
# ==============================================================================

def _load() -> list[dict]:
    if not REGISTER_PATH.exists():
        return []
    try:
        data = json.loads(REGISTER_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception as e:
        log.warning(f"active_jobs.json unreadable ({e}); starting empty")
        return []


def _save(records: list[dict]) -> None:
    try:
        atomic_write_json(REGISTER_PATH, records[-MAX_RECORDS:])
    except Exception as e:
        log.warning(f"Could not write active_jobs.json: {e}")


# ==============================================================================
# PUBLIC API
# ==============================================================================

def begin(mode: str, label: str, stage: str = "", context: Optional[dict] = None) -> str:
    """Register a job as running. Returns its job_id."""
    if mode not in MODES:
        log.warning(f"Unknown mode '{mode}' registered for recovery")
    job_id = uuid.uuid4().hex[:12]
    rec = {
        "job_id": job_id,
        "mode": mode,
        "label": label,
        "stage": stage,
        "context": dict(context or {}),
        "pid": os.getpid(),
        "started": time.time(),
        "updated": time.time(),
    }
    with _lock:
        records = _load()
        records.append(rec)
        _save(records)
    log.info(f"Job registered for recovery: {mode}/{label} ({job_id})")
    return job_id


def checkpoint(job_id: str, stage: Optional[str] = None, **context) -> None:
    """Update the stage / context of a running job, so a resume lands close to
    where it died rather than at the very start."""
    with _lock:
        records = _load()
        for r in records:
            if r.get("job_id") == job_id:
                if stage:
                    r["stage"] = stage
                if context:
                    r.setdefault("context", {}).update(context)
                r["updated"] = time.time()
                _save(records)
                return
    log.debug(f"checkpoint for unknown job {job_id}")


def finish(job_id: str) -> None:
    """Deregister. Call from the worker's finally block — a job that raised is
    NOT an interrupted job, only a dead process is."""
    if not job_id:
        return
    with _lock:
        records = _load()
        kept = [r for r in records if r.get("job_id") != job_id]
        if len(kept) != len(records):
            _save(kept)


def discard(job_id: str) -> bool:
    """User chose 'end it and start new'."""
    with _lock:
        records = _load()
        kept = [r for r in records if r.get("job_id") != job_id]
        changed = len(kept) != len(records)
        if changed:
            _save(kept)
    if changed:
        log.info(f"Interrupted job discarded: {job_id}")
    return changed


def is_interrupted(rec: dict) -> bool:
    """Registered but its process is gone -> the PC/bot died mid-job."""
    return not _pid_alive(int(rec.get("pid", -1)))


def list_interrupted(mode: Optional[str] = None) -> list[dict]:
    """Jobs left behind by a dead process, newest first. Optionally per-mode."""
    out = [r for r in _load() if is_interrupted(r)]
    if mode:
        out = [r for r in out if r.get("mode") == mode]
    return sorted(out, key=lambda r: r.get("updated", 0), reverse=True)


def list_running(mode: Optional[str] = None) -> list[dict]:
    """Jobs whose process is still alive (this bot, or another instance)."""
    out = [r for r in _load() if not is_interrupted(r)]
    if mode:
        out = [r for r in out if r.get("mode") == mode]
    return out


def get(job_id: str) -> Optional[dict]:
    for r in _load():
        if r.get("job_id") == job_id:
            return r
    return None


def describe(rec: dict) -> str:
    """One-line human summary for a banner / Discord embed."""
    age_min = (time.time() - rec.get("updated", time.time())) / 60
    ctx = rec.get("context") or {}
    ident = (ctx.get("script_id") or ctx.get("facts_id") or ctx.get("book_id") or ctx.get("song_id")
             or ctx.get("horror_id") or ctx.get("project_id") or "")
    bits = [f"{rec.get('label', 'job')}"]
    if rec.get("stage"):
        bits.append(f"stage: {rec['stage']}")
    if ident:
        bits.append(f"`{ident}`")
    return " · ".join(bits) + f" — interrupted {age_min:.0f} min ago"


def sweep_dead_records(max_age_days: float = 7.0) -> int:
    """Drop interrupted records nobody ever acted on. Keeps the banner honest."""
    cutoff = time.time() - max_age_days * 86400
    with _lock:
        records = _load()
        kept = [r for r in records
                if not (is_interrupted(r) and r.get("updated", 0) < cutoff)]
        dropped = len(records) - len(kept)
        if dropped:
            _save(kept)
    if dropped:
        log.info(f"Swept {dropped} stale interrupted job record(s)")
    return dropped


def _reset_for_tests() -> None:
    with _lock:
        REGISTER_PATH.unlink(missing_ok=True)
