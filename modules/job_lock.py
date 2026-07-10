"""
Claw Bot — shared GPU job queue (FIFO)

One GPU, several job sources: Discord commands, the NiceGUI dashboard
(worker threads in the same process), and the daily scheduler. Think render-farm
node allocation: one node, one job at a time — but jobs QUEUE instead of being
turned away. Submit three renders and they run first-come-first-served.

History: this used to be a plain try-lock. `acquire()` returned False when the
GPU was busy and every caller aborted with "GPU busy, try again". Two problems:
you had to babysit and resubmit, and `!facts` (which took no lock at all) could
start a second pipeline anyway — Ollama's 14B model landed in VRAM beside a
running Wan 14B on a 16 GB card and a 90s clip took 16 minutes.

Semantics:
- FIFO. Waiters take a ticket; the oldest ticket claims the GPU next.
- Reentrant for the SAME OWNER. An owner is the current asyncio Task when one
  is running, else the current thread. Chained pipelines (approve script ->
  storyboard -> video) run inside one Task and re-acquire without deadlocking.
  Two DIFFERENT Discord commands are different Tasks, so they queue behind each
  other properly — under the old thread-keyed rule they both "re-acquired" and
  interleaved on the GPU, because every Discord handler shares the event-loop
  thread.
- release() may be called from a different thread than acquire() — the dashboard
  claims in one thread and releases in a worker's finally block. (This is why
  threading.RLock isn't used: it forbids cross-thread release.)
- An on-disk marker (05_Config/job_lock.json) records the holder (label, pid,
  started). It blocks a SECOND bot process from starting GPU work and feeds
  !status / the dashboard. A marker from a dead or ancient pid is stolen.
  Ordering across processes is best-effort; within one process it is strict.

Usage — blocking (worker threads, dashboard):
    job_lock.acquire_blocking("dashboard:storyboard")
    try:
        ...GPU work...
    finally:
        job_lock.release()

Usage — async (Discord handlers; never blocks the event loop):
    await job_lock.acquire_async("discord:storyboard", on_queued=notify)
    try:
        ...GPU work...
    finally:
        job_lock.release()

`acquire()` is kept as the old non-blocking try-lock for callers that genuinely
want to bail (health checks, tests).
"""

import asyncio
import contextvars
import ctypes
import itertools
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

# How often a waiter re-checks the on-disk marker (another process may hold it,
# and a foreign release doesn't notify our condition variable).
POLL_SEC = 0.5

_guard = threading.Lock()
_cond = threading.Condition(_guard)

_depth = 0
_holder_owner = None      # opaque owner key, see _owner()
_holder_label = ""

_seq_counter = itertools.count(1)
_waiters: list["_Ticket"] = []      # FIFO, index 0 is next in line


class _Ticket:
    __slots__ = ("seq", "label", "enqueued_at", "claimed")

    def __init__(self, label: str):
        self.seq = next(_seq_counter)
        self.label = label
        self.enqueued_at = time.time()
        self.claimed = False

    def __repr__(self):
        return f"<Ticket #{self.seq} {self.label!r}>"


class QueueTimeout(RuntimeError):
    """Waited past the timeout without reaching the front of the queue."""


# ==============================================================================
# OWNERSHIP — Task when inside asyncio, else thread
# ==============================================================================

_owner_var: contextvars.ContextVar = contextvars.ContextVar("job_lock_owner",
                                                            default=None)
_owner_ids = itertools.count(1)


def _owner():
    """Identity used for reentrancy — a token carried in the context.

    Why not the Task or thread id?
      * Thread id: every Discord handler runs on the one event-loop thread, so
        two unrelated commands looked like the same owner and both "re-acquired",
        running on the GPU at once. That is the bug this queue exists to kill.
      * Task id: a chained stage wrapped in asyncio.wait_for / create_task /
        gather runs in a CHILD task with a different id — it would queue behind
        the very job holding the lock, and deadlock.

    A ContextVar threads the needle: asyncio copies the current context into
    each new Task (and asyncio.to_thread copies it into the worker), so children
    of the holder inherit the token and re-enter. A Discord command dispatched
    fresh from the event loop starts without the token, so it queues.
    """
    return _owner_var.get()


# ==============================================================================
# ON-DISK MARKER
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


def _foreign_marker_blocks() -> bool:
    """True when ANOTHER live process holds the on-disk marker."""
    marker = _read_marker()
    if not marker or marker.get("pid") == os.getpid():
        return False
    if _marker_is_stale(marker):
        log.warning(f"Stealing stale job lock from pid {marker.get('pid')} "
                    f"({marker.get('label')})")
        return False
    return True


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


# ==============================================================================
# INTERNALS — must hold _guard
# ==============================================================================

def _reenter_locked() -> bool:
    """Same owner already holds it -> bump depth. Caller holds _guard.

    A None token can never re-enter: an un-owned caller must queue.
    """
    global _depth
    token = _owner()
    if _depth > 0 and token is not None and token == _holder_owner:
        _depth += 1
        return True
    return False

def _write_marker(label: str) -> None:
    try:
        atomic_write_json(LOCK_PATH, {
            "label": label,
            "pid": os.getpid(),
            "started": time.time(),
        })
    except Exception as e:
        log.warning(f"Could not write job lock marker: {e}")


def _claim_locked(ticket: "_Ticket") -> bool:
    """Claim the GPU for `ticket` if it is next and nothing else holds it.
    Caller holds _guard."""
    global _depth, _holder_owner, _holder_label
    if _depth > 0:
        return False
    if not _waiters or _waiters[0] is not ticket:
        return False
    if _foreign_marker_blocks():
        return False
    _waiters.pop(0)
    ticket.claimed = True
    _depth = 1
    # Mint a token and stamp it on THIS context. Tasks/threads spawned from here
    # inherit it (context copy) and may re-enter; nobody else can.
    _holder_owner = next(_owner_ids)
    _owner_var.set(_holder_owner)
    _holder_label = ticket.label
    _write_marker(ticket.label)
    waited = time.time() - ticket.enqueued_at
    log.info(f"Job lock acquired: {ticket.label}"
             + (f" (queued {waited:.0f}s)" if waited > 1 else ""))
    return True


def _abandon_locked(ticket: "_Ticket") -> None:
    if not ticket.claimed and ticket in _waiters:
        _waiters.remove(ticket)
        _cond.notify_all()


# ==============================================================================
# PUBLIC API
# ==============================================================================

def position(ticket: "_Ticket") -> int:
    """1 = next to run. 0 = already running. Only meaningful while waiting."""
    with _guard:
        if ticket.claimed:
            return 0
        try:
            return _waiters.index(ticket) + 1
        except ValueError:
            return 0


def queue_snapshot() -> list[dict]:
    """Everything waiting right now, oldest first. For !status / the dashboard."""
    with _guard:
        return [
            {"seq": t.seq, "label": t.label,
             "waiting_sec": round(time.time() - t.enqueued_at, 1),
             "position": i + 1}
            for i, t in enumerate(_waiters)
        ]


def queue_depth() -> int:
    with _guard:
        return len(_waiters)


def acquire(label: str) -> bool:
    """Non-blocking try-lock (legacy). False means someone else holds the GPU.

    Prefer acquire_blocking / acquire_async — they queue instead of failing.
    """
    with _cond:
        if _reenter_locked():
            return True
        if _depth > 0 or _waiters or _foreign_marker_blocks():
            return False
        ticket = _Ticket(label)
        _waiters.append(ticket)
        return _claim_locked(ticket)


def acquire_blocking(label: str, timeout: float | None = None,
                     on_queued=None) -> bool:
    """Queue for the GPU and wait our turn. Blocks the calling THREAD.

    Never call this from the Discord event-loop thread — use acquire_async.
    `on_queued(position)` fires once, only if we actually have to wait.
    Raises QueueTimeout if `timeout` elapses first.
    """
    deadline = None if timeout is None else time.time() + timeout
    with _cond:
        if _reenter_locked():
            return True
        ticket = _Ticket(label)
        _waiters.append(ticket)
        notified = False
        try:
            while True:
                if _claim_locked(ticket):
                    return True
                if not notified and on_queued is not None:
                    pos = _waiters.index(ticket) + 1
                    _cond.release()
                    try:
                        on_queued(pos)
                    finally:
                        _cond.acquire()
                    notified = True
                    continue
                if deadline is not None and time.time() >= deadline:
                    _abandon_locked(ticket)
                    raise QueueTimeout(
                        f"{label}: waited {timeout:.0f}s, still behind "
                        f"{max(0, _waiters.index(ticket) if ticket in _waiters else 0)}")
                remaining = POLL_SEC if deadline is None else \
                    min(POLL_SEC, max(0.0, deadline - time.time()))
                _cond.wait(remaining)
        except BaseException:
            _abandon_locked(ticket)
            raise


async def acquire_async(label: str, timeout: float | None = None,
                        on_queued=None) -> bool:
    """Queue for the GPU from an asyncio handler. Yields instead of blocking.

    `on_queued(position)` is awaited once (if it returns an awaitable) the first
    time we find ourselves waiting — use it to post "Queued — 2 jobs ahead".
    Raises QueueTimeout if `timeout` elapses first.
    """
    deadline = None if timeout is None else time.time() + timeout
    with _cond:
        if _reenter_locked():
            return True
        ticket = _Ticket(label)
        _waiters.append(ticket)

    notified = False
    try:
        while True:
            with _cond:
                if _claim_locked(ticket):
                    return True
                pos = _waiters.index(ticket) + 1 if ticket in _waiters else 0
            if not notified and on_queued is not None:
                result = on_queued(pos)
                if asyncio.iscoroutine(result):
                    await result
                notified = True
            if deadline is not None and time.time() >= deadline:
                with _cond:
                    _abandon_locked(ticket)
                raise QueueTimeout(f"{label}: waited {timeout:.0f}s for the GPU")
            await asyncio.sleep(POLL_SEC)
    except BaseException:
        with _cond:
            _abandon_locked(ticket)
        raise


def release() -> None:
    """Release one level. Safe to call from a different thread than acquire()."""
    global _depth, _holder_owner, _holder_label
    with _cond:
        if _depth == 0:
            log.warning("job_lock.release() called while not held")
            return
        _depth -= 1
        if _depth == 0:
            _holder_owner = None
            _holder_label = ""
            try:
                LOCK_PATH.unlink(missing_ok=True)
            except Exception as e:
                log.warning(f"Could not remove job lock marker: {e}")
            log.info("Job lock released."
                     + (f" {len(_waiters)} job(s) queued." if _waiters else ""))
            _cond.notify_all()


def _reset_for_tests() -> None:
    global _depth, _holder_owner, _holder_label
    with _cond:
        _depth = 0
        _holder_owner = None
        _holder_label = ""
        _waiters.clear()
        LOCK_PATH.unlink(missing_ok=True)
        _cond.notify_all()
