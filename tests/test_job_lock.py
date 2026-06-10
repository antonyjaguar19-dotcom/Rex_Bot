import threading
import time

import pytest

from modules import job_lock as jl
from modules.file_utils import atomic_write_json


@pytest.fixture(autouse=True)
def isolated_lock(tmp_path, monkeypatch):
    """Point the on-disk marker at a temp file and reset in-process state."""
    monkeypatch.setattr(jl, "LOCK_PATH", tmp_path / "job_lock.json")
    jl._depth = 0
    jl._holder_ident = None
    jl._holder_label = ""
    yield
    jl._depth = 0
    jl._holder_ident = None
    jl._holder_label = ""


def test_acquire_release_cycle():
    assert jl.acquire("test:a")
    assert jl.LOCK_PATH.exists()
    assert jl.holder_label() == "test:a"
    jl.release()
    assert not jl.LOCK_PATH.exists()
    assert jl.holder_label() == "idle"


def test_other_thread_blocked():
    assert jl.acquire("test:a")
    got = {}
    t = threading.Thread(target=lambda: got.update(ok=jl.acquire("test:b")))
    t.start(); t.join()
    assert got["ok"] is False
    jl.release()


def test_same_thread_reentrant():
    # Discord pipeline chains (approve -> storyboard -> video) re-acquire
    # on the same event-loop thread.
    assert jl.acquire("test:outer")
    assert jl.acquire("test:inner")
    jl.release()
    assert jl.LOCK_PATH.exists(), "still held at depth 1"
    jl.release()
    assert not jl.LOCK_PATH.exists()


def test_cross_thread_release():
    # Dashboard pattern: UI thread acquires, worker thread releases.
    assert jl.acquire("test:dash")
    t = threading.Thread(target=jl.release)
    t.start(); t.join()
    assert not jl.LOCK_PATH.exists()
    assert jl.acquire("test:after")
    jl.release()


def test_stale_marker_from_dead_pid_is_stolen():
    atomic_write_json(jl.LOCK_PATH,
                      {"label": "ghost", "pid": 999_999_999, "started": time.time()})
    assert jl.acquire("test:steal")
    jl.release()


def test_release_when_not_held_is_safe():
    jl.release()  # must not raise
    assert jl.acquire("test:still-works")
    jl.release()
