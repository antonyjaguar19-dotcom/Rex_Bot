"""job_lock FIFO queue — ordering, fairness, reentrancy, timeouts.

The dangerous properties, and why each test exists:
  * FIFO      — three renders submitted must run 1,2,3, not last-in-first-out.
  * Fairness  — a fresh caller must not barge past someone already waiting.
  * Reentrancy — a chained pipeline re-acquiring inside its own Task/thread must
    NOT queue behind itself (that would deadlock the bot).
  * Task isolation — two different Discord commands share the event-loop thread;
    under the old thread-keyed rule they both "re-acquired" and interleaved on
    the GPU. They must now queue.
"""

import asyncio
import threading
import time

import pytest

from modules import job_lock as jl


@pytest.fixture(autouse=True)
def clean(tmp_path, monkeypatch):
    monkeypatch.setattr(jl, "LOCK_PATH", tmp_path / "job_lock.json")
    monkeypatch.setattr(jl, "POLL_SEC", 0.01)   # keep the suite fast
    jl._reset_for_tests()
    yield
    jl._reset_for_tests()


# ---------------------------------------------------------------- FIFO

def test_blocking_waiters_run_in_arrival_order():
    order = []
    started = []

    jl.acquire_blocking("holder")          # occupy the GPU

    def worker(n):
        started.append(n)
        jl.acquire_blocking(f"job{n}")
        order.append(n)
        time.sleep(0.02)
        jl.release()

    threads = []
    for n in (1, 2, 3):
        t = threading.Thread(target=worker, args=(n,))
        t.start()
        threads.append(t)
        # stagger so arrival order is unambiguous
        while n not in started:
            time.sleep(0.005)
        time.sleep(0.05)

    assert jl.queue_depth() == 3
    jl.release()                            # let them go
    for t in threads:
        t.join(timeout=5)
    assert order == [1, 2, 3]


def test_position_reports_place_in_line():
    jl.acquire_blocking("holder")
    positions = {}

    def worker(n):
        # enqueue via the async-free path, capturing our position once queued
        def note(pos):
            positions[n] = pos
        jl.acquire_blocking(f"job{n}", on_queued=note)
        jl.release()

    ts = []
    for n in (1, 2):
        t = threading.Thread(target=worker, args=(n,))
        t.start(); ts.append(t)
        time.sleep(0.08)

    assert positions == {1: 1, 2: 2}
    jl.release()
    for t in ts:
        t.join(timeout=5)


def test_new_caller_cannot_barge_past_a_waiter():
    """acquire() (try-lock) must fail while someone is queued, or FIFO is a lie."""
    jl.acquire_blocking("holder")
    waiting = threading.Event()

    def waiter():
        jl.acquire_blocking("queued", on_queued=lambda p: waiting.set())
        jl.release()

    t = threading.Thread(target=waiter); t.start()
    assert waiting.wait(2)

    jl.release()                       # holder done; "queued" is next in line
    # A barger tries the non-blocking path before the waiter wakes.
    assert jl.acquire("barger") is False
    t.join(timeout=5)


# ---------------------------------------------------------------- reentrancy

def test_same_thread_chain_reenters_without_deadlock():
    assert jl.acquire_blocking("outer")
    # a chained stage re-acquires inside the same thread — must return at once
    assert jl.acquire_blocking("inner", timeout=1)
    jl.release()
    jl.release()
    assert jl.queue_depth() == 0
    assert jl.current() is None


def test_reentrant_depth_keeps_lock_until_fully_released():
    jl.acquire_blocking("outer")
    jl.acquire_blocking("inner")
    jl.release()
    assert jl.current() is not None      # still held by outer
    jl.release()
    assert jl.current() is None


# ---------------------------------------------------------------- asyncio

def test_two_asyncio_tasks_queue_instead_of_interleaving():
    """Two Discord commands share the event-loop thread. They must NOT both
    acquire (the old thread-keyed reentrancy bug)."""
    concurrent = []
    peak = []

    async def job(n):
        await jl.acquire_async(f"cmd{n}")
        concurrent.append(n)
        peak.append(len(concurrent))
        await asyncio.sleep(0.05)
        concurrent.remove(n)
        jl.release()

    async def main():
        await asyncio.gather(job(1), job(2), job(3))

    asyncio.run(main())
    assert peak == [1, 1, 1], f"GPU ran {max(peak)} jobs at once"


def test_async_preserves_submission_order():
    order = []

    async def job(n):
        await jl.acquire_async(f"cmd{n}")
        order.append(n)
        await asyncio.sleep(0.01)
        jl.release()

    async def main():
        tasks = []
        for n in (1, 2, 3, 4):
            tasks.append(asyncio.create_task(job(n)))
            await asyncio.sleep(0.02)   # unambiguous arrival order
        await asyncio.gather(*tasks)

    asyncio.run(main())
    assert order == [1, 2, 3, 4]


def test_async_chain_within_one_task_reenters():
    """approve script -> storyboard -> video all await inside ONE task."""
    async def main():
        await jl.acquire_async("outer")
        await asyncio.wait_for(jl.acquire_async("inner"), timeout=1)  # no deadlock
        jl.release()
        jl.release()
    asyncio.run(main())
    assert jl.current() is None


def test_async_on_queued_reports_position():
    seen = []

    async def main():
        # The holder must run in its OWN task: a task spawned from a context that
        # already holds the lock inherits the owner token and re-enters by design
        # (that is the chained-pipeline case). Independent commands start clean.
        done = asyncio.Event()

        async def holder():
            await jl.acquire_async("holder")
            await done.wait()
            jl.release()

        async def other():
            await jl.acquire_async("second", on_queued=lambda p: seen.append(p))
            jl.release()

        h = asyncio.create_task(holder())
        await asyncio.sleep(0.05)
        o = asyncio.create_task(other())
        await asyncio.sleep(0.1)
        done.set()
        await asyncio.gather(h, o)

    asyncio.run(main())
    assert seen == [1]


# ---------------------------------------------------------------- timeouts

def test_blocking_timeout_gives_up_and_leaves_queue_clean():
    # hold from another thread, so the impatient caller is a genuine outsider
    holding = threading.Event()
    finish = threading.Event()

    def holder():
        jl.acquire_blocking("holder")
        holding.set()
        finish.wait(5)
        jl.release()

    t = threading.Thread(target=holder); t.start()
    assert holding.wait(2)

    with pytest.raises(jl.QueueTimeout):
        jl.acquire_blocking("impatient", timeout=0.1)
    assert jl.queue_depth() == 0          # ticket removed, no phantom waiter

    finish.set(); t.join(timeout=5)
    assert jl.acquire("after") is True    # queue is usable again
    jl.release()


def test_async_timeout_cleans_up():
    async def main():
        done = asyncio.Event()

        async def holder():
            await jl.acquire_async("holder")
            await done.wait()
            jl.release()

        async def impatient():
            with pytest.raises(jl.QueueTimeout):
                await jl.acquire_async("impatient", timeout=0.1)

        h = asyncio.create_task(holder())
        await asyncio.sleep(0.05)
        await asyncio.create_task(impatient())
        assert jl.queue_depth() == 0
        done.set()
        await h

    asyncio.run(main())


def test_cancelled_async_waiter_leaves_queue():
    async def main():
        done = asyncio.Event()

        async def holder():
            await jl.acquire_async("holder")
            await done.wait()
            jl.release()

        h = asyncio.create_task(holder())
        await asyncio.sleep(0.05)

        # spawn the waiter from a CLEAN context (as discord.py would), not from
        # the holder's — otherwise it would inherit the token and re-enter.
        t = asyncio.create_task(jl.acquire_async("doomed"))
        await asyncio.sleep(0.05)
        assert jl.queue_depth() == 1
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        assert jl.queue_depth() == 0     # cancellation must not wedge the queue
        done.set()
        await h

    asyncio.run(main())


# ---------------------------------------------------------------- snapshot

def test_queue_snapshot_shape():
    jl.acquire_blocking("holder")
    ev = threading.Event()

    def waiter():
        jl.acquire_blocking("render shot 3", on_queued=lambda p: ev.set())
        jl.release()

    t = threading.Thread(target=waiter); t.start()
    assert ev.wait(2)
    snap = jl.queue_snapshot()
    assert len(snap) == 1
    assert snap[0]["label"] == "render shot 3"
    assert snap[0]["position"] == 1
    assert snap[0]["waiting_sec"] >= 0
    jl.release()
    t.join(timeout=5)
