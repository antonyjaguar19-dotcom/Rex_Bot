"""job_recovery — the register of in-flight jobs used to offer Resume/Discard.

The load-bearing rules:
  * A job whose process is ALIVE is running, not interrupted (never offer to
    resume something that is currently rendering).
  * A job that merely RAISED is deregistered by finish() -- only a dead process
    leaves a record. That is the "I shut the PC down" case.
  * checkpoint() moves the stage so a resume re-enters near where it died.
"""

import os
import time

import pytest

from modules import job_recovery as jr


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(jr, "REGISTER_PATH", tmp_path / "active_jobs.json")
    jr._reset_for_tests()
    yield
    jr._reset_for_tests()


DEAD_PID = 2 ** 22          # not a live process


def _kill(job_id):
    """Simulate the process dying: rewrite the record's pid to a dead one."""
    recs = jr._load()
    for r in recs:
        if r["job_id"] == job_id:
            r["pid"] = DEAD_PID
    jr._save(recs)


# ---------------------------------------------------------------- lifecycle

def test_begin_registers_a_running_job():
    jid = jr.begin("facts", "facts reel", stage="write", context={"topic": "bees"})
    rec = jr.get(jid)
    assert rec["mode"] == "facts"
    assert rec["pid"] == os.getpid()
    assert rec["context"]["topic"] == "bees"
    # our own pid is alive -> running, not interrupted
    assert jr.list_interrupted() == []
    assert len(jr.list_running()) == 1


def test_finish_deregisters():
    jid = jr.begin("story", "video render")
    jr.finish(jid)
    assert jr.get(jid) is None
    assert jr.list_running() == []


def test_a_job_that_raised_is_not_interrupted():
    """finish() runs in the worker's finally, so an exception must not leave a
    record -- otherwise every failed render would nag you to resume it."""
    jid = jr.begin("music", "music video render")
    try:
        raise RuntimeError("render blew up")
    except RuntimeError:
        pass
    finally:
        jr.finish(jid)
    assert jr.list_interrupted() == []
    assert jr.get(jid) is None


def test_dead_process_leaves_an_interrupted_job():
    jid = jr.begin("horror", "horror render", stage="images")
    _kill(jid)
    interrupted = jr.list_interrupted()
    assert len(interrupted) == 1
    assert interrupted[0]["job_id"] == jid
    assert interrupted[0]["stage"] == "images"
    assert jr.list_running() == []


def test_live_job_is_never_offered_for_resume():
    jr.begin("story", "storyboard render")
    assert jr.list_interrupted() == []      # our pid is alive


# ---------------------------------------------------------------- checkpoint

def test_checkpoint_advances_stage_and_context():
    jid = jr.begin("story", "pipeline", stage="script")
    jr.checkpoint(jid, stage="storyboard", script_id="20260710_1200")
    jr.checkpoint(jid, stage="video")
    rec = jr.get(jid)
    assert rec["stage"] == "video"
    assert rec["context"]["script_id"] == "20260710_1200"


def test_checkpoint_on_unknown_job_is_harmless():
    jr.checkpoint("nope", stage="video")     # must not raise
    assert jr.get("nope") is None


# ---------------------------------------------------------------- filtering

def test_list_interrupted_filters_by_mode():
    a = jr.begin("facts", "facts reel")
    b = jr.begin("story", "video render")
    _kill(a); _kill(b)
    assert [r["job_id"] for r in jr.list_interrupted("facts")] == [a]
    assert [r["job_id"] for r in jr.list_interrupted("story")] == [b]
    assert len(jr.list_interrupted()) == 2


def test_interrupted_sorted_newest_first():
    a = jr.begin("story", "old")
    time.sleep(0.01)
    b = jr.begin("story", "new")
    _kill(a); _kill(b)
    assert [r["job_id"] for r in jr.list_interrupted()] == [b, a]


# ---------------------------------------------------------------- discard

def test_discard_removes_the_record():
    jid = jr.begin("manual", "manual animate shot 2")
    _kill(jid)
    assert jr.discard(jid) is True
    assert jr.list_interrupted() == []
    assert jr.discard(jid) is False          # idempotent


# ---------------------------------------------------------------- housekeeping

def test_sweep_drops_only_old_interrupted_records():
    old = jr.begin("story", "ancient")
    fresh = jr.begin("story", "recent")
    _kill(old); _kill(fresh)
    recs = jr._load()
    for r in recs:
        if r["job_id"] == old:
            r["updated"] = time.time() - 30 * 86400
    jr._save(recs)

    assert jr.sweep_dead_records(max_age_days=7) == 1
    remaining = [r["job_id"] for r in jr.list_interrupted()]
    assert remaining == [fresh]


def test_sweep_keeps_running_jobs():
    jr.begin("story", "alive")               # our pid
    assert jr.sweep_dead_records(max_age_days=0) == 0
    assert len(jr.list_running()) == 1


def test_corrupt_register_does_not_crash():
    jr.REGISTER_PATH.write_text("{ not json", encoding="utf-8")
    assert jr.list_interrupted() == []
    jid = jr.begin("facts", "after corruption")   # recovers by starting fresh
    assert jr.get(jid) is not None


def test_describe_mentions_stage_and_id():
    jid = jr.begin("story", "video render", stage="video",
                   context={"script_id": "20260710_0900"})
    _kill(jid)
    text = jr.describe(jr.list_interrupted()[0])
    assert "video render" in text and "video" in text and "20260710_0900" in text
