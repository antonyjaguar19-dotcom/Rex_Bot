"""Dashboard resume dispatch — a crashed job must re-enter at the right stage.

These exercise the routing only (no GPU): the pipeline actions are stubbed, so
what's under test is "given an interrupted record, which stage do we restart".
"""

import json

import pytest

from modules import dashboard_nicegui as d
from modules import job_recovery as jr


DEAD_PID = 2 ** 22


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(jr, "REGISTER_PATH", tmp_path / "active_jobs.json")
    monkeypatch.setattr(d, "SCRIPTS_DIR", tmp_path / "scripts")
    jr._reset_for_tests()
    d.S.script = None
    d.S.script_id = None
    d.S.active_job = None
    yield
    jr._reset_for_tests()


def _kill(job_id):
    recs = jr._load()
    for r in recs:
        if r["job_id"] == job_id:
            r["pid"] = DEAD_PID
    jr._save(recs)


def _interrupted(mode, label, stage="", context=None):
    jid = jr.begin(mode, label, stage=stage, context=context or {})
    _kill(jid)
    return jid


# ---------------------------------------------------------------- mode routing

@pytest.mark.parametrize("label,expected", [
    ("manual animate shot 3", "manual"),
    ("manual assembly", "manual"),
    ("facts reel", "facts"),
    ("horror render", "horror"),
    ("horror writing", "horror"),
    ("song generation", "music"),
    ("music video render", "music"),
    ("lyrics rewrite", "music"),
    ("video render", "story"),
    ("storyboard render", "story"),
    ("final assembly", "story"),
])
def test_label_routes_to_owning_tab(label, expected):
    assert d._mode_for_label(label) == expected


# ---------------------------------------------------------------- story stages

@pytest.mark.parametrize("stage,expected_fn", [
    ("script", "approve_script_gen_prompts"),
    ("prompts", "approve_script_gen_prompts"),
    ("storyboard", "approve_all_run_storyboard"),
    ("video", "run_video"),
    ("final", "assemble_final"),
])
def test_story_resume_reenters_at_recorded_stage(stage, expected_fn, monkeypatch, tmp_path):
    called = []
    for fn in ("approve_script_gen_prompts", "approve_all_run_storyboard",
               "run_video", "assemble_final"):
        monkeypatch.setattr(d, fn, lambda cb, _f=fn: called.append(_f))

    sid = "20260710_1200"
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / f"script_{sid}.json").write_text(
        json.dumps({"_id": sid, "title": "T", "shots": []}), encoding="utf-8")

    jid = _interrupted("story", "video render", stage=stage, context={"script_id": sid})
    rec = jr.get(jid)
    assert d._resume_story(rec, lambda: None) is None
    assert called == [expected_fn]
    assert d.S.script_id == sid          # state reloaded from disk


def test_story_resume_without_script_id_explains_itself():
    jid = _interrupted("story", "script generation", stage="script")
    problem = d._resume_story(jr.get(jid), lambda: None)
    assert problem and "no script id" in problem


def test_story_resume_when_script_deleted():
    jid = _interrupted("story", "video render", stage="video",
                       context={"script_id": "gone_forever"})
    problem = d._resume_story(jr.get(jid), lambda: None)
    assert problem and "gone from disk" in problem


# ---------------------------------------------------------------- other modes

def test_facts_resume_reloads_saved_story(monkeypatch):
    seen = {}
    monkeypatch.setattr(d, "resume_facts_render_action",
                        lambda story, cb: seen.update(story))

    import modules.facts_writer as fw
    monkeypatch.setattr(fw, "load_facts", lambda fid: {"_id": fid, "title": "Bees"})

    jid = _interrupted("facts", "facts reel", stage="images",
                       context={"facts_id": "20260710_0800"})
    assert d._resume_facts(jr.get(jid), lambda: None) is None
    assert seen["title"] == "Bees"


def test_facts_resume_before_writing_finished_cannot_resume():
    jid = _interrupted("facts", "facts reel", stage="write")   # no facts_id yet
    problem = d._resume_facts(jr.get(jid), lambda: None)
    assert problem and "start a new one" in problem


def test_music_resume_loads_song(monkeypatch):
    calls = []
    monkeypatch.setattr(d, "render_musicvideo_action", lambda cb: calls.append(True))
    import modules.song_generator as sgn
    monkeypatch.setattr(sgn, "load_song", lambda sid: {"_id": sid, "title": "S"})

    jid = _interrupted("music", "music video render", stage="visuals",
                       context={"song_id": "20260710_0700"})
    assert d._resume_music(jr.get(jid), lambda: None) is None
    assert calls == [True] and d.S.song["_id"] == "20260710_0700"


def test_horror_resume_loads_story(monkeypatch):
    calls = []
    monkeypatch.setattr(d, "render_horror_action", lambda cb: calls.append(True))
    import modules.horror_writer as hw
    monkeypatch.setattr(hw, "load_horror", lambda hid: {"_id": hid})

    jid = _interrupted("horror", "horror render", stage="final",
                       context={"horror_id": "20260710_0600"})
    assert d._resume_horror(jr.get(jid), lambda: None) is None
    assert calls == [True]


def test_manual_resume_replays_the_named_shot(monkeypatch, tmp_path):
    from modules import manual_mode as mm
    monkeypatch.setattr(mm, "MANUAL_DIR", tmp_path / "manual")
    monkeypatch.setattr(mm, "CURRENT_MARKER", tmp_path / "manual" / "_current.txt")
    proj = mm.create_project("P")

    animated = []
    monkeypatch.setattr(d, "manual_animate_action",
                        lambda n, cb: animated.append(n))

    jid = _interrupted("manual", "manual animate shot 4",
                       context={"project_id": proj["_id"]})
    assert d._resume_manual(jr.get(jid), lambda: None) is None
    assert animated == [4]


def test_manual_resume_missing_project():
    jid = _interrupted("manual", "manual assembly", context={"project_id": "nope"})
    problem = d._resume_manual(jr.get(jid), lambda: None)
    assert problem and "gone" in problem


# ---------------------------------------------------------------- guards

def test_resume_refuses_a_job_that_is_still_running(monkeypatch):
    notes = []
    monkeypatch.setattr(d.ui, "notify", lambda msg, **k: notes.append(msg))
    jid = jr.begin("story", "video render")     # our pid: alive
    d.resume_job_action(jid, lambda: None)
    assert any("running right now" in n for n in notes)


def test_discard_clears_the_record(monkeypatch):
    monkeypatch.setattr(d.ui, "notify", lambda *a, **k: None)
    jid = _interrupted("story", "video render", stage="video")
    d.discard_job_action(jid, lambda: None)
    assert jr.get(jid) is None
    assert jr.list_interrupted() == []


def test_push_checkpoint_is_noop_without_active_job():
    d.S.active_job = None
    d.S.push("just a log line")      # must not raise or write anything
    assert jr.list_interrupted() == []
