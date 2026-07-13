"""Web → Discord mirror: the plumbing that makes a browser-started job visible
(and approvable) from Discord.

Two things broke silently before this existed:
  * a reel rendered on the dashboard never showed its DESCRIPTION, because the
    dashboard looked for facts_<id>_description.txt while publish_kit writes
    <video-stem>_description.txt;
  * a job started in the browser produced no Discord message at all.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import dashboard_nicegui as dash   # noqa: E402


# --------------------------------------------------------------- description --

def test_description_found_under_publish_kit_name(tmp_path):
    video = tmp_path / "facts_20260713_072438_9x16.mp4"
    video.write_bytes(b"")
    (tmp_path / "facts_20260713_072438_9x16_description.txt").write_text(
        "5 wild facts #shorts", encoding="utf-8")
    found = dash._description_file(video)
    assert found is not None and "wild facts" in found.read_text(encoding="utf-8")


def test_description_found_under_legacy_per_id_name(tmp_path):
    """Older facts reels wrote the description without the aspect suffix."""
    video = tmp_path / "facts_20260710_090804_9x16.mp4"
    video.write_bytes(b"")
    (tmp_path / "facts_20260710_090804_description.txt").write_text(
        "legacy caption", encoding="utf-8")
    found = dash._description_file(video)
    assert found is not None and found.name.endswith("090804_description.txt")


def test_description_absent_is_none(tmp_path):
    video = tmp_path / "final_x_16x9.mp4"
    video.write_bytes(b"")
    assert dash._description_file(video) is None


# ------------------------------------------------------------------- mirror --

def test_mirror_files_orders_aspects(monkeypatch):
    sent = {}
    monkeypatch.setattr(dash, "_mirror",
                        lambda ev, **d: sent.update({"ev": ev, **d}))
    dash._mirror_files("reel_ready",
                       {"16x9": "b.mp4", "9x16": "a.mp4", "1x1": "c.mp4"},
                       mode="facts", title="Bees")
    assert sent["ev"] == "reel_ready"
    assert sent["files"] == ["a.mp4", "b.mp4", "c.mp4"]   # 9x16 first — it is the reel
    assert sent["mode"] == "facts" and sent["title"] == "Bees"


def test_mirror_files_accepts_a_list(monkeypatch):
    sent = {}
    monkeypatch.setattr(dash, "_mirror", lambda ev, **d: sent.update(d))
    dash._mirror_files("final_done", [Path("x.mp4")], mode="story", script_id="s1")
    assert sent["files"] == ["x.mp4"] and sent["script_id"] == "s1"


def test_progress_is_not_mirrored_when_idle(monkeypatch):
    calls = []
    monkeypatch.setattr(dash, "_mirror", lambda ev, **d: calls.append(ev))
    state = dash.State()
    state.active_job = None
    state.push("just browsing")
    assert calls == []          # idle chatter must not reach Discord


def test_progress_throttled_but_loud_lines_always_pass(monkeypatch):
    calls = []
    monkeypatch.setattr(dash, "_mirror", lambda ev, **d: calls.append(d["line"]))
    state = dash.State()
    state.active_job = {"id": "j1", "mode": "facts", "label": "facts reel",
                        "checkpointed": 0.0, "mirrored": 0.0}
    state.push("· step one")              # first line: mirrored
    state.push("· step two")              # inside the throttle window: dropped
    state.push("✅ Facts reel ready: x")   # loud: always mirrored
    state.push("Facts reel failed: boom")  # loud: always mirrored
    assert calls == ["· step one", "✅ Facts reel ready: x", "Facts reel failed: boom"]


# ---------------------------------------------------------------- coalescing --

def test_coalesce_keeps_only_the_newest_progress_per_job():
    import claw_bot as cb
    events = [
        {"id": 1, "type": "job_start", "data": {"job": "a"}},
        {"id": 2, "type": "job_progress", "data": {"job": "a", "line": "old"}},
        {"id": 3, "type": "job_progress", "data": {"job": "b", "line": "b1"}},
        {"id": 4, "type": "job_progress", "data": {"job": "a", "line": "new"}},
        {"id": 5, "type": "reel_ready", "data": {"files": []}},
    ]
    kept = cb._coalesce(events)
    ids = [e["id"] for e in kept]
    assert ids == [1, 3, 4, 5]           # #2 superseded by #4; nothing else dropped


def test_coalesce_leaves_non_progress_events_alone():
    import claw_bot as cb
    events = [{"id": 1, "type": "script_ready", "data": {"script_id": "s"}},
              {"id": 2, "type": "job_end", "data": {"job": "a"}}]
    assert cb._coalesce(events) == events


# --------------------------------------------------------- kokoro out of UI --

def test_mascot_voice_choices_have_no_kokoro():
    """Kokoro survives as a silent fallback inside the pipeline, but offering it
    beside a cloned voice only ever confused the picker."""
    from modules import runtime_settings as rs
    offered = ["clone"] + list(rs.VALID_QWEN_SPEAKERS)
    assert "kokoro" not in offered
    src = (Path(__file__).parent.parent / "modules" / "dashboard_nicegui.py") \
        .read_text(encoding="utf-8")
    facts_ui = src.split("FACTS SHORTS (own tab)")[1].split("STAGE 2")[0]
    # The only kokoro left in the Facts tab may be the migration that rewrites a
    # stale config — never an option, a label, or a tooltip the user reads.
    leaks = [ln.strip() for ln in facts_ui.splitlines()
             if "kokoro" in ln.lower()
             and not ln.lstrip().startswith("#")
             and 'get_mascot_tts_engine() == "kokoro"' not in ln]
    assert leaks == [], f"kokoro still shown in the Facts tab: {leaks}"
