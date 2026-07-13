"""Discord → Web: a render started in Discord has to be visible in the browser.

sync_bridge already carried dashboard jobs INTO Discord. The reverse did not
exist, so a `!facts` render left the dashboard showing "✓ Idle — ready" for half
an hour while the bot was plainly working — the web UI had no way to know.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import job_feed as jf                 # noqa: E402
from modules import dashboard_nicegui as dash      # noqa: E402


@pytest.fixture(autouse=True)
def feed(tmp_path, monkeypatch):
    monkeypatch.setattr(jf, "FEED_PATH", tmp_path / "bot_job.json")
    monkeypatch.setattr(jf, "_STATE", {}, raising=False)
    dash._REMOTE_CACHE.update({"at": 0.0, "job": {}})
    yield
    dash._REMOTE_CACHE.update({"at": 0.0, "job": {}})


def test_no_job_reads_empty():
    assert jf.read() == {}
    assert jf.is_running() is False


def test_a_running_job_is_visible_with_its_progress():
    jf.begin("j1", "facts", "facts reel")
    jf.push("🎙️ voicing 8 beats")
    jf.push("wan clip 3/8")
    d = jf.read()
    assert d["label"] == "facts reel" and d["mode"] == "facts"
    assert d["done"] is False and jf.is_running()
    assert d["lines"][-1] == "wan clip 3/8"


def test_finished_job_reports_how_it_ended_then_disappears(monkeypatch):
    jf.begin("j2", "facts", "facts reel")
    jf.end(ok=False, note="❌ failed")
    d = jf.read()
    assert d["done"] is True and d["ok"] is False
    assert d["lines"][-1] == "❌ failed"
    assert jf.is_running() is False

    # Two minutes later it stops cluttering the page.
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + 200)
    assert jf.read() == {}


def test_a_bot_that_died_mid_render_is_not_shown_as_running(monkeypatch):
    jf.begin("j3", "facts", "facts reel")
    jf.push("wan clip 1/8")
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + jf.STALE_AFTER_SEC + 60)
    d = jf.read()
    assert d["done"] is True and d["ok"] is False
    assert "died" in d["lines"][-1]


def test_lines_are_bounded():
    jf.begin("j4", "story", "video render")
    for i in range(120):
        jf.push(f"line {i}")
    assert len(jf.read()["lines"]) == jf.MAX_LINES


def test_push_after_end_is_ignored():
    jf.begin("j5", "facts", "facts reel")
    jf.end(ok=True)
    jf.push("late line")
    assert "late line" not in jf.read()["lines"]


# ------------------------------------------------------ what the page shows --

def test_dashboard_sees_the_discord_job():
    jf.begin("j6", "facts", "facts reel")
    jf.push("wan clip 2/8")
    assert dash._remote_job().get("label") == "facts reel"


def test_stepper_follows_the_render_by_what_it_says():
    """Lines copied VERBATIM from a real Discord facts render's log — a table of
    keywords I invented would only prove I can match my own guesses. `wan clip
    3/8` read as stage "write" until a live smoke test caught it."""
    cases = {
        "🧠 writing 6 facts about bees": "write",
        "facts written: 'Fire Facts' — 8 beats in 13s": "write",
        "🎙️ voicing 8 beats (clone)": "voice",
        "Mascot base rendered: still_04.png (9x16, comfyui_qwen_edit)": "images",
        "🎬 animating 8 clips (Wan I2V, narration as voice-over)": "assemble",
        "wan clip 3/8...": "assemble",
        "🎵 composing the music bed (cheerful, 34s reel)": "assemble",
        "🎵 bed verified in the finished reel: -31 dB under the voice": "assemble",
        "🖼️ thumbnail held on the front of facts_20260713_152428_9x16.mp4": "assemble",
        "✅ facts reel complete": "done",
    }
    for line, expected in cases.items():
        assert dash._remote_facts_stage({"lines": [line]}) == expected, line
