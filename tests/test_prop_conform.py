"""The Kontext prop-conform pass: repaint ONLY the prop, edit only shots that name one,
best-effort. The Kontext call + the lesson store are mocked; this pins the loop logic."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import modules.lesson_pipeline as lp            # noqa: E402
from modules import lesson_writer as lw, gpu_memory   # noqa: E402
import modules.comfyui_kontext_base as kx        # noqa: E402
import modules.lesson_objects as lo              # noqa: E402


_OBJS = [{"key": "doll", "noun": "doll", "desc": "a faceless block", "aliases": []},
         {"key": "puppy", "noun": "puppy", "desc": "a labrador", "aliases": []}]


@pytest.fixture()
def lesson(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "LESSONS_DIR", tmp_path)
    stills = tmp_path / "L1" / "stills"
    stills.mkdir(parents=True)
    for i in range(3):
        (stills / f"still_{i:02d}.png").write_bytes(b"\x89PNG fake")
    doc = {"beats": [
        {"mascot_scene": "the mascot holding a doll"},          # names doll
        {"mascot_scene": "the mascot waving, nothing in hand"},  # names nothing
        {"mascot_scene": "the mascot beside a puppy"},           # names puppy
    ], "objects": _OBJS}
    monkeypatch.setattr(lw, "load_lesson", lambda lid: doc)
    monkeypatch.setattr(gpu_memory, "acquire", lambda *a, **k: None)
    monkeypatch.setattr(gpu_memory, "release", lambda *a, **k: None)
    return tmp_path


def _mock_kontext(monkeypatch, ok=True):
    calls = []
    class R:
        def __init__(s, success): s.success = success; s.error = "" if success else "boom"
    def gen(prompt, output_path, **kw):
        calls.append({"prompt": prompt, "out": Path(output_path)})
        return R(ok)
    monkeypatch.setattr(kx, "generate", gen)
    monkeypatch.setattr(lp, "_kontext_available", lambda: (True, "ready"))
    return calls


def test_only_shots_that_name_a_prop_are_edited(lesson, monkeypatch):
    calls = _mock_kontext(monkeypatch)
    got = lp.conform_props("L1")
    assert got["conformed"] == [0, 2]                 # shot 1 (no prop) skipped
    assert len(calls) == 2
    # the instruction names the prop's canonical desc and forbids a face
    assert "faceless block" in calls[0]["prompt"]
    assert "stays exactly the same" in calls[0]["prompt"]


def test_a_failed_edit_keeps_the_drawn_still(lesson, monkeypatch):
    _mock_kontext(monkeypatch, ok=False)
    got = lp.conform_props("L1")
    assert got["conformed"] == []                     # nothing marked conformed


def test_skips_cleanly_when_kontext_is_absent(lesson, monkeypatch):
    monkeypatch.setattr(lp, "_kontext_available", lambda: (False, "not installed"))
    got = lp.conform_props("L1")
    assert got.get("skipped") is True
    assert got["conformed"] == []


def test_no_recurring_props_is_a_noop(lesson, monkeypatch):
    monkeypatch.setattr(lw, "load_lesson", lambda lid: {"beats": [], "objects": []})
    _mock_kontext(monkeypatch)
    got = lp.conform_props("L1")
    assert got["conformed"] == []
