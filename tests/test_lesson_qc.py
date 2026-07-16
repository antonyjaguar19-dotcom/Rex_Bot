"""The lesson visual-QC pass: LOOK at every still, re-draw the flagged ones, VRAM-safe,
capped, best-effort. The VLM + the renderer are mocked; this pins the loop logic."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import modules.lesson_pipeline as lp            # noqa: E402
from modules import image_qc, gpu_utils, lesson_writer as lw   # noqa: E402


@pytest.fixture()
def two_shot_lesson(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "LESSONS_DIR", tmp_path)
    stills = tmp_path / "L1" / "stills"
    stills.mkdir(parents=True)
    for i in range(2):
        (stills / f"still_{i:02d}.png").write_bytes(b"\x89PNG fake")
    monkeypatch.setattr(lw, "load_lesson", lambda lid: {"beats": [{}, {}]})
    monkeypatch.setattr(gpu_utils, "free_comfyui_vram", lambda *a, **k: (True, ""))
    monkeypatch.setattr(gpu_utils, "free_ollama_vram", lambda *a, **k: (True, ""))
    return tmp_path


def test_skips_cleanly_when_the_qc_model_is_absent(two_shot_lesson, monkeypatch):
    monkeypatch.setattr(image_qc, "available", lambda: (False, "not pulled"))
    got = lp.run_visual_qc("L1")
    assert got["skipped"] is True
    assert got["flagged"] == []


def test_all_clean_returns_no_flags_and_never_redraws(two_shot_lesson, monkeypatch):
    monkeypatch.setattr(image_qc, "available", lambda: (True, "ready"))
    monkeypatch.setattr(image_qc, "qc_still", lambda p, ctx="teaching": {"ok": True, "fails": []})
    redraws = []
    monkeypatch.setattr(lp, "redraw_still", lambda lid, i, cb=None: redraws.append(i))
    got = lp.run_visual_qc("L1")
    assert got["flagged"] == []
    assert redraws == [], "a clean shot must not be re-drawn"


def test_a_flagged_shot_is_redrawn_then_passes(two_shot_lesson, monkeypatch):
    monkeypatch.setattr(image_qc, "available", lambda: (True, "ready"))
    seen = {"round": 0}
    # shot 0 fails the FIRST look, passes after the re-draw; shot 1 is always clean
    def fake_qc(p, ctx="teaching"):
        i = int(Path(p).stem.split("_")[1])
        bad = (i == 0 and seen["round"] == 0)
        return {"ok": not bad, "fails": [{"key": "warm_face", "note": ""}] if bad else []}
    monkeypatch.setattr(image_qc, "qc_still", fake_qc)
    redraws = []
    def fake_redraw(lid, i, cb=None):
        redraws.append(i)
        seen["round"] = 1                # the re-draw fixes it
    monkeypatch.setattr(lp, "redraw_still", fake_redraw)
    got = lp.run_visual_qc("L1", max_rounds=2)
    assert redraws == [0], "only the flagged shot is re-drawn"
    assert got["flagged"] == []          # clean after the re-roll


def test_a_persistently_bad_shot_is_surfaced_not_looped(two_shot_lesson, monkeypatch):
    monkeypatch.setattr(image_qc, "available", lambda: (True, "ready"))
    monkeypatch.setattr(image_qc, "qc_still",
                        lambda p, ctx="teaching": {"ok": Path(p).stem.endswith("01"),
                                                   "fails": [{"key": "hands_busy", "note": ""}]})
    redraws = []
    monkeypatch.setattr(lp, "redraw_still", lambda lid, i, cb=None: redraws.append(i))
    got = lp.run_visual_qc("L1", max_rounds=2)
    assert got["flagged"] == [0]         # shot 0 never passes -> surfaced
    assert redraws.count(0) == 2         # re-drawn max_rounds times, then given up on
