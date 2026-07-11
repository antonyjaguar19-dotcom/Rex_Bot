"""Facts per-beat prompt editing (like story mode) + the upscale polish recipe."""
import json
import pytest
from modules import facts_writer as fw
from modules import upscaler


@pytest.fixture
def story(tmp_path, monkeypatch):
    monkeypatch.setattr(fw, "OUTPUTS_DIR", tmp_path)
    s = {"facts_id": "T1", "title": "T", "beats": [
        {"narration": "a", "image_prompt": "old-img"},
        {"narration": "b", "image_prompt": "old-img2"}]}
    (tmp_path / "facts_T1.json").write_text(json.dumps(s), encoding="utf-8")
    return tmp_path


def test_edit_image_prompt_persists(story):
    assert fw.set_beat_prompt("T1", 0, "image_prompt", "new backdrop")
    got = fw.load_facts("T1")
    assert got["beats"][0]["image_prompt"] == "new backdrop"
    assert got["beats"][1]["image_prompt"] == "old-img2"      # untouched


def test_edit_motion_and_scene(story):
    fw.set_beat_prompt("T1", 1, "motion_prompt", "slow push in")
    fw.set_beat_prompt("T1", 1, "mascot_scene", "mascot in a lab coat pointing")
    b = fw.load_facts("T1")["beats"][1]
    assert b["motion_prompt"] == "slow push in"
    assert b["mascot_scene"] == "mascot in a lab coat pointing"


def test_bad_field_rejected(story):
    with pytest.raises(ValueError):
        fw.set_beat_prompt("T1", 0, "narration", "hack")


def test_bad_index_and_id(story):
    assert fw.set_beat_prompt("T1", 9, "image_prompt", "x") is False
    assert fw.set_beat_prompt("NOPE", 0, "image_prompt", "x") is False


def test_polish_recipe_targets_1080_short_side_and_temporal_denoise():
    """The raw 4x looked clay/morphing; polish = supersample down + temporal
    denoise. Short side (not fixed WxH) so it fits 9x16 / 16x9 / 1x1."""
    assert upscaler.UPSCALE_TARGET_SHORT == 1080
    parts = upscaler.UPSCALE_DENOISE.split(":")
    assert len(parts) == 4
    luma_tmp = float(parts[2])
    assert luma_tmp >= 8, "temporal term must be strong enough to kill the crawl"
