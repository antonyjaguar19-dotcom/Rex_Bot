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


def test_mascot_reel_still_gets_a_title_and_thumbnail(tmp_path, monkeypatch):
    """The mascot branch used to `return out` straight out of render_facts, which
    skipped _attach_publish_kit — a finished mascot reel shipped with no title and
    no thumbnail beside it, and nothing said a word."""
    from modules import facts_pipeline as fp
    import modules.runtime_settings as rs

    monkeypatch.setattr(rs, "get_facts_mascot_mode", lambda: True)
    monkeypatch.setattr(fp, "STILLS_DIR", tmp_path)
    (tmp_path / "facts_T9").mkdir()
    (tmp_path / "facts_T9" / "still_00.png").write_bytes(b"")

    monkeypatch.setattr(fp, "_render_facts_mascot",
                        lambda *a, **k: {"9x16": tmp_path / "reel.mp4"})
    called = {}
    monkeypatch.setattr(fp, "_attach_publish_kit",
                        lambda story, out, stills: called.setdefault(
                            "kit", {"title": "T", "stills": len(stills)}) or
                        {"title": "T"})

    story = {"facts_id": "T9", "beats": [{"narration": "a"}]}
    out = fp.render_facts(story, aspect="9x16")

    assert called.get("kit"), "the publish kit must run for a mascot reel too"
    assert out["title"] == "T"
    assert called["kit"]["stills"] == 1, "the mascot stills are the thumbnail source"
