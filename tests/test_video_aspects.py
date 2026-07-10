"""A render's thumbnails follow the videos it actually produced.

Every caller of attach() took the default ("16x9", "9x16"), so a facts reel —
which only ever exists as 9x16 — paid a second ~25 s Qwen render for a landscape
thumbnail nothing could use, and horror (16x9 only) got a portrait one.
"""

import pytest
from PIL import Image

from modules import publish_kit as pk


def _mp4(d, name):
    p = d / name
    p.write_bytes(b"\x00")
    return p


def test_portrait_only_reel_gets_one_portrait_thumbnail(tmp_path):
    v = _mp4(tmp_path, "facts_20260710_184745_9x16.mp4")
    assert pk.video_aspects(v) == ("9x16",)


def test_landscape_only_story_gets_one_landscape_thumbnail(tmp_path):
    v = _mp4(tmp_path, "horror_20260706_073959_16x9.mp4")
    assert pk.video_aspects(v) == ("16x9",)


def test_a_multi_aspect_render_gets_both(tmp_path):
    v = _mp4(tmp_path, "story_1_9x16.mp4")
    _mp4(tmp_path, "story_1_16x9.mp4")
    _mp4(tmp_path, "story_1_1x1.mp4")
    assert pk.video_aspects(v) == ("16x9", "9x16")


def test_square_alone_is_not_a_thumbnail_aspect(tmp_path):
    """Nothing consumes a 1:1 cover; fall back to the video's own orientation."""
    v = _mp4(tmp_path, "song_1_1x1.mp4")
    assert pk.video_aspects(v) in (("9x16",), ("16x9",))


def test_sibling_discord_copies_do_not_add_aspects(tmp_path):
    v = _mp4(tmp_path, "facts_1_9x16.mp4")
    _mp4(tmp_path, "facts_1_9x16_discord.mp4")
    assert pk.video_aspects(v) == ("9x16",)


def test_an_unsuffixed_video_is_probed(tmp_path, monkeypatch):
    """Manual mode names its own files."""
    monkeypatch.setattr(pk, "_probe_orientation", lambda v: "16x9")
    v = _mp4(tmp_path, "manual_project_final.mp4")
    assert pk.video_aspects(v) == ("16x9",)


def test_probe_failure_assumes_portrait(tmp_path, monkeypatch, caplog):
    """Short-form is the default output of every pipeline here."""
    monkeypatch.setattr(pk.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no ffprobe")))
    with caplog.at_level("WARNING"):
        assert pk._probe_orientation(tmp_path / "x.mp4") == "9x16"


def test_attach_records_and_honours_the_detected_aspects(tmp_path, monkeypatch):
    import modules.runtime_settings as rs
    monkeypatch.setattr(rs, "get_mascot_thumbnails_enabled", lambda: False)
    monkeypatch.setattr(pk, "build_title", lambda *a, **k: "T")
    monkeypatch.setattr(pk, "build_headline", lambda *a, **k: ("T", "T"))
    frame = tmp_path / "still.png"
    Image.new("RGB", (1080, 1920)).save(frame)

    v = _mp4(tmp_path, "facts_1_9x16.mp4")
    kit = pk.attach(v, "T", source_image=frame)
    assert kit["aspects"] == ["9x16"]
    assert "thumb_9x16" in kit
    assert "thumb_16x9" not in kit, "a vertical reel needs no landscape thumbnail"


def test_an_explicit_aspects_argument_still_wins(tmp_path, monkeypatch):
    import modules.runtime_settings as rs
    monkeypatch.setattr(rs, "get_mascot_thumbnails_enabled", lambda: False)
    monkeypatch.setattr(pk, "build_title", lambda *a, **k: "T")
    monkeypatch.setattr(pk, "build_headline", lambda *a, **k: ("T", "T"))
    frame = tmp_path / "still.png"
    Image.new("RGB", (1080, 1920)).save(frame)
    v = _mp4(tmp_path, "facts_1_9x16.mp4")
    kit = pk.attach(v, "T", source_image=frame, aspects=("16x9",))
    assert kit["aspects"] == ["16x9"]
