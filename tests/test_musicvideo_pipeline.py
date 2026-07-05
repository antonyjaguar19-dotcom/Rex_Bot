"""No-GPU tests for the music-video pipeline (song schema, ACE workflow injection,
backend factory, runtime accessors, safety)."""

import json
import sys
from pathlib import Path

import pytest

_AGENT = Path(__file__).parent.parent.resolve()
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

PROJECT_ROOT = _AGENT.parent


# ----------------------------------------------------------------- import smoke
def test_imports():
    import modules.audio_backend            # noqa: F401
    import modules.audio_backends.comfyui_ace_step  # noqa: F401
    import modules.song_generator           # noqa: F401
    import modules.musicvideo_assembly      # noqa: F401
    import modules.musicvideo_pipeline      # noqa: F401


# ----------------------------------------------------------------- runtime settings
def test_pipeline_mode_roundtrip():
    from modules import runtime_settings as rs
    orig = rs.get_pipeline_mode()
    try:
        rs.set_pipeline_mode("music_video")
        assert rs.get_pipeline_mode() == "music_video"
        rs.set_pipeline_mode("story")
        assert rs.get_pipeline_mode() == "story"
        with pytest.raises(ValueError):
            rs.set_pipeline_mode("nonsense")
    finally:
        rs.set_pipeline_mode(orig)


def test_song_overrides_roundtrip():
    from modules import runtime_settings as rs
    rs.set_song_style_override("metal")
    assert rs.get_song_style_override() == "metal"
    rs.clear_song_style_override()
    assert rs.get_song_style_override() is None
    rs.set_visual_style_override("spectrum")
    assert rs.get_visual_style_override() == "spectrum"
    rs.clear_visual_style_override()
    with pytest.raises(ValueError):
        rs.set_vocal_type_override("kazoo")


# ----------------------------------------------------------------- song schema
def _fake_song(scenes=3):
    return {
        "title": "Test", "theme": "rain",
        "song_style": "pop", "vocal_type": "female",
        "ace_tags": "pop, female vocal", "bpm": 999, "keyscale": "",
        "visual_style": "cartoon", "visual_world": "neon city",
        "lyrics": "[Verse 1]\nla la la\n[Chorus]\noh oh",
        "scenes": [{"section": f"s{i}", "image_prompt": f"scene {i}"} for i in range(scenes)],
    }


def test_validate_and_default_scene_timing():
    from modules import song_generator as sgn
    song = sgn._validate_and_default(_fake_song(3), duration_sec=120, n_scenes=15)
    # bpm clamped, keyscale defaulted
    assert song["bpm"] == 200
    assert song["keyscale"] == "C major"
    # scene seconds sum ~= duration
    total = sum(s["seconds"] for s in song["scenes"])
    assert abs(total - 120) < 0.05
    assert len(song["scenes"]) == 3


def test_validate_trims_scene_overflow():
    from modules import song_generator as sgn
    song = sgn._validate_and_default(_fake_song(30), duration_sec=120, n_scenes=15)
    assert len(song["scenes"]) == 15


def test_validate_rejects_no_scenes():
    from modules import song_generator as sgn
    bad = _fake_song(0)
    with pytest.raises(ValueError):
        sgn._validate_and_default(bad, duration_sec=60, n_scenes=8)


def test_invalid_enums_default():
    from modules import song_generator as sgn
    s = _fake_song(2)
    s["song_style"] = "polka-core"
    s["vocal_type"] = "robot"
    s["visual_style"] = "hologram"
    out = sgn._validate_and_default(s, duration_sec=60, n_scenes=8)
    assert out["song_style"] == "pop"
    assert out["vocal_type"] == "auto"
    assert out["visual_style"] == "cartoon"


# ----------------------------------------------------------------- ACE workflow injection
def test_ace_workflow_injection():
    from modules.audio_backends.comfyui_ace_step import Backend
    cfg = {
        "_id": "comfyui_ace_step",
        "server_url": "http://127.0.0.1:8188",
        "workflow_file": "audio_ace_step_1_5_split.json",
        "steps": 8, "cfg": 1, "default_duration": 120,
    }
    be = Backend(cfg)
    wf = be._build_workflow(
        tags="rock, male vocal", lyrics="[Verse 1]\nhello",
        duration=90.0, bpm=128, keyscale="A minor", language="en", seed=42,
    )
    enc = [n for n in wf.values() if isinstance(n, dict)
           and n.get("class_type") == "TextEncodeAceStepAudio1.5"][0]["inputs"]
    assert enc["tags"] == "rock, male vocal"
    assert enc["lyrics"].startswith("[Verse 1]")
    assert enc["duration"] == 90.0
    assert enc["bpm"] == 128
    assert enc["keyscale"] == "A minor"
    assert enc["seed"] == 42
    latent = [n for n in wf.values() if isinstance(n, dict)
              and n.get("class_type") == "EmptyAceStep1.5LatentAudio"][0]["inputs"]
    assert latent["seconds"] == 90.0
    ks = [n for n in wf.values() if isinstance(n, dict)
          and n.get("class_type") == "KSampler"][0]["inputs"]
    assert ks["seed"] == 42
    assert ks["steps"] == 8


# ----------------------------------------------------------------- backend factory
def test_audio_backend_factory_loads():
    from modules import audio_backend
    be = audio_backend.get_active_backend()
    assert be.backend_id == "comfyui_ace_step"
    # contract method exists
    assert hasattr(be, "generate") and hasattr(be, "health_check")


# ----------------------------------------------------------------- scene prompt build
def test_scene_prompt_includes_style_suffix():
    from modules import musicvideo_pipeline as mvp
    song = _fake_song(1)
    song["visual_style"] = "spectrum"
    p = mvp._scene_prompt(song, song["scenes"][0])
    assert "scene 0" in p
    assert "neon city" in p
    # spectrum style suffix mentions abstract music visual
    assert "abstract" in p.lower()


def test_vocal_tags_appends_vocal():
    from modules import musicvideo_pipeline as mvp
    song = {"ace_tags": "lofi beats", "vocal_type": "male"}
    assert "male vocal" in mvp._vocal_tags(song)
    song2 = {"ace_tags": "ambient", "vocal_type": "instrumental"}
    assert "no vocals" in mvp._vocal_tags(song2)


# ----------------------------------------------------------------- safety reuse
def test_safety_filter_accepts_song_dict():
    from modules import safety_filter as sf
    ok, blocked, warnings = sf.check_safety({
        "title": "Sunny Day", "lyrics": "happy clouds and rainbows",
        "scenes": [{"image_prompt": "a bright meadow"}],
    })
    assert ok is True
    assert blocked == []


# ------------------------------------------------- lyric caption sync (WhisperX)
def test_lyric_caption_events_from_aligner(monkeypatch):
    """align_lyrics returns (text, start, end) spans; _lyric_caption_events must
    reorder to (start, end, text) events without crashing (regression: a float
    was passed where text was expected -> AttributeError)."""
    from modules import musicvideo_assembly as mva
    from modules import lyric_aligner

    monkeypatch.setattr(
        lyric_aligner, "align_lyrics",
        lambda audio, lines: [("first line", 0.5, 2.0), ("second line", 2.3, 4.1)],
    )
    events = mva._lyric_caption_events(
        song_dur=5.0, lyrics="[Verse]\nfirst line\nsecond line",
        song_audio=Path("fake.wav"),
    )
    assert events, "expected caption events from aligned spans"
    for t0, t1, text in events:
        assert isinstance(t0, float) and isinstance(t1, float)
        assert isinstance(text, str) and text
    # real alignment kept (starts after 0), not the 0..song_dur fallback spread
    assert events[0][0] == 0.5


def test_lyric_caption_events_fallback_when_no_alignment(monkeypatch):
    """No aligner result -> proportional char-length spread across the song
    (0..song_dur), never a crash or empty captions."""
    from modules import musicvideo_assembly as mva
    from modules import lyric_aligner

    monkeypatch.setattr(lyric_aligner, "align_lyrics", lambda audio, lines: None)
    events = mva._lyric_caption_events(
        song_dur=6.0, lyrics="[Chorus]\nla la la\nsing along",
        song_audio=Path("fake.wav"),
    )
    assert events
    assert events[0][0] == 0.0
    assert abs(events[-1][1] - 6.0) < 0.01
