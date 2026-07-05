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
def test_lyric_caption_events_uses_vocal_windows(monkeypatch):
    """Detected vocal windows (real singing regions, intro excluded) -> lyrics
    placed only inside them, NOT starting at 0 during instrumental intro."""
    from modules import musicvideo_assembly as mva
    from modules import lyric_aligner

    # song is 30s but vocals only 12..28s (12s instrumental intro)
    monkeypatch.setattr(lyric_aligner, "get_vocal_windows",
                        lambda audio: [(12.0, 28.0)])
    events = mva._lyric_caption_events(
        song_dur=30.0, lyrics="[Verse]\nfirst line here\nsecond line here",
        song_audio=Path("fake.wav"),
    )
    assert events
    # nothing before the vocals start (no captions on the intro music)
    assert events[0][0] >= 12.0 - 1e-6
    assert events[-1][1] <= 28.0 + 1e-6


def test_lyric_caption_events_guards_sparse_detection(monkeypatch):
    """Whisper barely heard the vocals (0.3s window) on a 17-line song ->
    detection is unreliable, must fall back to proportional spread across the
    whole song, NOT cram every line into 0.3s."""
    from modules import musicvideo_assembly as mva
    from modules import lyric_aligner

    monkeypatch.setattr(lyric_aligner, "get_vocal_windows",
                        lambda audio: [(37.5, 37.8)])
    lyr = "[Verse]\n" + "\n".join(f"line number {i}" for i in range(17))
    events = mva._lyric_caption_events(song_dur=40.0, lyrics=lyr,
                                       song_audio=Path("fake.wav"))
    assert events
    # fallback spans the whole song, not the 0.3s window
    assert events[0][0] == 0.0
    assert abs(events[-1][1] - 40.0) < 0.01


def test_lyric_caption_events_fallback_when_no_vocals(monkeypatch):
    """No vocal windows detected -> proportional char-length spread across the
    whole song, never a crash or empty captions."""
    from modules import musicvideo_assembly as mva
    from modules import lyric_aligner

    monkeypatch.setattr(lyric_aligner, "get_vocal_windows", lambda audio: None)
    events = mva._lyric_caption_events(
        song_dur=6.0, lyrics="[Chorus]\nla la la\nsing along",
        song_audio=Path("fake.wav"),
    )
    assert events
    assert events[0][0] == 0.0
    assert abs(events[-1][1] - 6.0) < 0.01


def test_group_windows_splits_on_instrumental_gap():
    from modules import lyric_aligner
    # words: a cluster 12-14, then a 5s instrumental gap, then 19-21
    words = [["a", 12.0, 12.5], ["b", 13.0, 14.0], ["c", 19.0, 19.5], ["d", 20.5, 21.0]]
    wins = lyric_aligner.group_windows(words, gap_sec=1.2)
    assert wins == [(12.0, 14.0), (19.0, 21.0)]


def test_distribute_lines_over_windows_stays_inside_and_skips_gaps():
    from modules import subtitles as subs
    lines = ["one", "two", "three", "four"]
    windows = [(12.0, 16.0), (20.0, 24.0)]      # two singing spans, 4s gap
    events = subs.distribute_lines_over_windows(lines, windows)
    assert len(events) == 4
    # every event must fall entirely inside one of the singing windows (never
    # in the 16-20s instrumental gap)
    for t0, t1, _ in events:
        in_a = 12.0 - 1e-6 <= t0 and t1 <= 16.0 + 1e-6
        in_b = 20.0 - 1e-6 <= t0 and t1 <= 24.0 + 1e-6
        assert in_a or in_b, f"event {t0}-{t1} leaked into the instrumental gap"
