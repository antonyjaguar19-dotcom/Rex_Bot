"""No-GPU tests for the horror-story pipeline (mode, adult safety, timing, prompts)."""

import sys
from pathlib import Path

import pytest

_AGENT = Path(__file__).parent.parent.resolve()
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))


def test_imports():
    import modules.horror_writer            # noqa: F401
    import modules.horror_assembly          # noqa: F401
    import modules.horrorstory_pipeline     # noqa: F401


def test_mode_includes_horror():
    from modules import runtime_settings as rs
    assert "horror_story" in rs.VALID_PIPELINE_MODES
    orig = rs.get_pipeline_mode()
    try:
        rs.set_pipeline_mode("horror_story")
        assert rs.get_pipeline_mode() == "horror_story"
    finally:
        rs.set_pipeline_mode(orig)


def test_horror_ambient_toggle():
    from modules import runtime_settings as rs
    rs.set_horror_ambient_enabled(False)
    assert rs.get_horror_ambient_enabled() is False
    rs.set_horror_ambient_enabled(True)
    assert rs.get_horror_ambient_enabled() is True


def test_adult_profile_allows_horror_blocks_sexual():
    from modules import safety_filter as sf
    horror = {"beats": [{"narration": "blood pooled by the corpse in the dark",
                         "image_prompt": "a bloody corpse in a cellar"}]}
    ok, blocked, _ = sf.check_safety(horror, profile="adult")
    assert ok is True and blocked == []
    # sexual content still blocked under adult
    bad = {"beats": [{"narration": "explicit sexual scene", "image_prompt": "x"}]}
    ok2, blocked2, _ = sf.check_safety(bad, profile="adult")
    assert ok2 is False and "sexual" in blocked2


def test_kids_profile_still_strict():
    from modules import safety_filter as sf
    ok, blocked, _ = sf.check_safety({"shots": [{"narration": "blood everywhere"}]}, profile="kids")
    assert ok is False and "blood" in blocked


def test_scene_durations_tile_full_timeline():
    from modules.horrorstory_pipeline import _scene_durations
    spans = [("a", 0.0, 2.0), ("b", 2.3, 5.0), ("c", 5.3, 8.0)]
    total = 8.5
    durs = _scene_durations(spans, total)
    assert len(durs) == 3
    assert abs(sum(durs) - total) < 0.01          # covers the whole audio
    assert durs[0] == pytest.approx(2.3)          # next start - this start
    assert durs[-1] == pytest.approx(total - 5.3)  # last runs to the end


def test_scene_prompt_injects_location_and_photoreal():
    from modules import horrorstory_pipeline as hsp
    beat = {"image_prompt": "Mara walks the fog-drowned pier at midnight",
            "location": "Pier", "characters": ["Mara"]}
    loc_map = {"Pier": "a rotting wooden pier swallowed by sea fog"}
    char_look = {"Mara": "34-year-old woman, short dark hair, yellow raincoat"}
    # prompt-token consistency: scene + location + char look + photoreal suffix
    p = hsp._scene_prompt(beat, loc_map, char_look)
    assert "fog-drowned pier" in p
    assert "rotting wooden pier" in p          # location desc injected
    assert "yellow raincoat" in p              # char look injected (prompt token)
    assert "photorealistic" in p.lower()


def test_flux_ckpt_ready_handles_missing(monkeypatch):
    from modules import horrorstory_pipeline as hsp
    from modules import model_registry
    monkeypatch.setattr(model_registry, "get_available", lambda *a, **k: None)
    assert hsp._flux_ckpt_ready() is False


def test_lora_autotrain_importable():
    # Per-character LoRA dropped from the horror pipeline but the training setup
    # is kept for later reuse.
    from modules import lora_autotrain
    assert hasattr(lora_autotrain, "ensure_character_loras")


def test_silence_window_planning(monkeypatch):
    """Windows snap to detected silences + tile the full audio length."""
    from modules import audio_segmenter as seg
    monkeypatch.setattr(seg, "_probe_duration", lambda p: 30.0)
    # pauses near the proportional boundaries (10s, 20s) for 3 equal beats
    monkeypatch.setattr(seg, "detect_silence_midpoints", lambda *a, **k: [9.7, 20.3, 25.0])
    durs = seg.plan_windows_from_silence(__import__("pathlib").Path("x.wav"),
                                         [10, 10, 10])
    assert len(durs) == 3
    assert abs(sum(durs) - 30.0) < 0.05
    # first boundary snapped to 9.7 -> first window ~9.7s
    assert abs(durs[0] - 9.7) < 0.1


def test_continuous_narration_engine_default():
    from modules import runtime_settings as rs
    # default horror engine is a real, runnable engine
    assert rs.get_horror_voice_engine() in ("kokoro", "voxcpm", "qwen")


def test_refine_windows_passes_through_single_sentence(monkeypatch):
    """A window with exactly one sentence already has exact timing — no slicing
    or silence re-detection should happen for it."""
    from modules import audio_segmenter as seg

    def _boom(*a, **k):
        raise AssertionError("should not slice a single-sentence window")
    monkeypatch.setattr(seg, "_extract_slice", _boom)

    events = seg.refine_windows_to_sentences(Path("x.wav"), [(0.0, 3.0, "Just one sentence.")])
    assert events == [(0.0, 3.0, "Just one sentence.")]


def test_refine_windows_splits_multi_sentence_on_real_silence(monkeypatch):
    """A 2-sentence window gets sliced and its internal split snapped to a real
    detected pause instead of guessed by character length."""
    from modules import audio_segmenter as seg

    monkeypatch.setattr(seg, "_extract_slice", lambda wav, t0, t1, out: out)
    # First sentence is much longer text but the REAL audio split (mocked) is 1:1 —
    # proves the refine path uses silence detection, not char-length proportion.
    monkeypatch.setattr(seg, "plan_windows_from_silence", lambda *a, **k: [1.0, 1.0])

    events = seg.refine_windows_to_sentences(
        Path("x.wav"), [(0.0, 2.0, "A very very long first sentence here. Short.")]
    )
    assert len(events) == 2
    assert events[0] == (0.0, 1.0, "A very very long first sentence here.")
    assert events[1] == (1.0, 2.0, "Short.")


def test_refine_windows_falls_back_on_slice_failure(monkeypatch):
    """If ffmpeg slicing/detection fails, fall back to the proportional
    char-length guess rather than dropping the window."""
    from modules import audio_segmenter as seg

    def _boom(*a, **k):
        raise RuntimeError("ffmpeg exploded")
    monkeypatch.setattr(seg, "_extract_slice", _boom)

    events = seg.refine_windows_to_sentences(Path("x.wav"), [(0.0, 10.0, "Hi. Bye.")])
    assert len(events) == 2
    assert events[0][2] == "Hi."
    assert events[1][2] == "Bye."
    assert events[0][0] == 0.0
    assert events[-1][1] == 10.0


def test_writer_constants_bounded():
    from modules import horror_writer as hw
    assert hw.MAX_BEATS <= 200
    assert "deep" in hw.DEFAULT_VOICE_DESIGN.lower()
