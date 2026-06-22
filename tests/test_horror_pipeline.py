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


def test_scene_prompt_has_photoreal_suffix():
    from modules import horrorstory_pipeline as hsp
    p = hsp._scene_prompt({"image_prompt": "a fog-drowned pier at midnight"})
    assert "fog-drowned pier" in p
    assert "photorealistic" in p.lower()


def test_writer_constants_bounded():
    from modules import horror_writer as hw
    assert hw.MAX_BEATS <= 200
    assert "deep" in hw.DEFAULT_VOICE_DESIGN.lower()
