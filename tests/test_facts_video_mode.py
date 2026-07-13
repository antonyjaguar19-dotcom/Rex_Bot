"""A facts reel is ANIMATED. That is no longer a setting.

An e2e run once "passed" while silently rendering Ken Burns stills, because the
harness forced animate=False. It proved nothing about the four-model collision
(Ollama -> Z-Image -> Wan -> Qwen) it existed to exercise. The dashboard toggle is
gone too, so a stale `kenburns` left in the settings file would have shipped a
slideshow with nothing on screen to say why — hence: the setting is not read.

Ken Burns survives inside facts_pipeline as the OOM rescue. A rescue, not a choice.
"""

import json

import pytest

from modules import runtime_settings as rs


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "SETTINGS_PATH", tmp_path / "runtime_settings.json")
    return rs


def test_facts_are_always_animated(settings):
    assert settings.get_facts_video_mode() == "wan"


def test_a_stale_kenburns_on_disk_cannot_downgrade_a_reel(settings):
    """The toggle that wrote this is gone; the value must not outlive it."""
    settings.SETTINGS_PATH.write_text(json.dumps({"facts_video_mode": "kenburns"}),
                                      encoding="utf-8")
    assert settings.get_facts_video_mode() == "wan"


def test_the_setter_is_gone():
    assert not hasattr(rs, "set_facts_video_mode")


def test_pipeline_reads_the_setting_when_animate_is_none():
    """render_facts(animate=None) must defer to runtime_settings, not guess.

    The programmatic override stays: a harness may still ask for stills.
    """
    import inspect

    import modules.facts_pipeline as fp
    src = inspect.getsource(fp.render_facts)
    assert 'rs.get_facts_video_mode() == "wan"' in src
    assert "if animate is None" in src
