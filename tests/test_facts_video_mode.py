"""facts_video_mode defaults to Wan.

An e2e run once "passed" while silently rendering Ken Burns stills, because the
harness forced animate=False. It proved nothing about the four-model collision
(Ollama -> Z-Image -> Wan -> Qwen) it existed to exercise.
"""

import json

import pytest

from modules import runtime_settings as rs


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "SETTINGS_PATH", tmp_path / "runtime_settings.json")
    return rs


def test_default_is_wan(settings):
    assert settings.get_facts_video_mode() == "wan"


def test_explicit_kenburns_is_honoured(settings):
    settings.set_facts_video_mode("kenburns")
    assert settings.get_facts_video_mode() == "kenburns"


def test_garbage_falls_back_to_wan(settings):
    settings.SETTINGS_PATH.write_text(json.dumps({"facts_video_mode": "nonsense"}),
                                      encoding="utf-8")
    assert settings.get_facts_video_mode() == "wan"


def test_pipeline_reads_the_setting_when_animate_is_none(monkeypatch):
    """render_facts(animate=None) must defer to runtime_settings, not guess."""
    import modules.facts_pipeline as fp
    import inspect
    src = inspect.getsource(fp.render_facts)
    assert 'rs.get_facts_video_mode() == "wan"' in src
    assert "if animate is None" in src
