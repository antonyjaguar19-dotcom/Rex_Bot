"""Mascot mode must not downgrade behind your back.

A run with Mascot ON, ComfyUI asleep, used to print "mascot mode off: Cannot
connect to ComfyUI" and then render abstract backdrops anyway — a reel that was
not the one asked for, discovered only after it finished. Now it raises, and it
raises BEFORE the story is written.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import facts_pipeline as fp   # noqa: E402


DEAD = (False, "comfyui_uso: Cannot connect to ComfyUI at http://127.0.0.1:8188.")


def test_preflight_raises_when_mascot_on_and_backend_dead(monkeypatch):
    from modules import mascot, runtime_settings as rs
    monkeypatch.setattr(rs, "get_facts_mascot_mode", lambda: True)
    monkeypatch.setattr(mascot, "is_available", lambda: DEAD)
    with pytest.raises(RuntimeError) as e:
        fp.preflight()
    msg = str(e.value)
    assert "Mascot mode is ON" in msg
    assert "Start ComfyUI" in msg          # tells you what to actually do


def test_preflight_raises_when_backdrops_backend_dead(monkeypatch):
    """Without the mascot a dead backend gave a reel of plain gradient cards."""
    from modules import mascot, runtime_settings as rs
    monkeypatch.setattr(rs, "get_facts_mascot_mode", lambda: False)
    monkeypatch.setattr(mascot, "backend_healthy", lambda: DEAD)
    with pytest.raises(RuntimeError) as e:
        fp.preflight()
    assert "gradient cards" in str(e.value)


def test_preflight_passes_when_healthy(monkeypatch):
    from modules import mascot, runtime_settings as rs
    monkeypatch.setattr(rs, "get_facts_mascot_mode", lambda: True)
    monkeypatch.setattr(mascot, "is_available", lambda: (True, "mascot.png + qwen"))
    seen = []
    fp.preflight(seen.append)
    assert seen and "preflight" in seen[0]


def test_render_raises_rather_than_falling_back(monkeypatch, tmp_path):
    """Even past preflight, a mascot render that yields nothing is an error —
    it must not silently continue into the backdrop path."""
    from modules import runtime_settings as rs
    monkeypatch.setattr(rs, "get_facts_mascot_mode", lambda: True)
    monkeypatch.setattr(fp, "preflight", lambda _p=None: None)
    monkeypatch.setattr(fp, "_render_facts_mascot",
                        lambda *a, **k: None)          # produced nothing
    story = {"facts_id": "t1", "beats": [{"kind": "fact", "narration": "hi"}]}
    with pytest.raises(RuntimeError) as e:
        fp.render_facts(story)
    assert "Mascot mode is ON" in str(e.value)
