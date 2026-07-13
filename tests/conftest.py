import sys
from pathlib import Path

import pytest

# Make `modules` importable exactly the way claw_bot.py does it
# (sys.path.insert of 02_Agent — never `python -m modules.X`).
AGENT_ROOT = Path(__file__).parent.parent.resolve()
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))


@pytest.fixture(autouse=True)
def _no_gpu_mascot(monkeypatch, tmp_path_factory):
    """The test suite must never render on the GPU.

    publish_kit.attach() prefers mascot artwork, and mascot art is a real USO
    render against ComfyUI. The moment a mascot image landed in
    02_Agent/assets/, the suite started calling the GPU: runtime went from
    seconds to minutes, and results depended on ComfyUI being up.

    Point the mascot at an empty directory for every test. Tests that WANT the
    mascot path (tests/test_mascot.py) override ASSETS_DIR themselves, and stub
    the renderer.
    """
    try:
        import modules.mascot as mas
        import modules.runtime_settings as rs
    except Exception:
        return
    empty = tmp_path_factory.mktemp("no_assets")
    monkeypatch.setattr(mas, "ASSETS_DIR", empty, raising=False)
    # ...and the same for the VOICE. The mascot's engine is the clone now, so a
    # test that voices a beat reaches for the reference clip — and the default
    # points at the owner's real recording in 02_Agent/assets. That drags the live
    # Chatterbox bridge (and the GPU) into a unit test, and puts a person's voice
    # sample in the loop. Point it at nothing: with no clip to clone, the pipeline
    # falls back exactly as it does on a machine that has none.
    monkeypatch.setattr(rs, "DEFAULT_MASCOT_VOICE_REF", empty / "mascot_voice.wav",
                        raising=False)


@pytest.fixture(autouse=True)
def _isolated_facts_memory(monkeypatch, tmp_path_factory):
    """No test may write to the real facts memory.

    facts_writer now records every fact it writes, so a test that generates a reel
    would put "Here is an interesting thing about bees number 1" on the live
    do-not-repeat list — poisoning the topic on the owner's machine. Tests that
    exercise the memory point it somewhere themselves; this is the floor.
    """
    try:
        import modules.facts_memory as fm
    except Exception:
        return
    d = tmp_path_factory.mktemp("facts_memory")
    monkeypatch.setattr(fm, "MEMORY_PATH", d / "_memory.json", raising=False)
    monkeypatch.setattr(fm, "PROJECT_ROOT", d, raising=False)
