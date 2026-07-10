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
    except Exception:
        return
    empty = tmp_path_factory.mktemp("no_assets")
    monkeypatch.setattr(mas, "ASSETS_DIR", empty, raising=False)
