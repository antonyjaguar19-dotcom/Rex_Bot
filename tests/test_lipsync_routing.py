"""P4 lip-sync: speaker→lipsync gating, registry lookup, default-off safety.
No GPU / ComfyUI / backend instantiation."""
import sys
from pathlib import Path

_AGENT = Path(__file__).parent.parent.resolve()
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

from modules import voice_casting as vc
from modules import model_registry as mr
from modules import runtime_settings as rs


def test_is_character_speaker():
    assert vc.is_character_speaker("Terry") is True
    assert vc.is_character_speaker("narrator") is False
    assert vc.is_character_speaker("") is False
    assert vc.is_character_speaker(None) is False


def test_lipsync_disabled_by_default():
    # Safety: lip-sync must be opt-in (S2V weights/workflow may not be installed).
    assert isinstance(rs.get_lipsync_enabled(), bool)
    assert rs.get_lipsync_backend_id() == "comfyui_wan22_s2v"


def test_s2v_registered_in_models_json():
    cfg = mr.get_available("video_backend", "comfyui_wan22_s2v")
    assert cfg is not None
    assert cfg["module_path"] == "modules.video_backends.comfyui_wan_s2v"
    assert cfg["_id"] == "comfyui_wan22_s2v"


def test_get_available_unknown_returns_none():
    assert mr.get_available("video_backend", "does_not_exist") is None
    assert mr.get_available("nonsense_type", "x") is None
