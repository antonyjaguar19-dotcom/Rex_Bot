import json

from modules import model_registry as mr


def _write_models(tmp_path, llm_section):
    models = {
        "llm_backend": llm_section,
        "image_backend": {"active": "z", "available": {"z": {}}},
        "video_backend": {"active": "wan", "available": {"wan": {}}},
    }
    p = tmp_path / "models.json"
    p.write_text(json.dumps(models), encoding="utf-8")
    return p


def test_role_present_resolves_to_mapped_model(tmp_path, monkeypatch):
    p = _write_models(tmp_path, {
        "active": "qwen14",
        "roles": {"creative": "qwen30", "structurer": "qwen14"},
        "available": {"qwen14": {"model_id": "14b"}, "qwen30": {"model_id": "30b"}},
    })
    monkeypatch.setattr(mr, "REGISTRY_FILE", p)
    cfg = mr.get_for_role("creative")
    assert cfg["_id"] == "qwen30" and cfg["model_id"] == "30b"


def test_unknown_role_falls_back_to_default_then_active(tmp_path, monkeypatch):
    # No 'default' key, unknown role -> active.
    p = _write_models(tmp_path, {
        "active": "qwen14",
        "roles": {"creative": "qwen30"},
        "available": {"qwen14": {"model_id": "14b"}, "qwen30": {"model_id": "30b"}},
    })
    monkeypatch.setattr(mr, "REGISTRY_FILE", p)
    cfg = mr.get_for_role("nonexistent")
    assert cfg["_id"] == "qwen14"


def test_no_roles_map_falls_back_to_active(tmp_path, monkeypatch):
    p = _write_models(tmp_path, {
        "active": "qwen14",
        "available": {"qwen14": {"model_id": "14b"}},
    })
    monkeypatch.setattr(mr, "REGISTRY_FILE", p)
    cfg = mr.get_for_role("creative")
    assert cfg["_id"] == "qwen14"


def test_role_points_at_missing_entry_falls_back_to_active(tmp_path, monkeypatch):
    p = _write_models(tmp_path, {
        "active": "qwen14",
        "roles": {"creative": "ghost_model"},
        "available": {"qwen14": {"model_id": "14b"}},
    })
    monkeypatch.setattr(mr, "REGISTRY_FILE", p)
    cfg = mr.get_for_role("creative")
    assert cfg["_id"] == "qwen14"
