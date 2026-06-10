import json

from modules import config_check as cc
from modules.file_utils import atomic_write_json


def test_real_configs_are_valid():
    """The configs on this machine must pass — if this fails, the bot
    would refuse to boot."""
    assert cc.validate_configs() == []


def test_broken_active_backend(tmp_path, monkeypatch):
    models = {
        "llm_backend": {"active": "MISSING", "available": {"qwen": {}}},
        "image_backend": {"active": "z", "available": {"z": {}}},
        "video_backend": {"active": "wan", "available": {"wan": {}}},
    }
    atomic_write_json(tmp_path / "models.json", models)
    atomic_write_json(tmp_path / "styles.json",
                      {"default": "storybook", "available": {"storybook": {}}})
    monkeypatch.setattr(cc, "MODELS_PATH", tmp_path / "models.json")
    monkeypatch.setattr(cc, "STYLES_PATH", tmp_path / "styles.json")
    monkeypatch.setattr(cc, "RUNTIME_PATH", tmp_path / "runtime_settings.json")
    errors = cc.validate_configs()
    assert len(errors) == 1 and "llm_backend" in errors[0]


def test_unparseable_json(tmp_path, monkeypatch):
    (tmp_path / "models.json").write_text("{not json", encoding="utf-8")
    atomic_write_json(tmp_path / "styles.json",
                      {"default": "s", "available": {"s": {}}})
    monkeypatch.setattr(cc, "MODELS_PATH", tmp_path / "models.json")
    monkeypatch.setattr(cc, "STYLES_PATH", tmp_path / "styles.json")
    monkeypatch.setattr(cc, "RUNTIME_PATH", tmp_path / "runtime_settings.json")
    errors = cc.validate_configs()
    assert any("not valid JSON" in e for e in errors)


def test_bom_detection(tmp_path, monkeypatch):
    bom_file = tmp_path / "secrets.env"
    bom_file.write_bytes(b"\xef\xbb\xbfDISCORD_BOT_TOKEN=abc\n")
    monkeypatch.setattr(cc, "SECRETS_PATH", bom_file)
    assert cc.warn_on_secrets_bom() is True

    clean = tmp_path / "clean.env"
    clean.write_bytes(b"DISCORD_BOT_TOKEN=abc\n")
    monkeypatch.setattr(cc, "SECRETS_PATH", clean)
    assert cc.warn_on_secrets_bom() is False


def test_disk_space_returns_sane_numbers():
    ok, free_gb = cc.check_disk_space(min_gb=0.001)
    assert ok is True
    assert free_gb > 0
