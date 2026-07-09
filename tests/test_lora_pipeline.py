"""No-GPU tests for the character-LoRA pipeline (store, captioner, trainer config)."""

import json
import sys
from pathlib import Path

import pytest

_AGENT = Path(__file__).parent.parent.resolve()
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

from modules.lora import store, captioner, trainer


# ----------------------------------------------------------------- naming
def test_slug_and_trigger():
    assert store.slug("Lily Anne") == "lily_anne"
    assert store.slug("  R2-D2!! ") == "r2_d2"
    assert store.trigger_for("Pip") == "rexj_pip"
    assert store.slug("") == "char"


# ----------------------------------------------------------------- captioner
def test_captioner_writes_sidecars(tmp_path):
    for n in ("a.png", "b.jpg", "c.txtonly.webp"):
        (tmp_path / n).write_bytes(b"x")
    (tmp_path / "notimage.json").write_text("{}")
    n = captioner.caption_dataset(tmp_path, "rexj_pip", "7yo boy, brass goggles.")
    assert n == 3
    # caption keeps trigger + class only (look stays out so trigger carries it)
    assert (tmp_path / "a.txt").read_text() == "rexj_pip, a boy"
    assert captioner.count_images(tmp_path) == 3
    assert captioner.class_from_look("a 7-year-old girl with a hat") == "girl"
    assert captioner.class_from_look("glowing orb") == "character"


# ----------------------------------------------------------------- store
@pytest.fixture
def redirect_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORE_DIR", tmp_path / "store")
    monkeypatch.setattr(store, "MANIFEST", tmp_path / "store" / "manifest.json")
    monkeypatch.setattr(store, "COMFY_LORA_DIR", tmp_path / "comfy")
    monkeypatch.setattr(store, "DATASETS_DIR", tmp_path / "datasets")
    return tmp_path


def test_register_copies_and_versions(redirect_store, tmp_path):
    src = tmp_path / "rexj_pip.safetensors"
    src.write_bytes(b"weights")
    e1 = store.register("Pip", src, source_script="s1")
    assert e1["trigger"] == "rexj_pip"
    assert e1["version"] == 1
    assert (store.STORE_DIR / "rexj_pip.safetensors").exists()
    assert (store.COMFY_LORA_DIR / "rexj_pip.safetensors").exists()
    # re-register bumps version
    e2 = store.register("Pip", src)
    assert e2["version"] == 2
    assert store.get("pip")["version"] == 2


def test_resolve_for_prompt(redirect_store, tmp_path):
    src = tmp_path / "x.safetensors"; src.write_bytes(b"w")
    store.register("Pip", src)
    store.register("Maya", src)
    hits = store.resolve_for_prompt("Pip waves at the door")
    assert [h["name"] for h in hits] == ["Pip"]
    # word-boundary: 'Pippa' must NOT match 'Pip'
    assert store.resolve_for_prompt("Pippa runs") == []
    assert store.resolve_for_prompt("empty street, no one") == []


def test_remove(redirect_store, tmp_path):
    src = tmp_path / "x.safetensors"; src.write_bytes(b"w")
    store.register("Pip", src)
    assert store.remove("Pip") is True
    assert store.get("pip") is None
    assert store.remove("Pip") is False


# ----------------------------------------------------------------- trainer config
def test_build_config_substitutes(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(trainer, "OUTPUT_DIR", tmp_path / "out")
    ds = tmp_path / "datasets" / "pip"; ds.mkdir(parents=True)
    cfg = trainer.build_config("pip", "rexj_pip", ds, steps=1200, rank=24)
    text = cfg.read_text()
    assert "rexj_pip" in text
    assert "steps: 1200" in text
    assert "linear: 24" in text
    assert "quantize: true" in text
    assert "low_vram: true" in text
    assert "black-forest-labs/FLUX.1-dev" in text
    assert str(ds).replace("\\", "/") in text
