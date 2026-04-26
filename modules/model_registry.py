"""
Claw Bot — Model Registry
Central access point for the pluggable model configuration.

All modules that need to know "which image model / LLM / video model is active"
should call this module. Swapping models is a single-file change in
05_Config/models.json — no code changes needed.
"""

import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("claw_bot.registry")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
REGISTRY_FILE = PROJECT_ROOT / "05_Config" / "models.json"


def _load() -> dict:
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(
            f"Model registry missing: {REGISTRY_FILE}. "
            f"Run Phase 4 Step 4.1 setup first."
        )
    return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))


def _save(data: dict):
    REGISTRY_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------- Read API ---------------

def get_active(backend_type: str) -> dict:
    """
    Return the config dict for the currently-active backend of a given type.
    backend_type: 'llm_backend' | 'image_backend' | 'video_backend'
    """
    reg = _load()
    section = reg.get(backend_type)
    if section is None:
        raise KeyError(f"Unknown backend type: {backend_type}")
    active_id = section.get("active")
    if active_id is None:
        return None   # not configured yet (e.g., video_backend in Phase 4)
    cfg = section["available"].get(active_id)
    if cfg is None:
        raise KeyError(f"Active '{active_id}' not found in available list of {backend_type}")
    # Attach the id so adapters know their own name
    cfg = dict(cfg)
    cfg["_id"] = active_id
    return cfg


def list_available(backend_type: str) -> dict:
    """Return all available backends of a type as {id: description}."""
    reg = _load()
    section = reg.get(backend_type, {})
    return {
        k: v.get("description", "")
        for k, v in section.get("available", {}).items()
    }


def get_image_defaults() -> dict:
    return _load().get("image_defaults", {})


def get_resolution(aspect_ratio: str = None) -> tuple[int, int]:
    """Return (width, height) for a given aspect ratio, or system default."""
    defaults = get_image_defaults()
    aspect_ratio = aspect_ratio or defaults.get("aspect_ratio", "16:9")
    res = defaults.get("resolutions", {}).get(aspect_ratio)
    if res is None:
        raise ValueError(f"Unknown aspect ratio: {aspect_ratio}")
    return (res["width"], res["height"])


def get_character_config() -> dict:
    return _load().get("character_consistency", {})


# --------------- Write API ---------------

def set_active(backend_type: str, backend_id: str) -> dict:
    """Switch the active backend. Returns the new active config."""
    reg = _load()
    section = reg.get(backend_type)
    if section is None:
        raise KeyError(f"Unknown backend type: {backend_type}")
    if backend_id not in section["available"]:
        raise KeyError(
            f"Backend '{backend_id}' not in available list for {backend_type}. "
            f"Available: {list(section['available'].keys())}"
        )
    section["active"] = backend_id
    _save(reg)
    log.info(f"Switched {backend_type} -> {backend_id}")
    return get_active(backend_type)


def set_image_default_aspect(aspect_ratio: str):
    """Change the default aspect ratio."""
    reg = _load()
    if aspect_ratio not in reg["image_defaults"]["resolutions"]:
        raise ValueError(f"Unknown aspect ratio: {aspect_ratio}")
    reg["image_defaults"]["aspect_ratio"] = aspect_ratio
    _save(reg)
    log.info(f"Default aspect ratio -> {aspect_ratio}")


# --------------- Standalone test ---------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("Active LLM:", get_active("llm_backend"))
    print()
    print("Active Image Backend:", get_active("image_backend"))
    print()
    print("Image defaults:", get_image_defaults())
    print()
    print("Resolution for 16:9:", get_resolution("16:9"))
    print("Resolution for 9:16:", get_resolution("9:16"))
    print()
    print("Available image backends:")
    for k, v in list_available("image_backend").items():
        print(f"  • {k}: {v}")