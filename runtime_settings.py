"""
Claw Bot — Runtime Settings

Persistent user overrides for style, resolution, steps, cfg.
Stored as JSON so they survive bot restarts.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import model_registry

log = logging.getLogger("claw_bot.runtime_settings")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SETTINGS_PATH = PROJECT_ROOT / "05_Config" / "runtime_settings.json"


# ==============================================================================
# LOAD / SAVE
# ==============================================================================

def _load() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"Could not read runtime_settings.json: {e}")
        return {}


def _save(data: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ==============================================================================
# INDIVIDUAL ACCESSORS
# ==============================================================================

def get_style_override() -> Optional[str]:
    """Return user-set default style, or None if not overridden."""
    return _load().get("style")


def set_style_override(style_id: str) -> None:
    data = _load()
    data["style"] = style_id
    _save(data)
    log.info(f"Style override set: {style_id}")


def clear_style_override() -> None:
    data = _load()
    data.pop("style", None)
    _save(data)
    log.info("Style override cleared (reverting to LLM choice)")


def get_resolution_override() -> Optional[str]:
    return _load().get("aspect_ratio")


def set_resolution_override(aspect: str) -> None:
    data = _load()
    data["aspect_ratio"] = aspect
    _save(data)
    log.info(f"Aspect ratio override set: {aspect}")


def get_steps_override() -> Optional[int]:
    val = _load().get("steps")
    return int(val) if val is not None else None


def set_steps_override(steps: int) -> None:
    data = _load()
    data["steps"] = int(steps)
    _save(data)
    log.info(f"Steps override set: {steps}")


def clear_steps_override() -> None:
    data = _load()
    data.pop("steps", None)
    _save(data)


def get_cfg_override() -> Optional[float]:
    val = _load().get("cfg")
    return float(val) if val is not None else None


def set_cfg_override(cfg: float) -> None:
    data = _load()
    data["cfg"] = float(cfg)
    _save(data)
    log.info(f"CFG override set: {cfg}")


def clear_cfg_override() -> None:
    data = _load()
    data.pop("cfg", None)
    _save(data)


def get_all_overrides() -> dict:
    """Return all active overrides."""
    return _load()


def clear_all_overrides() -> None:
    _save({})
    log.info("All runtime overrides cleared")

# --- Voice (Kokoro TTS) ---

def get_voice_override() -> Optional[str]:
    return _load().get("voice")


def set_voice_override(voice_id: str) -> None:
    data = _load()
    data["voice"] = voice_id
    _save(data)
    log.info(f"Voice override set: {voice_id}")


def clear_voice_override() -> None:
    data = _load()
    data.pop("voice", None)
    _save(data)


# --- Clip length (seconds) ---

def get_clip_length_override() -> Optional[float]:
    val = _load().get("clip_length")
    return float(val) if val is not None else None


def set_clip_length_override(seconds: float) -> None:
    data = _load()
    data["clip_length"] = float(seconds)
    _save(data)
    log.info(f"Clip length override set: {seconds}s")


def clear_clip_length_override() -> None:
    data = _load()
    data.pop("clip_length", None)
    _save(data)


# --- Sync mode (strict | loose) ---

def get_sync_mode_override() -> Optional[str]:
    return _load().get("sync_mode")


def set_sync_mode_override(mode: str) -> None:
    if mode not in ("strict", "loose"):
        raise ValueError(f"Invalid sync_mode: {mode!r}. Must be 'strict' or 'loose'.")
    data = _load()
    data["sync_mode"] = mode
    _save(data)
    log.info(f"Sync mode override set: {mode}")


def clear_sync_mode_override() -> None:
    data = _load()
    data.pop("sync_mode", None)
    _save(data)
    
# --- Upscale toggle ---

def get_upscale_enabled() -> bool:
    """Whether upscale runs in auto-assembly. Default True."""
    val = _load().get("upscale_enabled")
    return True if val is None else bool(val)


def set_upscale_enabled(enabled: bool) -> None:
    data = _load()
    data["upscale_enabled"] = bool(enabled)
    _save(data)
    log.info(f"Upscale enabled: {enabled}")


# --- Music mood ---

def get_music_mood_override() -> Optional[str]:
    return _load().get("music_mood")


def set_music_mood_override(mood: str) -> None:
    from modules.music_generator import VALID_MOODS
    if mood not in VALID_MOODS:
        raise ValueError(f"Invalid music_mood: {mood!r}. Must be one of {VALID_MOODS}.")
    data = _load()
    data["music_mood"] = mood
    _save(data)
    log.info(f"Music mood override set: {mood}")


def clear_music_mood_override() -> None:
    data = _load()
    data.pop("music_mood", None)
    _save(data)


# --- Character reference mode (cast-sheet anchoring) ---

def get_reference_mode_enabled() -> bool:
    """Whether storyboards use a cast-sheet reference latent for character consistency.
    Default False — opt in with !set_reference_mode on."""
    val = _load().get("reference_mode")
    return False if val is None else bool(val)


def set_reference_mode_enabled(enabled: bool) -> None:
    data = _load()
    data["reference_mode"] = bool(enabled)
    _save(data)
    log.info(f"Reference mode: {enabled}")


# ==============================================================================
# RESOLVED VALUES — what the pipeline actually uses
# ==============================================================================

def get_effective_style() -> str:
    """User override > registry default."""
    override = get_style_override()
    if override:
        return override
    return "storybook"  # fallback default


def get_effective_aspect_ratio() -> str:
    override = get_resolution_override()
    if override:
        return override
    defaults = model_registry.get_image_defaults()
    return defaults.get("aspect_ratio", "16:9")


def get_effective_steps() -> int:
    """User override > backend config > 8."""
    override = get_steps_override()
    if override is not None:
        return override
    cfg = model_registry.get_active("image_backend")
    return int(cfg.get("steps", 8))


def get_effective_cfg() -> float:
    override = get_cfg_override()
    if override is not None:
        return override
    cfg = model_registry.get_active("image_backend")
    return float(cfg.get("cfg", 1.0))

def get_effective_voice() -> str:
    """User override > Kokoro default ('af_heart')."""
    override = get_voice_override()
    if override:
        return override
    return "af_heart"


def get_effective_sync_mode() -> str:
    override = get_sync_mode_override()
    if override:
        return override
    return "strict"