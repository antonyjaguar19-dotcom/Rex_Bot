"""
Claw Bot — Audio Backend Abstract Interface

Every song/audio backend (ACE-Step, future MusicGen-song, etc.) implements this
contract. The rest of the bot calls `backend.generate(...)` and never cares which
model is active — same "USB port" pattern as image_backend.py / video_backend.py.

Used by the MUSIC VIDEO pipeline only. The story pipeline does not touch this.
"""

import importlib
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

# Ensure 02_Agent is on sys.path so `modules.*` always resolves to the same
# class objects (critical for issubclass()).
_AGENT_DIR = Path(__file__).parent.parent.resolve()
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from modules import model_registry   # noqa: E402

log = logging.getLogger("claw_bot.audio_backend")


# ==============================================================================
# ABSTRACT BASE CLASS
# ==============================================================================

class AudioBackend(ABC):
    """Contract every song/audio backend implements."""

    def __init__(self, config: dict):
        self.config = config
        self.backend_id = config.get("_id", "unknown")

    @abstractmethod
    def generate(
        self,
        tags: str,
        lyrics: str,
        duration_sec: float = 120.0,
        bpm: Optional[int] = None,
        keyscale: Optional[str] = None,
        language: str = "en",
        seed: Optional[int] = None,
        output_filename: Optional[str] = None,
    ) -> Path:
        """Render a song and return the path to the saved audio file (mp3/wav)."""
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> tuple[bool, str]:
        raise NotImplementedError


# ==============================================================================
# FACTORY — loads the active backend from the registry
# ==============================================================================

def get_active_backend() -> AudioBackend:
    cfg = model_registry.get_active("audio_backend")
    if cfg is None:
        raise RuntimeError(
            "No audio_backend configured in models.json. "
            "Add an 'audio_backend' section with an active entry."
        )
    module_path = cfg.get("module_path")
    if not module_path:
        raise ValueError(f"Audio backend {cfg.get('_id')} has no 'module_path' in config")

    log.info(f"Loading audio backend: {cfg['_id']} from {module_path}")

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"Could not import audio backend module '{module_path}'. Original: {e}"
        )

    if not hasattr(module, "Backend"):
        raise RuntimeError(f"Module '{module_path}' does not define a class named 'Backend'.")

    backend_class = module.Backend
    if not issubclass(backend_class, AudioBackend):
        raise TypeError(
            f"{module_path}.Backend must inherit from AudioBackend. "
            f"(Got MRO: {[c.__module__ + '.' + c.__name__ for c in backend_class.__mro__]})"
        )

    return backend_class(cfg)


# ==============================================================================
# Standalone test
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    backend = get_active_backend()
    print(f"Loaded backend: {backend.backend_id}")
    ok, msg = backend.health_check()
    print(f"Health: {'OK' if ok else 'FAIL'} — {msg}")
