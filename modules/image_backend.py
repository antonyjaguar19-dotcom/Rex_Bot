"""
Claw Bot — Image Backend Abstract Interface

Every image backend (Z-Image, SDXL, Flux, etc.) must implement this interface.
That way, the rest of the bot doesn't care which model is active — it just
calls backend.generate(prompt).

This is the "USB port" contract.
"""

import importlib
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

# Ensure 02_Agent folder is on sys.path. This guarantees that no matter how
# this module is invoked (python -m, direct, via another import) we always
# resolve `modules.*` to the same class objects — critical for issubclass().
_AGENT_DIR = Path(__file__).parent.parent.resolve()
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from modules import model_registry   # noqa: E402  (import after sys.path tweak)

log = logging.getLogger("claw_bot.image_backend")


# ==============================================================================
# ABSTRACT BASE CLASS — what every backend must implement
# ==============================================================================

class ImageBackend(ABC):
    """
    Contract every image backend implements.
    """

    def __init__(self, config: dict):
        self.config = config
        self.backend_id = config.get("_id", "unknown")

    @abstractmethod
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        aspect_ratio: str = "16:9",
        output_filename: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Path:
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> tuple[bool, str]:
        raise NotImplementedError

    @abstractmethod
    def format_prompt_for_backend(self, raw_prompt: str) -> str:
        raise NotImplementedError


# ==============================================================================
# FACTORY — loads the active backend from the registry
# ==============================================================================

def _instantiate(cfg: dict) -> ImageBackend:
    """Load + instantiate an image Backend from a registry config dict."""
    module_path = cfg.get("module_path")
    if not module_path:
        raise ValueError(f"Backend {cfg.get('_id')} has no 'module_path' in config")

    log.info(f"Loading image backend: {cfg.get('_id')} from {module_path}")

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"Could not import image backend module '{module_path}'. Original: {e}"
        )

    if not hasattr(module, "Backend"):
        raise RuntimeError(f"Module '{module_path}' does not define a class named 'Backend'.")

    backend_class = module.Backend
    if not issubclass(backend_class, ImageBackend):
        raise TypeError(
            f"{module_path}.Backend must inherit from ImageBackend. "
            f"(Got MRO: {[c.__module__ + '.' + c.__name__ for c in backend_class.__mro__]})"
        )

    return backend_class(cfg)


def get_active_backend() -> ImageBackend:
    return _instantiate(model_registry.get_active("image_backend"))


def get_named_backend(backend_id: str) -> ImageBackend:
    """Load a SPECIFIC image backend by registry id (not the active one).
    Used to pin a consistency backend (e.g. USO) for kids/music without
    changing the global active backend. Raises if the id is unknown."""
    cfg = model_registry.get_available("image_backend", backend_id)
    if not cfg:
        raise ValueError(f"Image backend '{backend_id}' not found in registry")
    return _instantiate(cfg)


# ==============================================================================
# Standalone test
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    backend = get_active_backend()
    print(f"Loaded backend: {backend.backend_id}")
    print(f"Config keys: {list(backend.config.keys())}")
    ok, msg = backend.health_check()
    print(f"Health: {'OK' if ok else 'FAIL'} — {msg}")