"""
Claw Bot — Video Backend Abstract Interface

Every video backend (LTX-2, Wan 2.2, future models) must implement this
interface. The rest of the bot doesn't care which model is active — it just
calls backend.generate(prompt, image).

This is the "USB port" contract for video, mirroring image_backend.py.
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

log = logging.getLogger("claw_bot.video_backend")


# ==============================================================================
# ABSTRACT BASE CLASS — what every backend must implement
# ==============================================================================

class VideoBackend(ABC):
    """
    Contract every video backend implements.
    """

    def __init__(self, config: dict):
        self.config = config
        self.backend_id = config.get("_id", "unknown")

    @abstractmethod
    def generate(
        self,
        prompt: str,
        input_image: Path,
        negative_prompt: Optional[str] = None,
        frame_count: int = 81,
        fps: int = 24,
        aspect_ratio: str = "16:9",
        output_filename: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Path:
        """
        Generate a video clip from a starting image + prompt.

        Args:
            prompt: Description of the action/motion in the clip.
            input_image: Path to the starting frame (storyboard image).
            negative_prompt: Things to avoid in generation.
            frame_count: Number of video frames (81 = ~3.4s @ 24fps).
            fps: Frames per second of the output video.
            aspect_ratio: e.g. "16:9".
            output_filename: Optional name for the output MP4.
            seed: Optional fixed seed for reproducibility.

        Returns:
            Path to the generated MP4 file.
        """
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

def build_backend(cfg: dict) -> VideoBackend:
    """Instantiate a video backend from its config dict (must carry
    'module_path' + '_id'). Shared by the active-loader and per-id loaders
    (e.g. the S2V lip-sync backend used only for character shots)."""
    module_path = cfg.get("module_path")
    if not module_path:
        raise ValueError(f"Backend {cfg.get('_id')} has no 'module_path' in config")

    log.info(f"Loading video backend: {cfg.get('_id')} from {module_path}")

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"Could not import video backend module '{module_path}'. Original: {e}"
        )

    if not hasattr(module, "Backend"):
        raise RuntimeError(f"Module '{module_path}' does not define a class named 'Backend'.")

    backend_class = module.Backend
    if not issubclass(backend_class, VideoBackend):
        raise TypeError(
            f"{module_path}.Backend must inherit from VideoBackend. "
            f"(Got MRO: {[c.__module__ + '.' + c.__name__ for c in backend_class.__mro__]})"
        )

    return backend_class(cfg)


def get_active_backend() -> VideoBackend:
    return build_backend(model_registry.get_active("video_backend"))


def get_named_backend(backend_id: str) -> VideoBackend:
    """Load a SPECIFIC video backend by registry id (not the active one).

    Mirrors image_backend.get_named_backend. Lets a mode PIN the model it was tuned
    against — lessons are animated on Wan 2.2 — without touching the global active
    backend that manual/facts may switch to something else. Raises if the id is unknown."""
    cfg = model_registry.get_available("video_backend", backend_id)
    if not cfg:
        raise ValueError(f"Video backend '{backend_id}' not found in registry")
    return build_backend(cfg)


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
