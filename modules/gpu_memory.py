"""
Claw Bot — GPU residency manager

One 16 GB card, several ~13 GB models. Any two of them collide:

    Wan 2.2 14B (video)        ~13.6 GB
    Qwen-Image-Edit (thumbs)   ~13.5 GB
    Flux.1-dev / USO           ~12.0 GB
    Ollama qwen2.5:14b (LLM)   ~12.6 GB
    Z-Image Turbo              ~6.0 GB

Before this module, ten different files called `ensure_vram_free`,
`free_comfyui_vram` and `free_ollama_vram` in whatever order seemed right
locally. Nothing tracked WHAT was resident, and that single missing fact caused
three separate production failures in one day:

  1. Ollama's 12.6 GB stayed loaded while Wan rendered. A 90-second clip took
     16 minutes, thrashing weights through PCIe.
  2. A "warm" 14-video batch called ensure_vram_free(14 GB) before each video.
     It looked at a card holding a resident 13.5 GB Qwen, saw ~1 GB free, and
     freed ComfyUI -- evicting the very model it was about to use. Fourteen
     cold loads, ~4 minutes each.
  3. Qwen ran with 211 MB free. ComfyUI's offload streams had nowhere to go and
     the CUDA context died mid-batch (illegal memory access). Twelve videos then
     silently fell back to still-frame thumbnails.

The fix is one idea: **know which model is resident**. Evicting is then only
necessary when a DIFFERENT model wants the card.

    with gpu_memory.resident(QWEN_EDIT):     # evicts others, or no-ops
        render(); render(); render()         # warm
    # released on exit

Callers holding the job lock own the card, so the tracked state is authoritative
for the duration. Outside the lock it is a best-effort hint.
"""

import logging
import threading
from contextlib import contextmanager
from typing import Optional

log = logging.getLogger("claw_bot.gpu_memory")

# --- known consumers -------------------------------------------------------
OLLAMA = "ollama"
WAN_VIDEO = "wan22_14b"
QWEN_EDIT = "qwen_image_edit"
FLUX_USO = "flux_uso"
ZIMAGE = "zimage"
SDXL = "sdxl"

MODEL_VRAM_GB = {
    OLLAMA: 12.6,
    WAN_VIDEO: 13.6,
    QWEN_EDIT: 13.5,
    FLUX_USO: 12.0,
    ZIMAGE: 6.0,
    SDXL: 7.0,
}

# ComfyUI needs working room on top of the weights: offload streams, the VAE
# decode spike, and the text encoder. Starving it is what killed the CUDA
# context — the fault fired with 211 MB free.
#
# ComfyUI reserves this itself when launched with `--reserve-vram 1.5` (see
# 00_Tools/launch_clawbot.ps1), so we do NOT add it to what a model "needs":
# a 13.5 GB model on a card with 14.4 GB free is a normal, working situation,
# and warning about it every single time would just teach you to ignore the log.
WORKING_HEADROOM_GB = 1.5

# Warn only when the load is genuinely marginal — the shape of the CUDA fault.
MIN_FREE_AFTER_LOAD_GB = 0.5

_lock = threading.Lock()
_resident: Optional[str] = None


# ==============================================================================
# STATE
# ==============================================================================

def resident_model() -> Optional[str]:
    """What we believe ComfyUI/Ollama currently holds, or None."""
    with _lock:
        return _resident


def _set_resident(label: Optional[str]) -> None:
    global _resident
    with _lock:
        _resident = label


def free_gb() -> float:
    from modules import gpu_utils
    stats = gpu_utils.get_vram_stats() or {}
    return float(stats.get("free_gb") or 0.0)


def needs_gb(label: str) -> float:
    """Weights only. ComfyUI reserves its own working room (--reserve-vram)."""
    return MODEL_VRAM_GB.get(label, 8.0)


def forget() -> None:
    """Drop the tracked state without freeing anything.

    Use when something outside our control took the card (a crash, another
    process, a CUDA fault). A dead context is not a resident model.
    """
    _set_resident(None)


# ==============================================================================
# EVICTION
# ==============================================================================

def unload_llm() -> None:
    """Drop Ollama's model. Call once the text is written, before any render."""
    from modules import gpu_utils
    try:
        gpu_utils.free_ollama_vram()
    except Exception as e:
        log.warning(f"could not unload Ollama: {e}")
    if resident_model() == OLLAMA:
        _set_resident(None)


def unload_image_models() -> None:
    """Ask ComfyUI to drop whatever it holds."""
    from modules import gpu_utils
    try:
        gpu_utils.free_comfyui_vram()
    except Exception as e:
        log.warning(f"could not free ComfyUI: {e}")
    if resident_model() not in (None, OLLAMA):
        _set_resident(None)


def evict_all() -> None:
    unload_llm()
    unload_image_models()


# ==============================================================================
# ACQUIRE
# ==============================================================================

def acquire(label: str) -> None:
    """Make room for `label` and mark it resident.

    A no-op when `label` is already the resident model — that is the whole
    point: a warm batch must not evict itself between renders.
    """
    if resident_model() == label:
        log.debug(f"{label} already resident; no eviction")
        return

    want = needs_gb(label)
    log.info(f"GPU: making room for {label} (weights ~{want:.1f} GB, "
             f"{free_gb():.1f} GB free, resident={resident_model()})")
    evict_all()

    have = free_gb()
    if have < want + MIN_FREE_AFTER_LOAD_GB:
        # Not fatal — ComfyUI offloads — but this is the exact shape of the
        # failure that killed the CUDA context, so say so.
        log.warning(f"GPU: only {have:.1f} GB free for {label} "
                    f"({want:.1f} GB of weights). Expect offloading; a CUDA "
                    f"illegal-access is possible if ComfyUI has no reserve.")
    _set_resident(label)


def release(label: Optional[str] = None) -> None:
    """Hand the card back so the next stage starts clean.

    Skips the free() when someone else already took over — releasing a model you
    no longer hold would evict theirs.
    """
    current = resident_model()
    if label is not None and current is not None and current != label:
        log.debug(f"release({label}) ignored; {current} holds the card")
        return
    unload_image_models()
    _set_resident(None)


@contextmanager
def resident(label: str):
    """Hold the card for `label` and release it afterwards, even on error.

        with gpu_memory.resident(gpu_memory.QWEN_EDIT):
            for aspect in aspects:
                render(aspect)          # warm: no eviction between renders
    """
    acquire(label)
    try:
        yield
    finally:
        release(label)


@contextmanager
def llm():
    """Hold the card for the LLM, then drop it before any renderer wants it.

    The pattern that matters for batches: write ALL the text first, unload once,
    then render. Alternating per item costs two model loads per item.
    """
    acquire(OLLAMA)
    try:
        yield
    finally:
        unload_llm()
        # Explicit, not relying on unload_llm's side effect: the card is free
        # whether or not the unload call could reach Ollama.
        if resident_model() == OLLAMA:
            _set_resident(None)


def summary() -> str:
    return (f"resident={resident_model() or 'none'}, "
            f"{free_gb():.1f} GB free of {MODEL_VRAM_GB.get(WAN_VIDEO, 0) + 2.4:.0f}+ GB")
