"""
Claw Bot — GPU utilities

Clean up VRAM between jobs:
- ComfyUI: call /free endpoint to unload models
- Ollama: stop the active model (unloads from VRAM)
- VRAM usage query via nvidia-smi
"""

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import model_registry

log = logging.getLogger("claw_bot.gpu_utils")


def get_vram_stats() -> dict:
    """Return GPU memory stats via nvidia-smi (MB)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return {"error": result.stderr.strip()}
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        used, free, total, util = [int(p) for p in parts]
        return {
            "used_mb": used,
            "free_mb": free,
            "total_mb": total,
            "util_percent": util,
            "free_gb": round(free / 1024, 2),
            "total_gb": round(total / 1024, 2),
        }
    except FileNotFoundError:
        return {"error": "nvidia-smi not found"}
    except Exception as e:
        return {"error": str(e)}


def free_comfyui_vram() -> tuple[bool, str]:
    """Tell ComfyUI to unload all models from VRAM."""
    try:
        cfg = model_registry.get_active("image_backend")
        server_url = cfg.get("server_url", "http://127.0.0.1:8188").rstrip("/")
        r = requests.post(
            f"{server_url}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=10,
        )
        if r.status_code == 200:
            log.info("ComfyUI VRAM freed.")
            return True, "ComfyUI models unloaded"
        return False, f"ComfyUI /free returned HTTP {r.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "ComfyUI not reachable (probably offline)"
    except Exception as e:
        return False, f"Failed to free ComfyUI VRAM: {e}"


def soft_free_comfyui_cache() -> tuple[bool, str]:
    """Drop ComfyUI's cached/inactive VRAM but KEEP the models loaded.

    Between back-to-back renders on the same model, ComfyUI's offload machinery
    fragments VRAM until its weight page-fault handler cannot bring weights back
    — `comfy_aimdo/model_vbar.py` raises "Fault failed: 2" and the job dies. It
    took 3-4 consecutive Wan-S2V clips to trigger. A full /free would fix it too
    but would evict the model and pay a ~4 min cold reload every clip; this only
    releases the cache.
    """
    try:
        cfg = model_registry.get_active("image_backend")
        server_url = cfg.get("server_url", "http://127.0.0.1:8188").rstrip("/")
        r = requests.post(
            f"{server_url}/free",
            json={"unload_models": False, "free_memory": True},
            timeout=10,
        )
        if r.status_code == 200:
            return True, "ComfyUI cache freed (models kept)"
        return False, f"ComfyUI /free returned HTTP {r.status_code}"
    except Exception as e:
        return False, f"soft free failed: {e}"


def _ollama_server_url() -> str:
    """Resolve the Ollama server URL from the LLM backend config (fallback localhost)."""
    server_url = "http://127.0.0.1:11434"
    try:
        llm_cfg = model_registry.get_active("llm_backend")
        if llm_cfg:
            server_url = (llm_cfg.get("server_url") or server_url).rstrip("/")
    except Exception:
        pass
    return server_url.rstrip("/")


def _ollama_resident_models(server_url: str) -> Optional[list[str]]:
    """Names of every model currently loaded in VRAM via /api/ps.

    Returns None if the server is unreachable (caller treats as 'nothing to do').
    """
    try:
        r = requests.get(f"{server_url}/api/ps", timeout=5)
        if r.status_code != 200:
            return []
        models = r.json().get("models", []) or []
        return [m.get("name") or m.get("model") for m in models if (m.get("name") or m.get("model"))]
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        log.warning(f"Ollama /api/ps failed ({e}); will fall back to configured model.")
        return []


def free_ollama_vram(model_name: Optional[str] = None) -> tuple[bool, str]:
    """Unload ALL resident LLMs from VRAM via the Ollama HTTP API.

    Queries `/api/ps` for every model currently loaded and evicts each with
    `keep_alive: 0`. Unloading only the *configured active* id (the old
    behaviour) leaves any second model — e.g. a creative + a structurer in a
    role-routed setup — hogging VRAM, starving the 12GB video model into
    RAM-offload (the 4min→15min slowdown). HTTP API only (no `ollama` CLI/PATH
    dependency); keep_alive=0 forces an immediate unload.
    """
    server_url = _ollama_server_url()

    resident = _ollama_resident_models(server_url)
    if resident is None:
        return True, "Ollama not running (nothing to unload)"

    targets = list(resident)
    if not targets:
        # /api/ps empty (nothing loaded) or unavailable — fall back to the
        # explicit / configured model name so a load that hasn't registered yet
        # still gets a keep_alive=0 eviction request.
        if model_name:
            targets = [model_name]
        else:
            try:
                llm_cfg = model_registry.get_active("llm_backend")
                name = (
                    llm_cfg.get("model_id")
                    or llm_cfg.get("model_name")
                    or llm_cfg.get("model")
                )
                if name:
                    targets = [name]
            except Exception:
                pass
        if not targets:
            return True, "No Ollama model resident (nothing to unload)"

    unloaded, failed = [], []
    for name in targets:
        try:
            r = requests.post(
                f"{server_url}/api/generate",
                json={"model": name, "keep_alive": 0},
                timeout=30,
            )
            if r.status_code == 200:
                unloaded.append(name)
                log.info(f"Ollama unloaded via API: {name}")
            else:
                failed.append(f"{name} (HTTP {r.status_code})")
                log.warning(f"Ollama unload {name} returned HTTP {r.status_code}")
        except Exception as e:
            failed.append(f"{name} ({type(e).__name__})")
            log.warning(f"Ollama unload {name} failed: {e}")

    if unloaded and not failed:
        return True, f"Ollama unloaded: {', '.join(unloaded)}"
    if unloaded:
        return True, f"Ollama unloaded: {', '.join(unloaded)}; failed: {', '.join(failed)}"
    return False, f"Ollama unload failed: {', '.join(failed)}"


def cleanup_after_job(free_comfy: bool = True, free_ollama: bool = False) -> dict:
    """
    Call after a major task completes.

    Args:
        free_comfy: unload ComfyUI models (images)
        free_ollama: unload Llama. Default False because the bot reuses it often.
    """
    before = get_vram_stats()
    actions = []
    if free_comfy:
        ok, msg = free_comfyui_vram()
        actions.append(("comfyui", ok, msg))
    if free_ollama:
        ok, msg = free_ollama_vram()
        actions.append(("ollama", ok, msg))
    time.sleep(1.5)
    after = get_vram_stats()
    return {"before": before, "after": after, "actions": actions}


def ensure_vram_free(min_gb: float = 5.0, force_ollama_unload: bool = True) -> dict:
    """
    Pre-flight before a heavy GPU job. If less than `min_gb` is free, force
    Ollama unload + ComfyUI /free, then re-check. Always returns the result
    of the final check (caller may decide to abort if still insufficient).
    """
    stats = get_vram_stats()
    free_gb = stats.get("free_gb")
    if free_gb is None or free_gb >= min_gb:
        return {"action": "noop", "stats": stats}

    log.warning(
        f"VRAM pre-flight: only {free_gb} GB free (need {min_gb}). Freeing models..."
    )
    if force_ollama_unload:
        free_ollama_vram()
    free_comfyui_vram()
    time.sleep(2.0)
    after = get_vram_stats()
    log.info(
        f"VRAM pre-flight done: {after.get('free_gb')} GB free "
        f"(was {free_gb} GB before, target {min_gb} GB)"
    )
    return {"action": "freed", "before": stats, "stats": after}


def check_comfyui_alive(server_url: Optional[str] = None) -> tuple[bool, str]:
    """Cheap reachability ping. Used by pre-pipeline health gate."""
    if server_url is None:
        try:
            cfg = model_registry.get_active("image_backend")
            server_url = cfg.get("server_url", "http://127.0.0.1:8188")
        except Exception:
            server_url = "http://127.0.0.1:8188"
    server_url = server_url.rstrip("/")
    try:
        r = requests.get(f"{server_url}/system_stats", timeout=3)
        if r.status_code == 200:
            return True, "online"
        return False, f"HTTP {r.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "offline"
    except Exception as e:
        return False, f"error: {type(e).__name__}"


def check_ollama_alive(server_url: str = "http://127.0.0.1:11434") -> tuple[bool, str]:
    try:
        r = requests.get(f"{server_url.rstrip('/')}/api/tags", timeout=3)
        if r.status_code == 200:
            return True, "online"
        return False, f"HTTP {r.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "offline"
    except Exception as e:
        return False, f"error: {type(e).__name__}"