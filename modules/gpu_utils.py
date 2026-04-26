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


def free_ollama_vram(model_name: Optional[str] = None) -> tuple[bool, str]:
    """Unload Llama from VRAM by calling `ollama stop <model>`."""
    if model_name is None:
        try:
            llm_cfg = model_registry.get_active("llm_backend")
            model_name = (
                llm_cfg.get("model_name")
                or llm_cfg.get("model")
                or "llama3.1:8b-instruct-q4_K_M"
            )
        except Exception:
            model_name = "llama3.1:8b-instruct-q4_K_M"

    try:
        result = subprocess.run(
            ["ollama", "stop", model_name],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            log.info(f"Ollama unloaded: {model_name}")
            return True, f"Ollama model '{model_name}' unloaded"
        if "could not locate ollama app" in (result.stderr or ""):
            return True, "Ollama not running (nothing to unload)"
        return False, f"ollama stop failed: {result.stderr.strip()}"
    except FileNotFoundError:
        return False, "ollama command not found"
    except Exception as e:
        return False, f"Failed to free Ollama VRAM: {e}"


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