"""
Character LoRA store — the asset registry.

A trained character LoRA is a reusable asset (like a published Maya char rig).
This module owns:
  - where trained LoRAs live:        03_Models/loras/characters/
  - the manifest (trigger word, base, version, source):  .../manifest.json
  - the copy into ComfyUI for inference: 01_ComfyUI/.../models/loras/
  - dataset folders for manual drop:  07_Training/datasets/<char>/
  - resolving which LoRAs a prompt needs (by character name match)

Manifest schema (manifest.json):
{
  "characters": {
    "<name_lower>": {
        "name": "Lily",
        "trigger": "rexj_lily",
        "lora_file": "rexj_lily.safetensors",
        "weight": 0.9,
        "base_model": "black-forest-labs/FLUX.1-dev",
        "rank": 16,
        "steps": 1500,
        "version": 1,
        "source_script": "20260618_231516",
        "trained_at": "2026-06-20T...Z"
    }
  }
}
"""

import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).parent.parent.parent.resolve()  # 02_Agent
import sys
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from modules.file_utils import atomic_write_json

log = logging.getLogger("claw_bot.lora.store")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()  # E:\Rexjaw_VFX
STORE_DIR = PROJECT_ROOT / "03_Models" / "loras" / "characters"
MANIFEST = STORE_DIR / "manifest.json"
DATASETS_DIR = PROJECT_ROOT / "07_Training" / "datasets"
COMFY_LORA_DIR = (
    PROJECT_ROOT / "01_ComfyUI" / "ComfyUI_windows_portable"
    / "ComfyUI" / "models" / "loras"
)


# ---------------------------------------------------------------------------
# naming
# ---------------------------------------------------------------------------

def slug(name: str) -> str:
    """Lowercase alnum slug from a character name. 'Lily Anne' -> 'lily_anne'."""
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    return s or "char"


def trigger_for(name: str) -> str:
    """Unique trigger token baked into training captions + inference prompts."""
    return f"rexj_{slug(name)}"


# ---------------------------------------------------------------------------
# dirs
# ---------------------------------------------------------------------------

def dataset_dir(name: str) -> Path:
    d = DATASETS_DIR / slug(name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_dirs():
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# manifest CRUD
# ---------------------------------------------------------------------------

def _load() -> dict:
    if not MANIFEST.exists():
        return {"characters": {}}
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"manifest unreadable ({e}); starting fresh")
        return {"characters": {}}


def all_characters() -> dict:
    return _load().get("characters", {})


def get(name: str) -> Optional[dict]:
    return all_characters().get(slug(name))


def register(
    name: str,
    lora_src: Path,
    *,
    base_model: str = "black-forest-labs/FLUX.1-dev",
    rank: int = 16,
    steps: int = 1500,
    weight: float = 0.9,
    source_script: str = "",
) -> dict:
    """Copy the trained .safetensors into the store + ComfyUI, write manifest entry.

    Returns the manifest entry. Bumps `version` if the character already exists.
    """
    ensure_dirs()
    lora_src = Path(lora_src)
    if not lora_src.exists():
        raise FileNotFoundError(f"LoRA file not found: {lora_src}")

    s = slug(name)
    trig = trigger_for(name)
    lora_file = f"{trig}.safetensors"

    # 1. copy into the asset store (versioned source of truth)
    store_path = STORE_DIR / lora_file
    shutil.copy2(lora_src, store_path)

    # 2. copy into ComfyUI so inference can load it by filename
    COMFY_LORA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lora_src, COMFY_LORA_DIR / lora_file)

    data = _load()
    chars = data.setdefault("characters", {})
    prev = chars.get(s, {})
    entry = {
        "name": name,
        "trigger": trig,
        "lora_file": lora_file,
        "weight": float(weight),
        "base_model": base_model,
        "rank": int(rank),
        "steps": int(steps),
        "version": int(prev.get("version", 0)) + 1,
        "source_script": source_script,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    chars[s] = entry
    atomic_write_json(MANIFEST, data)
    log.info(f"registered LoRA '{name}' -> {lora_file} (v{entry['version']})")
    return entry


def remove(name: str) -> bool:
    data = _load()
    chars = data.get("characters", {})
    s = slug(name)
    if s not in chars:
        return False
    lf = chars[s].get("lora_file")
    chars.pop(s)
    atomic_write_json(MANIFEST, data)
    for p in (STORE_DIR / lf, COMFY_LORA_DIR / lf) if lf else ():
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    return True


# ---------------------------------------------------------------------------
# inference resolution
# ---------------------------------------------------------------------------

def resolve_for_prompt(prompt_text: str) -> list[dict]:
    """Which registered characters appear in this prompt? Returns their LoRA entries.

    Word-boundary, case-insensitive name match against the rendered image prompt
    (same signal storyboard_generator uses to decide character vs empty frame).
    """
    p = (prompt_text or "").lower()
    if not p.strip():
        return []
    hits = []
    for s, e in all_characters().items():
        nm = (e.get("name") or "").strip().lower()
        if nm and re.search(rf"\b{re.escape(nm)}\b", p):
            hits.append(e)
    return hits


def comfy_lora_path(lora_file: str) -> Path:
    return COMFY_LORA_DIR / lora_file
