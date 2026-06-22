"""
Claw Bot — Auto-train character LoRAs for a story's bible.

For each recurring character that has no registered LoRA yet:
  locked_token -> a few photoreal hero portraits (active image backend)
              -> qwen_multiangle multi-angle dataset
              -> ai-toolkit LoRA train -> register in the LoRA store.

Everything is GATED: if ai-toolkit / flux1-dev aren't installed, training raises
a clear error which the caller catches and continues (degraded: no identity lock).
Once a character is registered, the flux_lora image backend auto-stacks its LoRA
whenever the character's NAME appears in a render prompt.
"""

import logging
import sys
from pathlib import Path
from typing import Callable, Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import image_backend
from modules.lora import store as lora_store
from modules.lora import qwen_multiangle, trainer
from modules.script_generator import get_style_description

log = logging.getLogger("claw_bot.lora_autotrain")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
HERO_DIR = PROJECT_ROOT / "07_Training" / "heroes"
N_HEROES = 3
# Flux LoRA train steps. ~800 = solid identity, faster than the 1500 default
# (flux's slow low-VRAM load dominates either way). Bump for max fidelity.
LORA_STEPS = 800
PHOTO_SUFFIX = (get_style_description("photoreal") or {}).get("prompt_suffix", "photorealistic")


def _gen_heroes(name: str, locked_token: str) -> list[Path]:
    """Generate a few photoreal hero portraits of the character (active backend)."""
    be = image_backend.get_active_backend()
    out = HERO_DIR / lora_store.slug(name)
    out.mkdir(parents=True, exist_ok=True)
    prompt = (f"{locked_token}, photorealistic studio portrait, head and shoulders, "
              f"front view, neutral grey background, soft even lighting, sharp focus, "
              f"{PHOTO_SUFFIX}")
    heroes = []
    import random
    for i in range(N_HEROES):
        p = out / f"hero_{i:02d}.png"
        try:
            res = be.generate(prompt=prompt, aspect_ratio="1:1",
                              seed=random.randint(1, 2**31 - 1),
                              output_filename=str(p))
            heroes.append(Path(res))
        except Exception as e:
            log.warning(f"hero gen {i} for '{name}' failed: {e}")
    return heroes


def ensure_character_loras(
    story: dict,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Train+register a LoRA for each bible character missing one.
    Returns {name: 'ready'|'skipped'|'failed'}. Never raises — degrades to
    no-LoRA when training infra is absent."""
    def _p(m):
        log.info(m)
        if progress_cb:
            try:
                progress_cb(m)
            except Exception:
                pass

    result = {}
    horror_id = story.get("horror_id") or story.get("_id") or ""
    for ch in story.get("characters", []):
        name = (ch.get("name") or "").strip()
        if not name:
            continue
        if lora_store.get(name):
            result[name] = "ready"
            _p(f"🧬 LoRA already exists for {name}")
            continue
        try:
            _p(f"🧬 training LoRA for {name} — generating hero portraits...")
            heroes = _gen_heroes(name, ch.get("locked_token") or ch.get("appearance", ""))
            if not heroes:
                raise RuntimeError("no hero portraits produced")
            _p(f"🧬 {name}: building multi-angle dataset...")
            n = qwen_multiangle.build_dataset(name, heroes)
            _p(f"🧬 {name}: {n} dataset frames — training (this is slow)...")
            trainer.train_character(
                name, ch.get("locked_token") or ch.get("appearance", ""),
                steps=LORA_STEPS, source_script=horror_id,
                progress_callback=lambda lbl, cur, tot: _p(f"🧬 {lbl} {cur}/{tot}"),
            )
            result[name] = "ready"
            _p(f"🧬 LoRA ready for {name}")
        except Exception as e:
            result[name] = "failed"
            log.warning(f"LoRA train failed for '{name}' ({e}); rendering without it.")
            _p(f"⚠️ LoRA unavailable for {name} ({type(e).__name__}) — using prompt tokens.")
    return result
