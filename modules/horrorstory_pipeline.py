"""
Claw Bot — Horror Story Pipeline (orchestrator)

Renders an approved horror story:
  1. VoxCPM Voice Design — one deep ominous narrator voices every beat; the per-
     beat audio spans give exact scene timing (narration = source of truth).
  2. Photorealistic still per beat (image_backend, no character lock).
  3. horror_assembly — Ken Burns + narration + ambient drone → 16x9 + watermark.

Front-end agnostic (returns paths); claw_bot / dashboard post the result.
"""

import logging
import random
import sys
from pathlib import Path
from typing import Callable, Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import image_backend
from modules import gpu_utils
from modules import runtime_settings as rs
from modules import horror_assembly as hasm
from modules.script_generator import get_style_description

log = logging.getLogger("claw_bot.horrorstory_pipeline")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
STILLS_DIR = PROJECT_ROOT / "04_Outputs" / "storyboards"
NARRATION_DIR = PROJECT_ROOT / "04_Outputs" / "audio"

PHOTO_STYLE_ID = "photoreal"


def _scene_prompt(beat: dict) -> str:
    suffix = (get_style_description(PHOTO_STYLE_ID) or {}).get("prompt_suffix", "")
    parts = [beat.get("image_prompt", "").strip()]
    if suffix:
        parts.append(suffix)
    return ", ".join(p for p in parts if p)


def _scene_durations(spans: list, total_dur: float) -> list[float]:
    """Tile the whole narration timeline across beats: beat i covers from its
    own start to the next beat's start (last beat runs to the end). spans items
    are (text, t_start, t_end)."""
    starts = [s[1] for s in spans]
    durs = []
    for i in range(len(starts)):
        nxt = starts[i + 1] if i + 1 < len(starts) else total_dur
        durs.append(max(0.5, round(nxt - starts[i], 3)))
    return durs


def render_horror(
    story: dict,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Full render: narration (VoxCPM) -> photoreal stills -> 16x9 assembly."""
    def _p(msg: str):
        log.info(msg)
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    horror_id = story.get("horror_id") or story.get("_id")
    beats = story.get("beats", [])
    if not beats:
        raise ValueError("Horror story has no beats.")
    voice_design = story.get("voice_design", "")

    # ---- 1. Narration (VoxCPM Voice Design — one deep narrator) ----
    _p(f"🎙️ voicing {len(beats)} beats (deep narrator)...")
    gpu_utils.ensure_vram_free()
    from modules.tts_voxcpm import VoxCPMTTS
    tts = VoxCPMTTS()
    groups = [(b["narration"], voice_design) for b in beats]
    NARRATION_DIR.mkdir(parents=True, exist_ok=True)
    narration_path = NARRATION_DIR / f"horror_{horror_id}.wav"
    narration_path, spans = tts.synthesize_designed(groups, output_path=narration_path)
    total_dur = hasm._probe_duration(narration_path)
    _p(f"🎙️ narration ready: {total_dur:.1f}s")

    # Per-beat scene durations from the real audio spans.
    if len(spans) != len(beats):
        log.warning(f"spans({len(spans)}) != beats({len(beats)}); tiling on spans.")
        beats = beats[:len(spans)]
    durations = _scene_durations(spans, total_dur)

    # ---- 2. Photoreal stills (one per beat) ----
    gpu_utils.free_comfyui_vram()
    img = image_backend.get_active_backend()
    out_dir = STILLS_DIR / f"horror_{horror_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_images: list[Path] = []
    for i, b in enumerate(beats):
        _p(f"🖼️ still {i+1}/{len(beats)}...")
        path = img.generate(
            prompt=_scene_prompt(b),
            aspect_ratio="16:9",
            seed=random.randint(1, 2**31 - 1),
            output_filename=str(out_dir / f"beat_{i:04d}.png"),
        )
        scene_images.append(Path(path))

    # ---- 3. Assemble (16x9 + ambient + watermark) ----
    _p("🎬 assembling horror video...")
    gpu_utils.free_comfyui_vram()
    out = hasm.assemble_horror(
        story, narration_path, durations, scene_images,
        ambient=rs.get_horror_ambient_enabled(), progress_cb=progress_cb,
    )
    out["narration_audio"] = narration_path
    _p("✅ horror video complete")
    return out
