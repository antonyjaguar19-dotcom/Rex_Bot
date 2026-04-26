"""
Claw Bot — Storyboard Generator (Session 1A)

Reads the 'style' field from a script and applies the corresponding
style prompt suffix. Supports any shot count (LLM decides).
Single-frame-per-shot mode (first_frame only — video gen later).
"""

# -----------------------------------------------------------------------------
# TODO (Phase 7 Polish): Character Consistency
# Currently each shot is generated independently -> characters drift across shots.
# FIX 1: Reference Image Cascade (ref shot1 as img2img anchor for shots 2+)
# FIX 2: Character LoRA training (per recurring character: Arjun, Riya, etc.)
# Decision made 2026-04-19: ship full pipeline first, polish in Phase 7.
# -----------------------------------------------------------------------------

import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import image_backend as ib
from modules import model_registry
from modules import script_generator as sg

log = logging.getLogger("claw_bot.storyboard_generator")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SCRIPTS_DIR = PROJECT_ROOT / "04_Outputs" / "scripts"
STORYBOARDS_DIR = PROJECT_ROOT / "04_Outputs" / "storyboards"


@dataclass
class FrameResult:
    shot_number: int
    frame_type: str
    prompt: str
    formatted_prompt: str
    image_path: str
    seed: int
    generated_at: str
    backend_id: str


@dataclass
class StoryboardResult:
    script_id: str
    backend_id: str
    aspect_ratio: str
    total_frames: int
    frames: list[FrameResult]
    generated_at: str
    success: bool
    error: Optional[str] = None
    style: Optional[str] = None


# ==============================================================================
# PROMPT FORMATTING — uses the script's chosen style
# ==============================================================================

def _characters_description(script: dict) -> str:
    chars = script.get("characters", [])
    parts = []
    for ch in chars:
        if isinstance(ch, dict):
            parts.append(f"{ch.get('name', 'A character')} — {ch.get('appearance', '')}")
        elif isinstance(ch, str):
            parts.append(ch)
    return " ".join(parts) if parts else ""


def _style_suffix(script: dict) -> str:
    """Look up the style the LLM picked for this script."""
    style_id = script.get("style") or sg.get_default_style()
    style_info = sg.get_style_description(style_id)
    return style_info.get("prompt_suffix", "")


def _rewrite_as_paragraph(raw_prompt: str, script: dict, shot: dict) -> str:
    """
    Build the final prompt sent to the image backend.
    Structure: [cinematic opener] + [character grounding] + [shot content] + [style suffix]
    """
    cleaned = raw_prompt.strip()
    # Strip any accidental style tags from shot prompts (LLM might include them)
    for tag in [
        "storybook illustration", "Pixar-inspired", "pastel colors",
        "cartoon style", "anime style", "watercolor", "pixel art",
    ]:
        if tag.lower() in cleaned.lower():
            # Very light cleanup: remove trailing "rendered as X" etc.
            cleaned = cleaned.rstrip(" .,") 

    char_para = _characters_description(script)
    setting = script.get("setting", "")

    parts = [
        f"A cinematic illustration with a gentle composition. "
        f"The scene takes place in {setting.lower() if setting else 'an evocative setting'}.",
    ]
    if char_para:
        parts.append(f"Characters in the scene: {char_para}.")
    parts.append(cleaned.rstrip(".") + ".")
    style_suffix = _style_suffix(script)
    if style_suffix:
        parts.append(f"Visually: {style_suffix}.")

    return " ".join(parts)


# ==============================================================================
# FILE I/O
# ==============================================================================

def _load_script(script_id: str) -> dict:
    path = SCRIPTS_DIR / f"script_{script_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_path(script_id: str) -> Path:
    return STORYBOARDS_DIR / script_id / "storyboard.json"


def _save_manifest(result: StoryboardResult):
    path = _manifest_path(result.script_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(result)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Manifest saved: {path}")


# ==============================================================================
# MAIN API
# ==============================================================================

def generate_storyboard(
    script_id: str,
    aspect_ratio: str = "16:9",
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    skip_existing: bool = False,
) -> StoryboardResult:
    """Generate one image per shot (LLM-determined shot count)."""
    script = _load_script(script_id)
    shots = script.get("shots", [])
    if len(shots) == 0:
        raise ValueError(f"Script {script_id} has no shots")

    backend = ib.get_active_backend()
    backend_id = backend.backend_id
    style_used = script.get("style", "storybook")
    log.info(
        f"Generating storyboard for script {script_id} "
        f"({len(shots)} shots, style={style_used}) using backend {backend_id}"
    )

    story_dir = STORYBOARDS_DIR / script_id
    story_dir.mkdir(parents=True, exist_ok=True)

    frames: list[FrameResult] = []
    total = len(shots)
    current = 0

    try:
        for shot in shots:
            shot_num = shot.get("shot_number", current + 1)
            for frame_type in ("first",):
                current += 1
                prompt_key = f"{frame_type}_frame_prompt"
                raw_prompt = shot.get(prompt_key) or shot.get("visual_description", "")
                if not raw_prompt:
                    raise ValueError(
                        f"Shot {shot_num} missing both '{prompt_key}' and 'visual_description'"
                    )

                image_rel_path = f"04_Outputs/storyboards/{script_id}/shot{shot_num}_{frame_type}.png"
                image_abs_path = PROJECT_ROOT / image_rel_path

                if skip_existing and image_abs_path.exists():
                    log.info(f"[{current}/{total}] Skipping existing: shot{shot_num}_{frame_type}")
                    if progress_callback:
                        progress_callback(f"Skipped shot {shot_num} (already exists)", current, total)
                    frames.append(FrameResult(
                        shot_number=shot_num, frame_type=frame_type,
                        prompt=raw_prompt, formatted_prompt="(existing)",
                        image_path=image_rel_path, seed=-1,
                        generated_at=datetime.now(timezone.utc).isoformat(),
                        backend_id=backend_id,
                    ))
                    continue

                if progress_callback:
                    progress_callback(
                        f"Generating shot {shot_num} of {total}...", current, total
                    )

                formatted = _rewrite_as_paragraph(raw_prompt, script, shot)
                log.info(f"[{current}/{total}] Rendering shot{shot_num}_{frame_type}")

                saved_path = backend.generate(
                    prompt=formatted,
                    aspect_ratio=aspect_ratio,
                    output_filename=f"{script_id}/shot{shot_num}_{frame_type}.png",
                )
                frames.append(FrameResult(
                    shot_number=shot_num,
                    frame_type=frame_type,
                    prompt=raw_prompt,
                    formatted_prompt=formatted,
                    image_path=str(saved_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    seed=-1,
                    generated_at=datetime.now(timezone.utc).isoformat(),
                    backend_id=backend_id,
                ))
    except Exception as e:
        log.exception(f"Storyboard generation failed at frame {current}/{total}")
        result = StoryboardResult(
            script_id=script_id, backend_id=backend_id, aspect_ratio=aspect_ratio,
            total_frames=total, frames=frames,
            generated_at=datetime.now(timezone.utc).isoformat(),
            success=False, error=str(e), style=style_used,
        )
        _save_manifest(result)
        raise

    result = StoryboardResult(
        script_id=script_id, backend_id=backend_id, aspect_ratio=aspect_ratio,
        total_frames=total, frames=frames,
        generated_at=datetime.now(timezone.utc).isoformat(),
        success=True, style=style_used,
    )
    _save_manifest(result)
    log.info(f"Storyboard complete: {len(frames)}/{total} frames saved under {story_dir}")
    return result


def regenerate_shot(
    script_id: str,
    shot_number: int,
    which_frame: str = "first",
    aspect_ratio: str = "16:9",
) -> list[FrameResult]:
    """Re-render a single shot."""
    script = _load_script(script_id)
    shot = next((s for s in script["shots"] if s.get("shot_number") == shot_number), None)
    if shot is None:
        raise ValueError(f"Shot {shot_number} not found in script {script_id}")

    frames_to_redo = ["first", "last"] if which_frame == "both" else [which_frame]

    backend = ib.get_active_backend()
    story_dir = STORYBOARDS_DIR / script_id
    story_dir.mkdir(parents=True, exist_ok=True)

    results: list[FrameResult] = []
    for frame_type in frames_to_redo:
        raw_prompt = shot.get(f"{frame_type}_frame_prompt") or shot.get("visual_description", "")
        formatted = _rewrite_as_paragraph(raw_prompt, script, shot)
        log.info(f"Re-rendering shot{shot_number}_{frame_type}")
        saved_path = backend.generate(
            prompt=formatted,
            aspect_ratio=aspect_ratio,
            output_filename=f"{script_id}/shot{shot_number}_{frame_type}.png",
        )
        results.append(FrameResult(
            shot_number=shot_number, frame_type=frame_type,
            prompt=raw_prompt, formatted_prompt=formatted,
            image_path=str(saved_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            seed=-1,
            generated_at=datetime.now(timezone.utc).isoformat(),
            backend_id=backend.backend_id,
        ))

    _update_manifest_with_reruns(script_id, results)
    return results


def get_storyboard_status(script_id: str) -> Optional[StoryboardResult]:
    path = _manifest_path(script_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = [FrameResult(**f) for f in data.get("frames", [])]
    return StoryboardResult(
        script_id=data["script_id"], backend_id=data["backend_id"],
        aspect_ratio=data["aspect_ratio"], total_frames=data["total_frames"],
        frames=frames, generated_at=data["generated_at"],
        success=data.get("success", False), error=data.get("error"),
        style=data.get("style"),
    )


def _update_manifest_with_reruns(script_id: str, new_frames: list[FrameResult]):
    existing = get_storyboard_status(script_id)
    if existing is None:
        return
    new_keys = {(f.shot_number, f.frame_type) for f in new_frames}
    updated_frames = []
    for f in existing.frames:
        if (f.shot_number, f.frame_type) in new_keys:
            replacement = next(
                n for n in new_frames
                if n.shot_number == f.shot_number and n.frame_type == f.frame_type
            )
            updated_frames.append(replacement)
        else:
            updated_frames.append(f)
    existing.frames = updated_frames
    existing.generated_at = datetime.now(timezone.utc).isoformat()
    _save_manifest(existing)