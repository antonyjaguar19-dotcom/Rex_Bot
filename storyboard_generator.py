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
from modules import runtime_settings as rs

CAST_SHEET_SEED = 777  # fixed so the cast sheet is reproducible per script

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
    narration: str = ""           # what the narrator says — QA story-coherence check
    beat: str = ""                # story beat type (hook/spark/consequence etc.)
    visual_description: str = ""  # one-sentence scene description


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

def _characters_description(script: dict, raw_prompt: str = "") -> str:
    """Build the locked character description prepend conditionally.
    Only injects characters whose names appear in the raw_prompt.
    """
    chars = script.get("characters", [])
    parts = []
    prompt_lower = raw_prompt.lower()
    for ch in chars:
        if isinstance(ch, dict):
            name = ch.get("name", "").strip()
            
            # Skip this character if their name isn't in the prompt
            if raw_prompt and name and name.lower() not in prompt_lower:
                continue
                
            locked = (ch.get("locked_visual_token") or ch.get("appearance") or "").strip()
            if name and locked:
                parts.append(f"{name} — {locked}")
            elif name:
                parts.append(name)
        elif isinstance(ch, str):
            # Legacy string format
            name = ch.split(",")[0].split("-")[0].strip()
            if raw_prompt and name and name.lower() not in prompt_lower:
                continue
            parts.append(name)
    return "; ".join(parts) if parts else ""


def _style_suffix(script: dict) -> str:
    """Look up the style the LLM picked for this script."""
    style_id = script.get("style") or sg.get_default_style()
    style_info = sg.get_style_description(style_id)
    return style_info.get("prompt_suffix", "")


def _rewrite_as_paragraph(raw_prompt: str, script: dict, shot: dict) -> str:
    """
    Build the final prompt sent to the image backend.
    Structure: [style] + [locked character sheet] + [scene] + [pose/framing] + [setting]

    Character-sheet locking: the character description is identical across
    every shot in the storyboard, so Flux.2's CLIP encoder sees the same
    character tokens on every render → far less drift.
    """
    cleaned = raw_prompt.strip()
    # Strip any accidental style tags from shot prompts (LLM might include them)
    for tag in [
        "storybook illustration", "Pixar-inspired", "pastel colors",
        "cartoon style", "anime style", "watercolor", "pixel art",
    ]:
        if tag.lower() in cleaned.lower():
            cleaned = cleaned.rstrip(" .,")

    # Pass the prompt in so we only fetch characters actually in this shot
    char_para = _characters_description(script, raw_prompt=cleaned)
    setting = script.get("setting", "")
    style_suffix = _style_suffix(script)

    parts = []
    # 1. Style anchor — Flux.2 weights early tokens most. Style first.
    if style_suffix:
        parts.append(f"Art style: {style_suffix}.")

    # 2. LOCKED CHARACTER SHEET — placed second so the character tokens are
    # near the front of the prompt where attention is strongest. Same words
    # every shot = same character every shot.
    if char_para:
        parts.append(f"Character sheet (must match exactly across all shots): {char_para}.")

    # 3. Shot content — pose, framing, action. No character appearance words.
    parts.append(cleaned.rstrip(".") + ".")

    # 4. Setting anchor at the end as soft context
    if setting:
        parts.append(f"Scene takes place in {setting.lower().rstrip('.')}.")

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

def _generate_cast_sheet(script: dict, script_id: str, backend, aspect_ratio: str):
    """Generate ONE reference image of all main characters (front-facing, neutral,
    plain background, fixed seed). Returns its Path, or None on failure / no chars."""
    chars = script.get("characters", [])
    tokens = []
    for c in chars:
        if isinstance(c, dict):
            name = c.get("name", "").strip()
            tok = (c.get("locked_visual_token") or c.get("appearance") or "").strip()
            if tok:
                tokens.append(f"{name}: {tok}" if name else tok)
    if not tokens:
        return None

    style_suffix = _style_suffix(script)
    parts = []
    if style_suffix:
        parts.append(f"Art style: {style_suffix}.")
    parts.append(
        "Character reference sheet. Full body, front facing, neutral standing pose, "
        "arms relaxed at sides, plain solid light-gray studio background, even soft lighting, "
        "no props, no scenery, characters clearly separated side by side. "
        + "Characters: " + "; ".join(tokens) + "."
    )
    cast_prompt = " ".join(parts)
    log.info(f"Generating cast sheet for {script_id} ({len(tokens)} character(s))")
    try:
        ref_path = backend.generate(
            prompt=cast_prompt,
            aspect_ratio=aspect_ratio,
            output_filename=f"{script_id}/_cast_sheet.png",
            seed=CAST_SHEET_SEED,
        )
        log.info(f"Cast sheet saved: {ref_path}")
        return ref_path
    except Exception as e:
        log.warning(f"Cast sheet generation failed (continuing without reference): {e}")
        return None

def _generate_environment_sheet(script: dict, script_id: str, backend, aspect_ratio: str):
    """Generate ONE reference image of the setting."""
    setting = script.get("setting", "").strip()
    if not setting:
        return None

    style_suffix = _style_suffix(script)
    parts = []
    if style_suffix:
        parts.append(f"Art style: {style_suffix}.")
    parts.append(
        f"Environment reference sheet. Wide landscape shot of {setting}. "
        "No characters, empty scenery, beautiful lighting, highly detailed background."
    )
    env_prompt = " ".join(parts)
    log.info(f"Generating environment sheet for {script_id}")
    try:
        ref_path = backend.generate(
            prompt=env_prompt,
            aspect_ratio=aspect_ratio,
            output_filename=f"{script_id}/_env_sheet.png",
            seed=CAST_SHEET_SEED + 1,
        )
        log.info(f"Environment sheet saved: {ref_path}")
        return ref_path
    except Exception as e:
        log.warning(f"Environment sheet generation failed: {e}")
        return None


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

    # Cast sheet & Environment sheet: anchoring characters and backgrounds
    reference_image = None
    environment_image = None
    if rs.get_reference_mode_enabled():
        if progress_callback:
            progress_callback("Generating character cast sheet...", 0, len(shots))
        reference_image = _generate_cast_sheet(script, script_id, backend, aspect_ratio)
        
        if progress_callback:
            progress_callback("Generating environment sheet...", 0, len(shots))
        environment_image = _generate_environment_sheet(script, script_id, backend, aspect_ratio)

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

                # Character consistency: locked tokens in the prompt + a cast-sheet
                # reference latent (when reference mode is on). The reference is passed
                # only if the backend supports it; otherwise we fall back cleanly.
                _gen_kwargs = dict(
                    prompt=formatted,
                    aspect_ratio=aspect_ratio,
                    output_filename=f"{script_id}/shot{shot_num}_{frame_type}.png",
                )
                if reference_image is not None or environment_image is not None:
                    try:
                        saved_path = backend.generate(
                            reference_image=reference_image, 
                            environment_image=environment_image, 
                            **_gen_kwargs
                        )
                    except TypeError:
                        try: # Fallback if backend only supports 1 reference
                            saved_path = backend.generate(reference_image=reference_image, **_gen_kwargs)
                        except TypeError:
                            saved_path = backend.generate(**_gen_kwargs)
                else:
                    saved_path = backend.generate(**_gen_kwargs)
                frames.append(FrameResult(
                    shot_number=shot_num,
                    frame_type=frame_type,
                    prompt=raw_prompt,
                    formatted_prompt=formatted,
                    image_path=str(saved_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    seed=-1,
                    generated_at=datetime.now(timezone.utc).isoformat(),
                    backend_id=backend_id,
                    narration=shot.get("narration", ""),
                    beat=shot.get("beat", ""),
                    visual_description=shot.get("visual_description", ""),
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

        _gen_kwargs = dict(
            prompt=formatted,
            aspect_ratio=aspect_ratio,
            output_filename=f"{script_id}/shot{shot_number}_{frame_type}.png",
        )
        ref_path = STORYBOARDS_DIR / script_id / "_cast_sheet.png"
        env_path = STORYBOARDS_DIR / script_id / "_env_sheet.png"
        if rs.get_reference_mode_enabled() and ref_path.exists():
            c_ref = ref_path if ref_path.exists() else None
            e_ref = env_path if env_path.exists() else None
            try:
                saved_path = backend.generate(reference_image=c_ref, environment_image=e_ref, **_gen_kwargs)
            except TypeError:
                try:
                    saved_path = backend.generate(reference_image=c_ref, **_gen_kwargs)
                except TypeError:
                    saved_path = backend.generate(**_gen_kwargs)
        else:
            saved_path = backend.generate(**_gen_kwargs)
        results.append(FrameResult(
            shot_number=shot_number, frame_type=frame_type,
            prompt=raw_prompt, formatted_prompt=formatted,
            image_path=str(saved_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            seed=-1,
            generated_at=datetime.now(timezone.utc).isoformat(),
            backend_id=backend.backend_id,
            narration=shot.get("narration", ""),
            beat=shot.get("beat", ""),
            visual_description=shot.get("visual_description", ""),
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