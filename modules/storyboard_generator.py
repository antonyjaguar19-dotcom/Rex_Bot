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
import re
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
from modules import beat_policy as bp
from modules import gpu_utils
from modules import prompt_assembler as pa
from modules import script_generator as sg
from modules import runtime_settings as rs
from modules import prompt_approval as pap
from modules.file_utils import atomic_write_json

CAST_SHEET_SEED = 777  # fixed so the cast sheet is reproducible per script

# Anti-ghost: Z-Image's human bias paints faint T-pose mannequins / "horror-film"
# shadow figures into empty environment frames. Negative terms alone are weak at the
# low cfg atmosphere shots use, so we ALSO assert emptiness positively and bump cfg.
EMPTY_SCENE_POSITIVE = (
    "The scene is completely empty and unoccupied — no people, no characters, "
    "no figures, no silhouettes anywhere in frame."
)

log = logging.getLogger("claw_bot.storyboard_generator")


def _prompt_has_character(script: dict, prompt_text: str) -> bool:
    """True if any cast member's name appears in the actual IMAGE prompt text.

    Note: we scan the rendered prompt, NOT narration — narration may name a
    character who is off-screen in an establishing/environment frame.
    """
    p = (prompt_text or "").lower()
    if not p.strip():
        return False
    for c in script.get("characters", []) or []:
        if isinstance(c, dict):
            name = (c.get("name") or "").strip().lower()
            if name and name in p:
                return True
    return False


def _is_empty_frame(script: dict, shot: dict, formatted: str) -> bool:
    """Decide if a shot is a legitimately character-FREE establishing frame.

    Only a `wide` shot may be empty (a location/atmosphere establishing beat).
    closeup/medium/insert are inherently character shots — never assert emptiness
    on them (doing so on a closeup-of-faces made flux render a bare room). Even a
    wide is only empty when NEITHER the rendered prompt NOR the narration names a
    cast member.
    """
    if (shot.get("shot_type") or "").strip().lower() != "wide":
        return False
    return not (_prompt_has_character(script, formatted)
                or _prompt_has_character(script, shot.get("narration", "")))

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

def _characters_description(script: dict, raw_prompt: str = "", match_text: str = "") -> str:
    """Build the locked character description prepend conditionally.
    Injects a character whose name appears in the prompt OR in `match_text`
    (the shot's narration + speaker). This stops identity drift when the LLM
    writes the first_frame_prompt by description ("small robot") instead of by
    name — the narration still names the character, so the locked token is
    injected and flux renders the right one.
    """
    chars = script.get("characters", [])
    parts = []
    corpus = f"{raw_prompt} {match_text}".lower()
    have_filter = bool(raw_prompt or match_text)
    for ch in chars:
        if isinstance(ch, dict):
            name = ch.get("name", "").strip()

            # Skip this character if their name isn't anywhere in the corpus
            if have_filter and name and name.lower() not in corpus:
                continue

            locked = (ch.get("locked_visual_token") or ch.get("appearance") or "").strip()
            if name and locked:
                parts.append(f"{name} — {locked}")
            elif name:
                parts.append(name)
        elif isinstance(ch, str):
            # Legacy string format
            name = ch.split(",")[0].split("-")[0].strip()
            if have_filter and name and name.lower() not in corpus:
                continue
            parts.append(name)
    return "; ".join(parts) if parts else ""


def _style_suffix(script: dict) -> str:
    """Look up the style the LLM picked for this script."""
    style_id = script.get("style") or sg.get_default_style()
    style_info = sg.get_style_description(style_id)
    return style_info.get("prompt_suffix", "")


def _char_medium(script: dict) -> str:
    """How characters are rendered in the active style's medium."""
    style_id = script.get("style") or sg.get_default_style()
    return sg.get_style_description(style_id).get("character_medium", "")


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

    # Match characters by the shot's narration + speaker too, so a character
    # named only in the narration (prompt described them by type) still gets the
    # locked token injected — prevents the protagonist rendering as a stranger.
    match_text = f"{shot.get('narration', '')} {shot.get('speaker', '')}"
    char_para = _characters_description(script, raw_prompt=cleaned, match_text=match_text)
    # Character-focused shot (closeup/medium/insert) but nobody matched by name —
    # the prompt/narration referred to them as "the twins"/"they" or is a subject-
    # less continuation fragment. Inject the FULL cast so the right beings still
    # render (otherwise flux invents strangers or, flagged as empty, draws a bare
    # room on a closeup-of-faces).
    _stype = (shot.get("shot_type") or "").strip().lower()
    if not char_para and _stype in ("closeup", "medium", "insert"):
        char_para = _characters_description(script)
    # MINIMAL styles (stickman doodle): the detailed character tokens ("in a
    # floral dress") fight the stick-figure simplification — drop the character
    # sheet so the LoRA renders generic stick figures. Names in the scene line
    # still drive who's present.
    try:
        from modules.image_backends.comfyui_sdxl_ipadapter import MINIMAL_STYLES
    except Exception:
        MINIMAL_STYLES = {"stickman"}
    if (script.get("style") or "").strip().lower() in MINIMAL_STYLES:
        char_para = ""
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
        # Strip leading preposition so we don't produce "Scene takes place in in ..."
        s = setting.strip().rstrip(".").lstrip()
        s = re.sub(r"^(in|at|on|inside|within)\s+", "", s, flags=re.IGNORECASE)
        parts.append(f"Scene takes place in {s.lower()}.")

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
    atomic_write_json(path, data)
    log.info(f"Manifest saved: {path}")


# ==============================================================================
# MAIN API
# ==============================================================================

def _kids_backend():
    """Kids storytelling renders on SDXL + style-LoRA + IP-Adapter for character
    consistency (cast-sheet anchored). This is pinned HERE rather than via the
    global `active` backend so the other modes are untouched — horror pins
    flux_lora itself, music keeps the active backend. Falls back to the active
    backend if SDXL+IPA is unavailable/unhealthy."""
    try:
        import importlib
        cfg = model_registry.get_available("image_backend", "comfyui_sdxl_ipadapter")
        if cfg:
            mod = importlib.import_module(cfg["module_path"])
            b = mod.Backend(cfg)
            ok, _msg = b.health_check()
            if ok:
                return b
            log.warning(f"SDXL+IPA backend unhealthy ({_msg}); using active backend.")
    except Exception as e:
        log.warning(f"SDXL+IPA backend unavailable ({e}); using active backend.")
    return ib.get_active_backend()


def _generate_cast_sheet(script: dict, script_id: str, backend, aspect_ratio: str):
    """Generate ONE reference image of all main characters (front-facing, neutral,
    plain background, fixed seed). Returns its Path, or None on failure / no chars.

    This image is the IP-Adapter character anchor, so it MUST render the right
    SPECIES (SDXL+Pixar LoRA is heavily human-biased — animal tokens alone draw a
    human) and MUST be a clean lineup, NOT a contact-sheet collage (the "reference
    sheet" wording + wide aspect triggers grids). Hardened accordingly.
    """
    chars = [c for c in script.get("characters", []) if isinstance(c, dict)]
    tokens = []
    for c in chars:
        name = c.get("name", "").strip()
        tok = (c.get("locked_visual_token") or c.get("appearance") or "").strip()
        if tok:
            tokens.append(f"{name}, {tok}" if name else tok)
    if not tokens:
        return None

    types = {(c.get("type") or "").strip().lower() for c in chars}
    non_human = types and types <= {"animal", "creature"}
    n = len(tokens)
    subject = "anthropomorphic cartoon characters" if non_human else "characters"

    style_suffix = _style_suffix(script)
    parts = []
    if style_suffix:
        parts.append(f"Art style: {style_suffix}.")
    lineup = "a single full-body character" if n == 1 else \
        f"{n} full-body {subject} standing side by side in a row"
    parts.append(
        f"Character reference lineup: {lineup} on a plain solid light-gray studio "
        f"background, front facing, neutral standing pose, even soft lighting, clear "
        f"separation between characters, full body visible, simple flat background, "
        f"no props, no scenery. " + "; ".join(tokens) + "."
    )
    cast_prompt = " ".join(parts)

    # Always block the collage/grid failure; block humans when the cast is animals.
    neg = ("blurry, low quality, deformed, extra limbs, distorted, watermark, text, "
           "grid, collage, contact sheet, multiple panels, split screen, montage, "
           "tiled layout, picture frames")
    if non_human:
        neg += ", human, person, boy, girl, man, woman, human face, human skin, realistic human"

    log.info(f"Generating cast sheet for {script_id} ({n} character(s), "
             f"{'non-human' if non_human else 'human'} cast)")
    try:
        ref_path = backend.generate(
            prompt=cast_prompt,
            negative_prompt=neg,
            aspect_ratio="1:1",          # square avoids the wide contact-sheet grid
            output_filename=f"{script_id}/_cast_sheet.png",
            seed=CAST_SHEET_SEED,
            style=script.get("style"),
        )
        log.info(f"Cast sheet saved: {ref_path}")
        return ref_path
    except Exception as e:
        log.warning(f"Cast sheet generation failed (continuing without reference): {e}")
        return None


def _char_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "char").lower()).strip("_") or "char"


def _generate_character_sheets(script: dict, script_id: str, backend) -> dict:
    """Render ONE clean SOLO reference per character (`_cast_<name>.png`).

    Solo renders are reliable (the matrix proved single-subject = perfect),
    whereas a single combined sheet collapses two different animal species into
    one (rabbit + tortoise -> two rabbits). Each solo sheet is the per-character
    IP-Adapter anchor; multi-character shots composite the relevant solos
    side-by-side (see _shot_reference), so SDXL never has to invent two distinct
    species in one generation.

    Returns {character_name: abs Path}. Also writes a combined _cast_sheet.png
    (montage of the solos) for the dashboard.
    """
    chars = [c for c in script.get("characters", []) if isinstance(c, dict)]
    style_suffix = _style_suffix(script)
    sheets: dict[str, Path] = {}
    for i, c in enumerate(chars):
        name = (c.get("name") or "").strip()
        tok = (c.get("locked_visual_token") or c.get("appearance") or "").strip()
        if not (name and tok):
            continue
        typ = (c.get("type") or "").strip().lower()
        non_human = typ in ("animal", "creature")
        subject = ("a single anthropomorphic cartoon animal" if typ == "animal"
                   else "a single robot/creature" if typ == "creature"
                   else "a single character")
        parts = []
        if style_suffix:
            parts.append(f"Art style: {style_suffix}.")
        parts.append(
            f"Solo character reference of EXACTLY ONE character: {subject}, {tok}, "
            f"completely alone, single subject centered in frame, standing front "
            f"facing in a neutral pose on a plain solid light-gray studio background, "
            f"even soft lighting, full body visible, simple flat background, no props, "
            f"no other characters, only one character in the entire image."
        )
        prompt = " ".join(parts)
        neg = ("blurry, low quality, deformed, extra limbs, distorted, watermark, text, "
               "grid, collage, contact sheet, multiple characters, two characters, "
               "two animals, pair, couple, group, crowd, duplicate, clone, twin, "
               "second character, reflection")
        if non_human:
            neg += ", human, person, boy, girl, man, woman, human face, human skin"
        try:
            p = backend.generate(
                prompt=prompt, negative_prompt=neg, aspect_ratio="1:1",
                output_filename=f"{script_id}/_cast_{_char_slug(name)}.png",
                seed=CAST_SHEET_SEED + i, style=script.get("style"),
            )
            sheets[name] = Path(p)
            log.info(f"Character sheet for {name}: {p}")
        except Exception as e:
            log.warning(f"Character sheet for {name} failed: {e}")
    # Combined montage for the dashboard (best-effort).
    if sheets:
        try:
            _composite_refs(list(sheets.values()),
                            STORYBOARDS_DIR / script_id / "_cast_sheet.png")
        except Exception as e:
            log.warning(f"Combined cast montage failed: {e}")
    return sheets


def _composite_refs(paths: list, out_path: Path) -> Path:
    """Stitch per-character solo references side-by-side on a light-gray canvas.
    Used as the IP-Adapter reference for a multi-character shot — both species
    are already rendered correctly, so IP-Adapter conditions on both distinctly
    instead of SDXL collapsing them."""
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in paths if Path(p).exists()]
    if not imgs:
        raise RuntimeError("no reference images to composite")
    if len(imgs) == 1:
        imgs[0].save(out_path)
        return out_path
    h = min(im.height for im in imgs)
    resized = [im.resize((int(im.width * h / im.height), h)) for im in imgs]
    gap = 16
    w = sum(im.width for im in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (w, h), (228, 228, 230))
    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def _shot_present_chars(script: dict, shot: dict) -> list:
    """Names of cast members referenced in this shot (narration + speaker + prompt)."""
    corpus = (f"{shot.get('narration','')} {shot.get('speaker','')} "
              f"{shot.get('first_frame_prompt','')}").lower()
    out = []
    for c in script.get("characters", []):
        if isinstance(c, dict):
            n = (c.get("name") or "").strip()
            if n and n.lower() in corpus:
                out.append(n)
    return out


def _shot_reference(script: dict, shot: dict, sheets: dict, story_dir: Path) -> list:
    """Per-shot IP-Adapter reference set — a LIST of this shot's character solo
    sheets. The backend feeds 2+ as an image BATCH (combine_embeds=concat) so
    both identities are held WITHOUT a side-by-side layout (compositing them into
    one image made the output a split-screen). 1 char present -> [its sheet];
    2+ -> [each sheet]; a character-focused shot naming no one -> the whole cast."""
    if not sheets:
        return []
    present = [n for n in _shot_present_chars(script, shot) if n in sheets]
    if not present:
        if (shot.get("shot_type") or "").strip().lower() in ("closeup", "medium", "insert"):
            present = list(sheets.keys())
        else:
            return []
    return [sheets[n] for n in present]


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
            style=script.get("style"),
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

    backend = _kids_backend()
    backend_id = backend.backend_id
    style_used = script.get("style", "pixar")
    log.info(
        f"Generating storyboard for script {script_id} "
        f"({len(shots)} shots, style={style_used}) using backend {backend_id}"
    )

    story_dir = STORYBOARDS_DIR / script_id
    story_dir.mkdir(parents=True, exist_ok=True)

    # Per-character cast sheets (one clean solo reference each) + environment.
    # Per-shot we pick/compose the right character references — see _shot_reference.
    # MINIMAL styles (stickman doodle) have no distinctive character identity, so a
    # cast sheet + IP-Adapter would FIGHT the simplification — render prompt-only.
    char_sheets: dict = {}
    environment_image = None
    try:
        from modules.image_backends.comfyui_sdxl_ipadapter import MINIMAL_STYLES
    except Exception:
        MINIMAL_STYLES = {"stickman"}
    minimal_style = (script.get("style") or "").strip().lower() in MINIMAL_STYLES
    if rs.get_reference_mode_enabled() and not minimal_style:
        if progress_callback:
            progress_callback("Generating per-character reference sheets...", 0, len(shots))
        char_sheets = _generate_character_sheets(script, script_id, backend)
        if progress_callback:
            progress_callback("Generating environment sheet...", 0, len(shots))
        environment_image = _generate_environment_sheet(script, script_id, backend, aspect_ratio)
    elif minimal_style:
        log.info(f"Style '{script.get('style')}' is minimal — skipping cast sheet / IP-Adapter "
                 f"(prompt-only render).")

    frames: list[FrameResult] = []
    total = len(shots)
    current = 0

    try:
        for shot in shots:
            shot_num = shot.get("shot_number", current + 1)
            # Per-shot IP-Adapter references: this shot's character solo sheets.
            shot_refs = _shot_reference(script, shot, char_sheets, story_dir)
            reference_image = shot_refs[0] if shot_refs else None
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

                # PREFERRED PATH: user-approved prompt + seed from prompt_approval
                approved = pap.get_shot_prompts(script_id, shot_num) or {}
                approved_img_prompt = (approved.get("image_prompt") or "").strip()
                approved_img_seed = approved.get("image_seed", -1)
                if approved_img_prompt:
                    formatted = approved_img_prompt
                    log.info(f"[{current}/{total}] Using user-approved image prompt for shot {shot_num}")
                else:
                    # FALLBACK: legacy on-the-fly assembly (no approval pass run)
                    try:
                        formatted = pa.assemble_image_prompt(
                            script=script,
                            shot=shot,
                            style_suffix=_style_suffix(script),
                            frame_type=frame_type,
                        )
                    except Exception as _e:
                        log.warning(
                            f"Shot {shot_num} {frame_type}: LLM assembler failed "
                            f"({_e}); falling back to mechanical assembler"
                        )
                        formatted = _rewrite_as_paragraph(raw_prompt, script, shot)
                beat = (shot.get("beat") or "").strip().lower()
                # Beat-aware negative augment + cfg bias
                from modules.image_backends.comfyui_zimage_base import DEFAULT_NEGATIVE as _ZIMG_NEG
                default_neg = getattr(backend, "DEFAULT_NEGATIVE", None) or _ZIMG_NEG
                negative = bp.merge_negative(default_neg, beat)
                backend_default_cfg = float(
                    getattr(backend, "cfg", model_registry.get_active("image_backend").get("cfg", 1.0))
                )
                effective_cfg = bp.image_cfg_for(beat, backend_default_cfg)
                # Anti-ghost: only a WIDE establishing frame with no character in
                # prompt OR narration is treated as empty. Negative terms alone are
                # weak (esp. at atmosphere's lowered cfg), so assert emptiness
                # POSITIVELY and raise cfg. Never on a closeup/medium/insert.
                no_char_frame = _is_empty_frame(script, shot, formatted)
                if no_char_frame:
                    negative = bp.merge_negative(negative, "atmosphere")
                    if "empty" not in formatted.lower():
                        formatted = formatted.rstrip() + " " + EMPTY_SCENE_POSITIVE
                    effective_cfg = max(effective_cfg, backend_default_cfg + 1.0)
                log.info(f"[{current}/{total}] Rendering shot{shot_num}_{frame_type} "
                         f"(beat={beat}, cfg={effective_cfg}, empty_frame={no_char_frame})")

                # VRAM pre-flight — make sure we have headroom before submitting
                try:
                    gpu_utils.ensure_vram_free(min_gb=5.0, force_ollama_unload=True)
                except Exception as _e:
                    log.warning(f"VRAM pre-flight noop: {_e}")

                # Per-shot retry with backoff — handles transient Comfy timeouts
                # Character consistency: locked tokens in the prompt + a cast-sheet
                # reference latent (when reference mode is on). The reference is passed
                # only if the backend supports it; otherwise we fall back cleanly.
                _gen_kwargs = dict(
                    prompt=formatted,
                    negative_prompt=negative,
                    aspect_ratio=aspect_ratio,
                    output_filename=f"{script_id}/shot{shot_num}_{frame_type}.png",
                    cfg_override=effective_cfg,
                    style=script.get("style"),   # SDXL+IPA picks the per-style LoRA; others ignore
                )
                # Pass user-approved seed (if any). -1 = random (let backend pick).
                if isinstance(approved_img_seed, int) and approved_img_seed > 0:
                    _gen_kwargs["seed"] = approved_img_seed
                last_err = None
                saved_path = None
                for attempt in range(1, 4):
                    try:
                        if reference_image is not None or environment_image is not None:
                            try:
                                saved_path = backend.generate(
                                    reference_image=reference_image,
                                    reference_images=shot_refs,
                                    environment_image=environment_image,
                                    **_gen_kwargs
                                )
                            except TypeError:
                                try:
                                    saved_path = backend.generate(
                                        reference_image=reference_image, **_gen_kwargs
                                    )
                                except TypeError:
                                    # Backend accepts neither reference nor cfg_override/
                                    # negative_prompt — drop the optional extras (mirrors
                                    # the no-reference path below).
                                    fallback = {k: v for k, v in _gen_kwargs.items()
                                                if k not in ("cfg_override", "negative_prompt", "style")}
                                    saved_path = backend.generate(**fallback)
                        else:
                            try:
                                saved_path = backend.generate(**_gen_kwargs)
                            except TypeError:
                                # Backend doesn't accept cfg_override / negative_prompt — drop them
                                fallback = {k: v for k, v in _gen_kwargs.items()
                                            if k not in ("cfg_override", "negative_prompt", "style")}
                                saved_path = backend.generate(**fallback)
                        break  # success
                    except Exception as e:
                        last_err = e
                        log.warning(
                            f"Shot {shot_num} {frame_type} attempt {attempt}/3 failed: "
                            f"{type(e).__name__}: {e}"
                        )
                        if attempt < 3:
                            import time as _time
                            _time.sleep(5 * attempt)  # 5s, 10s
                            try:
                                gpu_utils.free_comfyui_vram()
                            except Exception:
                                pass
                if saved_path is None:
                    raise RuntimeError(
                        f"Shot {shot_num} {frame_type} failed after 3 attempts: {last_err}"
                    )
                used_seed = int(getattr(backend, "_last_seed", -1) or -1)
                frames.append(FrameResult(
                    shot_number=shot_num,
                    frame_type=frame_type,
                    prompt=raw_prompt,
                    formatted_prompt=formatted,
                    image_path=str(saved_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    seed=used_seed,
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

    backend = _kids_backend()
    story_dir = STORYBOARDS_DIR / script_id
    story_dir.mkdir(parents=True, exist_ok=True)

    results: list[FrameResult] = []
    for frame_type in frames_to_redo:
        raw_prompt = shot.get(f"{frame_type}_frame_prompt") or shot.get("visual_description", "")

        approved = pap.get_shot_prompts(script_id, shot_number) or {}
        approved_img_prompt = (approved.get("image_prompt") or "").strip()
        approved_img_seed = approved.get("image_seed", -1)
        if approved_img_prompt:
            formatted = approved_img_prompt
            log.info(f"Regen shot{shot_number} {frame_type}: using user-approved prompt")
        else:
            try:
                formatted = pa.assemble_image_prompt(
                    script=script,
                    shot=shot,
                    style_suffix=_style_suffix(script),
                    frame_type=frame_type,
                    character_medium=_char_medium(script),
                )
            except Exception as _e:
                log.warning(
                    f"Regen shot{shot_number} {frame_type}: LLM assembler failed "
                    f"({_e}); falling back to mechanical assembler"
                )
                formatted = _rewrite_as_paragraph(raw_prompt, script, shot)
        log.info(f"Re-rendering shot{shot_number}_{frame_type}")

        beat = (shot.get("beat") or "").strip().lower()
        from modules.image_backends.comfyui_zimage_base import DEFAULT_NEGATIVE as _ZIMG_NEG
        default_neg = getattr(backend, "DEFAULT_NEGATIVE", None) or _ZIMG_NEG
        negative = bp.merge_negative(default_neg, beat)
        backend_default_cfg = float(
            getattr(backend, "cfg", model_registry.get_active("image_backend").get("cfg", 1.0))
        )
        effective_cfg = bp.image_cfg_for(beat, backend_default_cfg)
        # Anti-ghost: only a WIDE establishing frame with no character in prompt
        # OR narration is empty — never a closeup/medium/insert.
        no_char_frame = _is_empty_frame(script, shot, formatted)
        if no_char_frame:
            negative = bp.merge_negative(negative, "atmosphere")
            if "empty" not in formatted.lower():
                formatted = formatted.rstrip() + " " + EMPTY_SCENE_POSITIVE
            effective_cfg = max(effective_cfg, backend_default_cfg + 1.0)
        _gen_kwargs = dict(
            prompt=formatted,
            negative_prompt=negative,
            aspect_ratio=aspect_ratio,
            output_filename=f"{script_id}/shot{shot_number}_{frame_type}.png",
            cfg_override=effective_cfg,
            style=script.get("style"),
        )
        if isinstance(approved_img_seed, int) and approved_img_seed > 0:
            _gen_kwargs["seed"] = approved_img_seed
        try:
            gpu_utils.ensure_vram_free(min_gb=5.0, force_ollama_unload=True)
        except Exception:
            pass
        story_dir = STORYBOARDS_DIR / script_id
        env_path = story_dir / "_env_sheet.png"
        # Rebuild the per-character sheet map from disk (_cast_<slug>.png); pick
        # this shot's reference. Fall back to the combined sheet if solos absent.
        try:
            from modules.image_backends.comfyui_sdxl_ipadapter import MINIMAL_STYLES
        except Exception:
            MINIMAL_STYLES = {"stickman"}
        minimal_style = (script.get("style") or "").strip().lower() in MINIMAL_STYLES
        shot_refs = []
        if rs.get_reference_mode_enabled() and not minimal_style:
            sheets = {}
            for c in script.get("characters", []):
                if isinstance(c, dict) and c.get("name"):
                    p = story_dir / f"_cast_{_char_slug(c['name'])}.png"
                    if p.exists():
                        sheets[c["name"]] = p
            if sheets:
                shot_refs = _shot_reference(script, shot, sheets, story_dir)
            else:
                combined = story_dir / "_cast_sheet.png"
                shot_refs = [combined] if combined.exists() else []
        c_ref = shot_refs[0] if shot_refs else None
        if c_ref is not None:
            e_ref = env_path if env_path.exists() else None
            try:
                saved_path = backend.generate(reference_image=c_ref, reference_images=shot_refs,
                                              environment_image=e_ref, **_gen_kwargs)
            except TypeError:
                try:
                    saved_path = backend.generate(reference_image=c_ref, **_gen_kwargs)
                except TypeError:
                    saved_path = backend.generate(**_gen_kwargs)
        else:
            try:
                saved_path = backend.generate(**_gen_kwargs)
            except TypeError:
                fallback = {k: v for k, v in _gen_kwargs.items()
                            if k not in ("cfg_override", "negative_prompt", "style")}
                saved_path = backend.generate(**fallback)
        used_seed = int(getattr(backend, "_last_seed", -1) or -1)
        results.append(FrameResult(
            shot_number=shot_number, frame_type=frame_type,
            prompt=raw_prompt, formatted_prompt=formatted,
            image_path=str(saved_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            seed=used_seed,
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