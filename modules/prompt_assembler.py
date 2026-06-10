"""
Claw Bot — Prompt Assembler

Final-stage LLM pass that builds the actual image-generation prompt sent to
ComfyUI for each shot.

Why this exists:
  The old mechanical assembler (storyboard_generator._rewrite_as_paragraph)
  concatenated style + character-sheet + tailor_body + setting and used a
  name-in-text filter to decide which characters to inject. For atmosphere
  beats it sometimes injected a character anyway (pronoun leak, partial
  match, environment_sheet kwarg fallback, etc.).

  This module hands the job to the same local LLM the bot already uses,
  giving it explicit knowledge of:
    - shot beat (atmosphere → NO characters, ever)
    - tailored frame body (the source of truth for what's in the shot)
    - character roster with locked_visual_token (only inject if they belong)
    - setting, style suffix

  The LLM emits ONE clean Z-Image paragraph.

Pipeline position:
  script_generator → shot_tailor → prompt_assembler → image backend
"""

import json
import logging
import re
import time as _t
from typing import Optional

import requests

from modules import model_registry

log = logging.getLogger("claw_bot.prompt_assembler")


# Beats where the final image must contain ZERO characters.
NO_CHARACTER_BEATS = {"atmosphere"}


# ==============================================================================
# SYSTEM PROMPT — narrow, schema-light
# ==============================================================================

def _build_system_prompt(beat: str, no_chars: bool) -> str:
    char_rule = (
        "ABSOLUTE RULE: this shot's beat is a BREATHING SHOT. The output prompt "
        "MUST contain ZERO characters. No people. No animals. No silhouettes. "
        "No distant figures. No body parts. No clothing. Setting + camera + "
        "light only. If you mention a character in the output, you have failed."
        if no_chars else
        "Inject EVERY character whose locked_visual_token is provided BELOW. "
        "Use the locked_visual_token verbatim for each character — never "
        "paraphrase the appearance. Combine each character's appearance with "
        "their pose/action in a single sentence per character. Do not mention "
        "any character whose locked_visual_token was NOT provided to you."
    )

    return f"""You output ONE paragraph of plain text — the final image prompt sent to Z-Image Turbo, a paragraph-style diffusion model. No JSON. No markdown. No labels. No preamble. Just the prompt paragraph.

You are a senior prompt engineer. You assemble inputs into ONE clean Z-Image prompt.

# THIS SHOT'S BEAT: "{beat}"

# CHARACTER RULE (this is the most important rule)

{char_rule}

# Z-IMAGE PROMPT PHILOSOPHY

Z-Image renders best from prose that reads like a museum description of a painting that already exists. Concrete nouns + present-tense verbs + spatial relationships. Z-Image ignores vague atmospheric words and drops entities buried under adjectives.

GOOD: "A small green snail with a brown-and-white striped shell rests on a wide green leaf. Three orange marigolds grow behind it. Morning light comes from the upper left."

BAD: "A serene moment of curiosity unfolds as the brave little snail experiences wonder in a vibrant garden filled with gentle life."

# COMPOSITION

1. **Order matters.** Style anchor first → subject + key attribute next → secondary entities → setting → lighting/camera. Z-Image weights early tokens more.
2. **Front-load the main subject** in the first sentence.
3. **Concrete, present-tense, visual verbs.** Yes: rests, stands, leans, glides, faces. No: experiences, feels, embodies, radiates.
4. **Camera framing as spatial info, not film-school jargon.** "the snail fills most of the frame, viewed from slightly above" — not "cinematic close-up".
5. **Lighting = direction + color.** "warm yellow light from the upper right" — not "soft golden glow".
6. **Length: 60-110 words.** Tight. Every word does visual work.
7. **No abstract mood words.** Banned: serene, joyful, wonder, vibrant, lush, magical, whimsical, cinematic, expressive, captures, evokes, dreamlike, ethereal.
8. **No background activity without a physical anchor.** Don't write "kids playing"; write "two kids sit on a picnic blanket on the grass". Every character or background actor needs a surface or object they touch.
9. **No floating figures.** Swings have visible chains and frames. Slides have visible support posts. Birds in mid-air = ok. Children in mid-air = not ok unless on a swing or mid-jump near a surface.
10. **Anatomy lock for non-human characters.** Snails glide on underside, shell attached; frogs crouch low with folded hind legs; birds keep body horizontal with symmetric wings; fish undulate side-to-side; insects keep body horizontal with fluttering wings; quadrupeds move on all four legs.

# ASSEMBLY ORDER (for character beats)

Sentence 1: art style anchor (use the STYLE SUFFIX provided below verbatim).
Sentence 2: main character — appearance (verbatim locked_visual_token) + pose + position in frame.
Sentence 3: any secondary character(s) — same treatment.
Sentence 4: setting elements anchored (objects, surfaces, background fixtures).
Sentence 5: camera framing + lighting direction + color.

# ASSEMBLY ORDER (for atmosphere / breathing beats)

Sentence 1: art style anchor.
Sentence 2-3: setting subject (the place, the weather, the time of day) with concrete props.
Sentence 4-5: camera framing + lighting direction + color.
(Zero characters. Zero body parts. Zero clothing.)

# OUTPUT

ONE paragraph, plain text, 60-110 words. Start with the style anchor. End with lighting/camera. Nothing else."""


# ==============================================================================
# OLLAMA CALL
# ==============================================================================

def _call_llm(user_prompt: str, system_prompt: str, *, temperature: float = 0.35) -> str:
    cfg = model_registry.get_active("llm_backend")
    model_name = (
        cfg.get("model_id") or cfg.get("model_name") or cfg.get("model")
        or "qwen2.5:14b-instruct-q6_K"
    )
    url = cfg.get("server_url", "http://127.0.0.1:11434") + "/api/generate"

    combined = (
        f"SYSTEM INSTRUCTIONS (STRICTLY FOLLOW THESE):\n{system_prompt}\n\n"
        f"USER REQUEST:\n{user_prompt}"
    )
    payload = {
        "model": model_name,
        "prompt": combined,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_ctx": 8192,
            "num_predict": 512,
        },
    }
    log.info(f"Assembler → {model_name}, temp={temperature}")
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    return r.json().get("response", "").strip()


# ==============================================================================
# OUTPUT NORMALIZATION
# ==============================================================================

def _strip_meta(text: str) -> str:
    """Remove <think> blocks, code fences, leading labels."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"```[a-z]*", "", cleaned, flags=re.IGNORECASE).replace("```", "")
    cleaned = re.sub(
        r"^(?:Prompt|Output|Final|Image prompt)\s*[:\-]\s*",
        "",
        cleaned.strip(),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    # Collapse to one paragraph
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _scrub_character_names(text: str, names: list[str]) -> tuple[str, list[str]]:
    """For breathing beats, hard-strip any sentence that names a character."""
    if not text or not names:
        return text, []
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    name_pattern = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in names if n) + r")\b",
        re.IGNORECASE,
    )
    kept, dropped = [], []
    for s in sentences:
        (dropped if name_pattern.search(s) else kept).append(s)
    return " ".join(kept).strip(), dropped


# ==============================================================================
# CHARACTER ROSTER FILTER
# ==============================================================================

def _shot_relevant_characters(script: dict, shot: dict) -> list[dict]:
    """Return only the characters that BELONG in this shot.

    A character belongs if:
      - Their name appears in narration, visual_description, first_frame_prompt,
        last_frame_prompt, or character_pose.
    """
    chars = script.get("characters", [])
    if not isinstance(chars, list):
        return []

    haystack_parts = [
        shot.get("narration", ""),
        shot.get("visual_description", ""),
        shot.get("first_frame_prompt", ""),
        shot.get("last_frame_prompt", ""),
        shot.get("character_pose", ""),
    ]
    haystack = " ".join(p for p in haystack_parts if isinstance(p, str)).lower()
    if not haystack.strip():
        return []

    relevant = []
    for c in chars:
        if not isinstance(c, dict):
            continue
        name = (c.get("name") or "").strip()
        if name and name.lower() in haystack:
            relevant.append(c)
    return relevant


# ==============================================================================
# PUBLIC API
# ==============================================================================

def assemble_image_prompt(
    script: dict,
    shot: dict,
    style_suffix: str = "",
    frame_type: str = "first",
) -> str:
    """
    Build the final Z-Image prompt for a single shot/frame via the LLM.

    Args:
        script:       full script dict (for characters + setting)
        shot:         single shot dict (must have beat + first_frame_prompt etc.)
        style_suffix: art-style anchor sentence (from styles.json)
        frame_type:   "first" or "last" — picks which body prompt to seed from

    Returns:
        One-paragraph plain-text prompt. Raises on hard LLM failure — caller
        should catch and fall back to the mechanical assembler.
    """
    _start = _t.time()

    beat = (shot.get("beat") or "").strip().lower() or "hook"
    no_chars = beat in NO_CHARACTER_BEATS

    body_key = f"{frame_type}_frame_prompt"
    body = (shot.get(body_key) or shot.get("first_frame_prompt") or shot.get("visual_description") or "").strip()
    if not body:
        raise ValueError(f"Shot {shot.get('shot_number')} has no usable prompt body for {body_key}")

    setting = (script.get("setting") or "").strip()
    # Strip leading prepositions from setting so the LLM doesn't double them
    setting_clean = re.sub(r"^(in|at|on|inside|within)\s+", "", setting, flags=re.IGNORECASE).strip().rstrip(".")

    # Only the characters that belong in this shot (or none for breathing beats)
    relevant_chars = [] if no_chars else _shot_relevant_characters(script, shot)
    char_lines = []
    for c in relevant_chars:
        name = (c.get("name") or "").strip()
        lock = (c.get("locked_visual_token") or c.get("appearance") or "").strip()
        if name and lock:
            char_lines.append(f"- {name}: {lock}")

    user_payload = {
        "beat": beat,
        "frame_type": frame_type,
        "style_suffix": style_suffix or "",
        "setting": setting_clean,
        "tailored_frame_body": body,
        "character_pose": (shot.get("character_pose") or "").strip(),
        "camera_angle": (shot.get("camera_angle") or "").strip(),
        "narration": (shot.get("narration") or "").strip(),
    }
    if char_lines:
        user_payload["characters_for_this_shot"] = "\n".join(char_lines)
    elif no_chars:
        user_payload["characters_for_this_shot"] = (
            "NONE — this is a breathing shot. Do not include any character."
        )
    else:
        user_payload["characters_for_this_shot"] = (
            "NONE — the tailored body does not name any character. Render setting only."
        )

    user_prompt = (
        f"Assemble the final Z-Image prompt for this shot. Inputs:\n\n"
        f"```\n{json.dumps(user_payload, indent=2, ensure_ascii=False)}\n```\n\n"
        f"Output ONE paragraph of plain text per the system rules. Start with the "
        f"style anchor sentence (from `style_suffix`). For breathing beats, "
        f"include ZERO characters. For character beats, inject every character "
        f"from `characters_for_this_shot` with their full locked appearance "
        f"verbatim. End with camera framing + lighting. 60-110 words."
    )

    raw = _call_llm(user_prompt, _build_system_prompt(beat, no_chars))
    final = _strip_meta(raw)

    # Safety net: if it's a breathing beat and the LLM still slipped a name in,
    # scrub the offending sentence(s).
    if no_chars:
        all_names = [
            (c.get("name") or "").strip()
            for c in script.get("characters", [])
            if isinstance(c, dict) and (c.get("name") or "").strip()
        ]
        scrubbed, dropped = _scrub_character_names(final, all_names)
        if dropped:
            log.warning(
                f"Assembler scrubbed {len(dropped)} character-naming sentence(s) "
                f"from breathing beat output: {dropped}"
            )
            final = scrubbed

    if not final or len(final.split()) < 25:
        raise RuntimeError(
            f"Assembler output too short ({len(final.split())} words). Falling back."
        )

    log.info(
        f"Assembled prompt for shot {shot.get('shot_number')} ({beat}, "
        f"chars={len(char_lines)}) in {_t.time() - _start:.1f}s, "
        f"{len(final.split())} words"
    )
    return final


# ==============================================================================
# MOTION PROMPT (Wan 2.2 14B I2V)
# ==============================================================================

def _build_motion_system_prompt(beat: str, no_chars: bool) -> str:
    char_rule = (
        "This is a BREATHING SHOT — no characters in the starting frame. "
        "Describe ONLY environmental motion: drifting leaves, swaying grass, "
        "shifting light, slow camera push or pull, gentle parallax. "
        "Never mention people, animals, or named characters."
        if no_chars else
        "Characters in the starting frame perform ONE clear, simple action. "
        "Use present-tense action verbs. No dialog. No mouth movement. No lip sync. "
        "Never write speech."
    )

    return f"""You output ONE paragraph of plain text — the motion prompt for Wan 2.2 14B image-to-video. No JSON. No markdown. No labels. No preamble. Just the motion paragraph.

You are a senior animation director. You describe how the starting frame should ANIMATE over ~3-5 seconds. The starting frame is fixed; you describe the motion that emerges from it.

# THIS SHOT'S BEAT: "{beat}"

# CHARACTER RULE

{char_rule}

# WAN 2.2 I2V PROMPT PHILOSOPHY

Wan 2.2 reads physical action verbs and camera movement directives. It ignores adjectives about mood. Lead with the largest motion in the frame.

GOOD: "The snail slowly inches forward along the leaf. Its eye-stalks tilt up. A small breeze rustles the marigolds behind it. The camera drifts in slightly."

BAD: "A beautiful sense of wonder unfolds as the brave little snail experiences the joy of discovery in this magical garden moment."

# COMPOSITION RULES

1. **First sentence: main subject motion.** Concrete verb. What moves first.
2. **Second sentence: secondary motion** (other characters, environment, props).
3. **Last sentence: camera move.** Short, explicit: "slow push in", "static hold", "gentle pan left", "subtle drift up".
4. **Length: 35-70 words.** Tight. Every word maps to pixels moving.
5. **No abstract motion words.** Banned: gracefully, beautifully, magically, dreamlike, ethereal, captures, evokes.
6. **No speech, no mouth movement, no lip sync.** Add: "No speech. No mouth movement." at end.
7. **No mode change.** Don't introduce new characters, new locations, or scene cuts. Wan animates the existing frame.

# OUTPUT

ONE paragraph, plain text, 35-70 words. Pure motion description. End with the camera move and the no-speech disclaimer."""


def assemble_motion_prompt(
    script: dict,
    shot: dict,
    approved_image_prompt: str = "",
) -> str:
    """
    Build the Wan 2.2 14B I2V motion prompt for a single shot via the LLM.

    Args:
        script: full script dict
        shot:   single shot dict (must have beat)
        approved_image_prompt: the already-approved Z-Image prompt for this
            shot — gives the LLM ground truth on what's in the starting frame.

    Returns:
        One-paragraph plain-text motion prompt. Raises on LLM failure.
    """
    _start = _t.time()

    beat = (shot.get("beat") or "").strip().lower() or "hook"
    no_chars = beat in NO_CHARACTER_BEATS

    # Seed motion description from the shot's own motion_prompt if present,
    # else fall back to narration / visual_description for context.
    motion_seed = (
        (shot.get("motion_prompt") or "").strip()
        or (shot.get("action") or "").strip()
        or (shot.get("visual_description") or "").strip()
        or (shot.get("narration") or "").strip()
    )

    relevant_chars = [] if no_chars else _shot_relevant_characters(script, shot)
    char_names = [c.get("name", "").strip() for c in relevant_chars if c.get("name")]

    user_payload = {
        "beat": beat,
        "starting_frame_description": approved_image_prompt or shot.get("first_frame_prompt", ""),
        "intended_motion_seed": motion_seed,
        "camera_angle": (shot.get("camera_angle") or "").strip(),
        "character_pose": (shot.get("character_pose") or "").strip(),
        "characters_in_frame": char_names if char_names else (
            "NONE — breathing shot, environment motion only." if no_chars
            else "NONE — no character named in this shot."
        ),
    }

    user_prompt = (
        f"Write the Wan 2.2 14B I2V motion prompt for this shot. Inputs:\n\n"
        f"```\n{json.dumps(user_payload, indent=2, ensure_ascii=False)}\n```\n\n"
        f"Output ONE paragraph, 35-70 words, per the system rules. Describe how "
        f"the starting frame animates. End with camera direction + "
        f"'No speech. No mouth movement.'"
    )

    raw = _call_llm(user_prompt, _build_motion_system_prompt(beat, no_chars), temperature=0.45)
    final = _strip_meta(raw)

    if no_chars:
        all_names = [
            (c.get("name") or "").strip()
            for c in script.get("characters", [])
            if isinstance(c, dict) and (c.get("name") or "").strip()
        ]
        scrubbed, dropped = _scrub_character_names(final, all_names)
        if dropped:
            log.warning(
                f"Motion assembler scrubbed {len(dropped)} character-naming "
                f"sentence(s) from breathing beat output: {dropped}"
            )
            final = scrubbed

    # Always append the no-speech guard (belt-and-suspenders)
    if "no speech" not in final.lower():
        final = final.rstrip(".") + ". No speech. No mouth movement."

    if not final or len(final.split()) < 15:
        raise RuntimeError(
            f"Motion assembler output too short ({len(final.split())} words). Falling back."
        )

    log.info(
        f"Assembled motion prompt for shot {shot.get('shot_number')} ({beat}) "
        f"in {_t.time() - _start:.1f}s, {len(final.split())} words"
    )
    return final


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import sys
    from pathlib import Path
    if len(sys.argv) < 2:
        print("Usage: python prompt_assembler.py <path_to_script.json> [shot_number]")
        sys.exit(1)
    src = Path(sys.argv[1])
    shot_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    s = json.loads(src.read_text(encoding="utf-8"))
    sh = next((x for x in s.get("shots", []) if x.get("shot_number") == shot_num), None)
    if not sh:
        print(f"Shot {shot_num} not found"); sys.exit(1)
    img = assemble_image_prompt(s, sh, style_suffix="rendered in warm Pixar-style 3D animation")
    print("\n=== IMAGE ===\n" + img)
    mot = assemble_motion_prompt(s, sh, approved_image_prompt=img)
    print("\n=== MOTION ===\n" + mot)
