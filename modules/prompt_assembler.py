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

# shot_type → concrete spatial framing the diffusion model understands.
SHOT_TYPE_FRAMING = {
    "wide": ("the whole location is visible and any character occupies less "
             "than a third of the frame height, viewed from a distance"),
    "medium": ("the character is seen from the waist up or full body, "
               "filling about half the frame"),
    "closeup": ("the character's face fills most of the frame"),
    "insert": ("the hands (or paws or beak) and the object they touch fill "
               "the frame; the face is partly or fully out of frame"),
}


def _build_system_prompt(beat: str, no_chars: bool, shot_type: str = "") -> str:
    char_rule = (
        "ABSOLUTE RULE: this shot's beat is a BREATHING SHOT. The output prompt "
        "MUST contain ZERO characters. No people. No animals. No silhouettes. "
        "No distant figures. No body parts. No clothing. Setting + camera + "
        "light only. If you mention a character in the output, you have failed."
        if no_chars else
        "Inject EVERY character whose locked_visual_token is provided BELOW. "
        "Use the locked_visual_token verbatim for each character — never "
        "paraphrase the appearance, and NEVER drop the age words (the token "
        "starts with the character's age; it must survive into your output "
        "verbatim). Combine each character's appearance with their pose/action "
        "in a single sentence per character. State each character's visual "
        "traits ONCE, only via the locked_visual_token — do NOT add extra "
        "adjectives elsewhere that repeat traits already in the token (if the "
        "token says 'white fur', never also write 'snowy-white fur gleaming'). "
        "After that one sentence, refer to the character by NAME only. "
        "Do not mention any character whose locked_visual_token was NOT "
        "provided to you. "
        "Each character below is listed as '- Name: appearance  [Name is he/him]'. "
        "BIND that name to that exact appearance (which states the species) and "
        "pronoun. NEVER call one character by the other's name or species, and "
        "NEVER give a character a body part or feature that belongs to the other "
        "(e.g. do not give a rabbit a shell, or a tortoise long ears). The text in "
        "[square brackets] is an INSTRUCTION — obey it but NEVER copy the brackets, "
        "the word 'pronoun', or the 'he/him'/'she/her' label into your prose; weave "
        "the pronoun in naturally. Use ONLY each character's given "
        "pronoun (he/him or she/her), every time — never switch it. If two "
        "characters are present, keep them clearly separated in space with explicit "
        "depth (who is in front, who is behind, who is left/right) so neither "
        "floats or overlaps the other. "
        "Describe ONLY what is visibly inside this frame. NEVER write that a "
        "character is absent, implied, off-screen, 'not present', 'unseen', or "
        "'implied by context'. If a character is not in this shot, do not mention "
        "them at all. "
        "If the inputs name a `speaking_character`, THAT character is the main "
        "subject of this shot: place them in the foreground with their face "
        "turned toward the camera (for a closeup their face fills most of the "
        "frame) so the viewer can see who is talking and their mouth is visible. "
        "Any other character is secondary — smaller and/or further back."
    )

    framing = SHOT_TYPE_FRAMING.get((shot_type or "").strip().lower(), "")
    shot_type_rule = (
        f"\n# SHOT TYPE: \"{shot_type}\"\n\n"
        f"Compose the frame so that: {framing}. "
        f"Do NOT paste that wording verbatim, and do NOT write rule-like or "
        f"camera-operator phrases ('the camera framing shows', 'occupies less "
        f"than a third of the frame height', 'a medium shot that fills half the "
        f"frame', 'viewed from a distance the whole location is visible'). "
        f"Instead TRANSLATE it into plain painting language — where the subject "
        f"sits in the picture, how near or far it reads, and what surrounds it "
        f"(e.g. 'the rabbit is small against the wide meadow' or 'the tortoise's "
        f"face nearly fills the picture'). This framing overrides any framing "
        f"implied by the body text.\n"
        if framing else ""
    )

    return f"""You output ONE paragraph of plain text — the final image prompt sent to Z-Image Turbo, a paragraph-style diffusion model. No JSON. No markdown. No labels. No preamble. Just the prompt paragraph.

You are a senior prompt engineer. You assemble inputs into ONE clean Z-Image prompt.

# THIS SHOT'S BEAT: "{beat}"
{shot_type_rule}

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
5. **Lighting = direction + color.** "warm yellow light from the upper right" — not "soft golden glow". The inputs give a `light_direction`; use THAT direction verbatim in EVERY shot so lighting never flips between shots (continuity).
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
    # Image + motion prompt craft is a reasoning/creativity task — route it to
    # the 'prompt' role (a stronger model) when configured; fall back to active.
    cfg = model_registry.get_for_role("prompt") or model_registry.get_active("llm_backend")
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
    # Defensive: strip any character-label syntax the LLM copied verbatim from
    # the reference block ("[Pip is she/her]", "she/her: ...").
    cleaned = re.sub(r"\[[^\]]*\b(?:he/him|she/her)\b[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:he/him|she/her)\s*[:,]\s*", "", cleaned, flags=re.IGNORECASE)
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
    spk = (shot.get("speaker") or "").strip().lower()
    for c in chars:
        if not isinstance(c, dict):
            continue
        name = (c.get("name") or "").strip()
        if not name:
            continue
        # A character belongs if named in the shot text OR is the speaker of the
        # line (dialogue narration rarely names the speaker, so name-match alone
        # silently drops the talker — the very face we may need to lip-sync).
        if name.lower() in haystack or name.lower() == spk:
            relevant.append(c)
    return relevant


def _resolve_shot_characters(script: dict, shot: dict) -> list[dict]:
    """Like _shot_relevant_characters but never leaves a CHARACTER beat empty.

    Empty/half-populated character shots are a top defect (the storyboard seeds
    the video, so a missing character there is missing for the whole shot, and
    any character the LLM then invents drifts off-model). Fallback:
      - closeup/insert with nobody named → the single speaking subject (or first
        cast member) so the frame is not crowded;
      - medium/wide with nobody named → the full cast so the scene is populated.
    """
    relevant = _shot_relevant_characters(script, shot)
    if relevant:
        return relevant
    cast = [c for c in script.get("characters", [])
            if isinstance(c, dict) and (c.get("name") or "").strip()
            and (c.get("locked_visual_token") or c.get("appearance"))]
    if not cast:
        return []
    stype = (shot.get("shot_type") or "").strip().lower()
    if stype in ("closeup", "insert"):
        spk = (shot.get("speaker") or "").strip().lower()
        for c in cast:
            if (c.get("name") or "").strip().lower() == spk:
                return [c]
        return cast[:1]
    return cast


def _light_direction(script: dict) -> str:
    """One fixed light direction for the whole film so lighting never flips
    between shots. Persisted on the script when available; defaults otherwise."""
    return (script.get("_light_direction") or "the upper right").strip()


def _speaking_character(shot: dict) -> str:
    """The character who SPEAKS this shot's line (empty for narrator voiceover).
    The image/motion LLM frames and animates this character so the viewer sees
    who is talking and the right mouth is available for lip-sync."""
    spk = (shot.get("speaker") or "").strip()
    return "" if (not spk or spk.lower() == "narrator") else spk


def _pronoun(c: dict) -> str:
    """Stable he/him or she/her for a character. Prefers an explicit `gender`
    field; falls back to the assigned Kokoro voice prefix (af_/bf_ = female,
    am_/bm_ = male) so the spoken voice and the written pronoun always agree."""
    g = (c.get("gender") or "").strip().lower()
    if g not in ("male", "female"):
        v = (c.get("voice") or "").strip().lower()[:2]
        if v in ("af", "bf"):
            g = "female"
        elif v in ("am", "bm"):
            g = "male"
    return {"male": "he/him", "female": "she/her"}.get(g, "")


def _char_block(c: dict) -> str:
    """One reference line per character for BOTH the image and motion LLM:
    'Name: <locked appearance> [Name is he/him]'. The appearance carries the
    species (so the model can never swap a rabbit's ears for a tortoise's shell),
    and the bracketed pronoun is an instruction the model must obey but never
    copy verbatim into the prose."""
    name = (c.get("name") or "").strip()
    lock = (c.get("locked_visual_token") or c.get("appearance") or "").strip()
    pron = _pronoun(c)
    line = f"- {name}: {lock}"
    if pron:
        line += f"  [{name} is {pron}]"
    return line


# ==============================================================================
# PUBLIC API
# ==============================================================================

def assemble_image_prompt(
    script: dict,
    shot: dict,
    style_suffix: str = "",
    frame_type: str = "first",
    character_medium: str = "",
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
    relevant_chars = [] if no_chars else _resolve_shot_characters(script, shot)
    char_lines = []
    for c in relevant_chars:
        name = (c.get("name") or "").strip()
        lock = (c.get("locked_visual_token") or c.get("appearance") or "").strip()
        if name and lock:
            char_lines.append(_char_block(c))

    user_payload = {
        "beat": beat,
        "shot_type": (shot.get("shot_type") or "").strip().lower(),
        "frame_type": frame_type,
        "style_suffix": style_suffix or "",
        "light_direction": _light_direction(script),
        "setting": setting_clean,
        "tailored_frame_body": body,
        "character_pose": (shot.get("character_pose") or "").strip(),
        "camera_angle": (shot.get("camera_angle") or "").strip(),
        "narration": (shot.get("narration") or "").strip(),
        "speaking_character": _speaking_character(shot),
        "character_medium": (character_medium or "").strip(),
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
        f"verbatim, binding each name to its listed species. If `character_medium` "
        f"is non-empty, render EVERY character in that medium (e.g. a stick figure / "
        f"clay model / felt puppet) — keep their identity colors and key features "
        f"for consistency, but recast the form into the medium. End with camera "
        f"framing + lighting that comes from `light_direction` verbatim. "
        f"60-110 words."
    )

    raw = _call_llm(user_prompt, _build_system_prompt(
        beat, no_chars, shot_type=(shot.get("shot_type") or "")))
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


def assemble_image_prompt_mechanical(
    script: dict,
    shot: dict,
    style_suffix: str = "",
    frame_type: str = "first",
    character_medium: str = "",
) -> str:
    """Deterministic Z-Image prompt — NO LLM. Used when the LLM assembler fails
    (Ollama hiccup / VRAM), so a prompt ALWAYS carries the style anchor + every
    relevant character's locked appearance + shot framing + lighting. Keeps the
    look on-style and characters consistent even on the fallback path."""
    beat = (shot.get("beat") or "").strip().lower() or "hook"
    no_chars = beat in NO_CHARACTER_BEATS
    body = (shot.get(f"{frame_type}_frame_prompt")
            or shot.get("first_frame_prompt")
            or shot.get("visual_description") or "").strip()

    parts: list[str] = []
    if style_suffix:
        parts.append(style_suffix.strip().rstrip("."))
    if body:
        parts.append(body.rstrip("."))
    if not no_chars:
        med = (character_medium or "").strip().rstrip(".")
        for c in _resolve_shot_characters(script, shot):
            name = (c.get("name") or "").strip()
            lock = (c.get("locked_visual_token") or c.get("appearance") or "").strip()
            if name and lock:
                # medium FIRST so the diffusion model leads with the style
                # (e.g. "drawn as a simple stick figure, Whiskers is a cat ...")
                line = f"{name} is {lock}"
                parts.append(f"drawn {med}, {line}" if med else line)
    st = (shot.get("shot_type") or "").strip().lower()
    if st in SHOT_TYPE_FRAMING:
        parts.append(SHOT_TYPE_FRAMING[st])
    light = _light_direction(script)
    if light:
        parts.append(light.rstrip("."))
    return ". ".join(p for p in parts if p) + "."


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
        "Never write speech. "
        "Each character is listed below as '- Name: appearance  [Name is he/him]'. "
        "Bind each name to its appearance (which states the species) and pronoun "
        "EXACTLY — NEVER call one character by the other's name or species, use ONLY "
        "each character's given pronoun and never switch it, and never give a "
        "character a body part that belongs to the other (no shell on a rabbit, no "
        "long ears on a tortoise). The [square-bracket] note is an instruction — "
        "never copy it into your prose. Only animate "
        "characters listed below; do not add or reference a character who is not in "
        "the starting frame, and never write that a character is implied, off-screen, "
        "unseen or 'not present'. "
        "If the inputs name a `speaking_character`, the ONE main action belongs to "
        "THAT character (a look toward the listener, a lean, a small gesture as they "
        "speak); do not make another character the main mover. Keep 'No mouth "
        "movement' — the talking mouth is added later by lip-sync."
    )

    return f"""You output ONE paragraph of plain text — the motion prompt for Wan 2.2 14B image-to-video. No JSON. No markdown. No labels. No preamble. Just the motion paragraph.

You are a senior animation director. You describe how the starting frame should ANIMATE over ~3-5 seconds. The starting frame is fixed; you describe the motion that emerges from it.

# THIS SHOT'S BEAT: "{beat}"

# CHARACTER RULE

{char_rule}

# WAN 2.2 I2V PROMPT PHILOSOPHY

Wan 2.2 reads physical action verbs and real camera vocabulary. It was trained on detailed cinematic captions, so it DOES respond to lighting and atmosphere CHANGES (light shifting warmer, a glint moving, dust catching the sun) and to named camera moves. Lead with the largest motion in the frame.

GOOD: "The snail slowly inches forward along the leaf. Warm light shifts across its shell as a breeze rustles the marigolds behind it. The camera dollies in slowly."

BAD: "A beautiful sense of wonder unfolds as the brave little snail experiences the joy of discovery in this magical garden moment."

# COMPOSITION RULES — SIMPLE IS BETTER

A motion prompt is ONE main action plus ONE camera move. Wan stalls or
hallucinates when given several competing actions — keep it minimal.

1. **First sentence: the ONE main action.** Concrete verb, one subject. "Max drops a pebble into the bucket."
2. **Optional second sentence: one small secondary motion** — a moving object (grass sways) OR a lighting/atmosphere change (light warms, a glint slides across, dust drifts in a sunbeam). Wan 2.2 renders these well. Skip it if the main action is enough.
3. **Last sentence: the ONE camera move.** Use Wan's real camera vocabulary, explicit and short: "slow dolly in", "static hold", "gentle pan left", "slow tracking shot following the subject", "subtle crane up", "slow tilt down". Match the framing: insert/closeup shots want "static hold" or "slow dolly in"; wide shots want a slow pan, tracking shot, or crane.
4. **Length: 25-50 words.** Tight. Every word maps to pixels moving.
5. **No abstract mood words.** Banned: gracefully, beautifully, magically, dreamlike, ethereal, captures, evokes. (A concrete lighting CHANGE is fine — "light warms" is allowed; "a magical glow" is not.)
6. **No speech, no mouth movement, no lip sync.** Add: "No speech. No mouth movement." at end.
7. **No mode change.** Don't introduce new characters, new locations, or scene cuts. Wan animates the existing frame.
8. **The action comes from the frame + narration, NOT from the beat name.** Animate only what the starting-frame description and the shot's narration actually show. NEVER invent an action the narration doesn't state — no lifting, pulling, fighting, pushing rocks, or helping just because the beat is called "struggle"/"choice"/etc. A "struggle" beat is often only tired or steady effort (trudging on, panting, slowing down). If the narration says a character walks steadily, the motion is steady walking — nothing more.

# OUTPUT

ONE paragraph, plain text, 25-50 words. One action, one camera move. End with the camera move and the no-speech disclaimer."""


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

    relevant_chars = [] if no_chars else _resolve_shot_characters(script, shot)
    # Full reference blocks (name + species-bearing appearance + pronoun) so the
    # motion LLM can bind each name to its species and never swap a rabbit's ears
    # for a tortoise's shell. `type` alone ("animal") is not enough.
    char_labels = [_char_block(c) for c in relevant_chars if (c.get("name") or "").strip()]

    user_payload = {
        "beat": beat,
        "shot_type": (shot.get("shot_type") or "").strip().lower(),
        "starting_frame_description": approved_image_prompt or shot.get("first_frame_prompt", ""),
        "intended_motion_seed": motion_seed,
        "camera_angle": (shot.get("camera_angle") or "").strip(),
        "character_pose": (shot.get("character_pose") or "").strip(),
        "speaking_character": _speaking_character(shot),
        "characters_in_frame": char_labels if char_labels else (
            "NONE — breathing shot, environment motion only." if no_chars
            else "NONE — no character named in this shot."
        ),
    }

    user_prompt = (
        f"Write the Wan 2.2 14B I2V motion prompt for this shot. Inputs:\n\n"
        f"```\n{json.dumps(user_payload, indent=2, ensure_ascii=False)}\n```\n\n"
        f"Output ONE paragraph, 25-50 words, per the system rules: ONE main "
        f"action, optionally one small secondary motion, then ONE camera move. "
        f"End with 'No speech. No mouth movement.'"
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
