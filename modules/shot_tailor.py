"""
Claw Bot — Shot Tailor

Third-pass LLM (Qwen 2.5 14B) that runs AFTER script approval and AFTER
prompt_polisher. Reads each shot's `beat`, `narration`, and `visual_description`
and rewrites first_frame_prompt / last_frame_prompt / motion_prompt to MATCH
the beat type exactly — nothing more, nothing less.

Why this exists:
  Polisher fixes Z-Image-friendly prose but can drift away from the script's
  intent. A "reaction" beat may end up with full-body action; an "atmosphere"
  beat may sneak a character action in. Shot Tailor enforces beat discipline,
  explicit camera angle, and explicit character pose — while keeping every
  character's locked_visual_token / clothing anchors intact.

Pipeline order:
  script_generator  ->  prompt_polisher  ->  shot_tailor  ->  storyboard_generator
"""

import json
import logging
import re
import time as _t
from datetime import datetime
from pathlib import Path

import requests

# Reuse polisher helpers — no duplication of clothing-anchor logic
from modules.prompt_polisher import (
    _sanitize_characters,
    _enforce_appearance_in_prompt,
)

log = logging.getLogger("claw_bot.shot_tailor")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
OUTPUTS_DIR = PROJECT_ROOT / "04_Outputs" / "scripts"

QWEN_MODEL = "qwen2.5:14b-instruct-q6_K"
OLLAMA_URL = "http://127.0.0.1:11434"


# ==============================================================================
# BEAT TEMPLATES — strict per-beat rules
# Each entry: what the frames CAN show, what they MUST NOT show, camera, motion.
# ==============================================================================

BEAT_TEMPLATES = {
    "atmosphere": {
        "intent": "Pure breathing shot. Establish the world. ZERO characters.",
        "frame_focus": (
            "Wide environment shot. Setting elements ONLY — sky, ground, trees, "
            "props, weather, light. NO characters of any kind. NO people. NO "
            "animals. NO silhouettes. NO distant figures. The frame is empty "
            "of any living subject."
        ),
        "camera": "wide establishing shot OR high-angle drone shot",
        "motion": (
            "Camera-only motion. Slow drift, slow pan, slow push-in. "
            "Ambient world motion (leaves rustle, light shifts, dust drifts) "
            "is allowed. NO character motion — there are no characters in frame."
        ),
        "first_vs_last": (
            "Same empty setting; light direction, camera position, or weather "
            "shifts subtly between first and last frame."
        ),
        "forbidden": (
            "NO character names. NO character body parts. NO character clothing. "
            "NO character poses. NO close-ups. NO expressions. NO action verbs "
            "tied to any creature. The character[] array exists for the story "
            "but DOES NOT APPEAR in this shot."
        ),
    },
    "reaction": {
        "intent": "Hold on the character's face right after something happened.",
        "frame_focus": (
            "Extreme close-up (ECU) on the character's face. Face fills 60-80% "
            "of frame. Background blurred or simplified. Show only the visible "
            "micro-expression — eyes widening, brow lifting, mouth parting."
        ),
        "camera": "extreme close-up, eye-level, slight 3/4 angle",
        "motion": (
            "Tiny facial change between first and last frame (eyes widen, breath "
            "in). Camera: static hold OR slow push-in 5%. No body motion."
        ),
        "first_vs_last": (
            "First frame: neutral/just-before expression. Last frame: the felt "
            "reaction landed (wider eyes, tilted head, soft inhale)."
        ),
        "forbidden": "no walking, no running, no full body, no wide shots, no setting detail.",
    },
    "observation": {
        "intent": "Character watches something happen to others. Quiet.",
        "frame_focus": (
            "Medium or over-the-shoulder (OTS) shot. Character in foreground "
            "(facing away or 3/4), the watched subject visible in the background. "
            "Character's body is still — only the gaze acts."
        ),
        "camera": "OTS or medium shot, eye-level",
        "motion": (
            "Character head turns slowly OR stays static. The watched subject "
            "moves. Camera: static hold or very slow dolly."
        ),
        "first_vs_last": (
            "First frame: character begins watching. Last frame: same posture, "
            "watched subject has progressed in their action."
        ),
        "forbidden": "no character speaking, no character moving toward subject, no action verbs on main character.",
    },
    "moment-of-decision": {
        "intent": "Held beat BEFORE the choice. Tension is the point.",
        "frame_focus": (
            "Tight medium-close-up. Character is still: head bowed, hands "
            "clenched, breath held, looking at the object/path of the choice. "
            "Single light source. No background activity."
        ),
        "camera": "medium close-up, low or eye-level, slight Dutch angle allowed",
        "motion": (
            "Almost no motion. Shoulders rise once with a breath. Camera: very "
            "slow push-in ~3% across the shot. No decision yet — hold the pause."
        ),
        "first_vs_last": (
            "First frame: character looking down/away, gathering. Last frame: "
            "gaze just beginning to lift toward the object of the choice. "
            "The decision is not made yet."
        ),
        "forbidden": "no acting on the choice, no walking, no smiling, no resolution.",
    },
    "hook": {
        "intent": "Show the character wanting or feeling something specific.",
        "frame_focus": (
            "Medium shot. Character mid-action, body posture conveying the want "
            "(hurrying, reaching, peering). Setting visible but secondary."
        ),
        "camera": "medium shot, eye-level OR slight low-angle",
        "motion": "Character begins the want-action. Camera: gentle follow or static.",
        "first_vs_last": (
            "First frame: character about to act. Last frame: character "
            "committed to the action."
        ),
        "forbidden": "no narrator-style abstractions, no static portrait.",
    },
    "spark": {
        "intent": "The unexpected thing happens.",
        "frame_focus": (
            "Action-driven medium or wide shot. The disruption is visible "
            "(object appearing, obstacle blocking, stranger entering). "
            "Character reacts physically."
        ),
        "camera": "medium or wide, slight low-angle to dramatize",
        "motion": (
            "The disruption enters frame; character body responds (steps back, "
            "raises arm, turns head). Camera: snap zoom or static catch."
        ),
        "first_vs_last": "First: just before disruption. Last: disruption present, character reacting.",
        "forbidden": "no calm staging, no symmetrical composition.",
    },
    "struggle": {
        "intent": "Character tries something or hesitates.",
        "frame_focus": (
            "Medium shot showing the effort. Hands, posture, strained "
            "expression. The obstacle is in frame."
        ),
        "camera": "medium shot, slight handheld feel acceptable",
        "motion": "Character attempts the action — pushes, reaches, leans. Camera: gentle handheld or static.",
        "first_vs_last": "First: attempt begins. Last: mid-attempt, outcome unresolved.",
        "forbidden": "no resolution, no success or failure shown yet.",
    },
    "choice": {
        "intent": "Character makes the visible decision.",
        "frame_focus": (
            "Medium close-up on the decisive action. Hands or body crossing "
            "the threshold of the choice (handing over the cookie, stepping "
            "around the turtle)."
        ),
        "camera": "medium close-up, eye-level, clean composition",
        "motion": (
            "Single committed action — hand extends, foot steps, head nods. "
            "Camera: slow push-in to emphasize."
        ),
        "first_vs_last": "First: about to act. Last: action completed.",
        "forbidden": "no hesitation, no reverse motion.",
    },
    "consequence": {
        "intent": "The world responds. Visible payoff.",
        "frame_focus": (
            "Wider shot showing the result — a smile, a transformation, a "
            "reward, a small wonder. Both the character and the response are "
            "visible."
        ),
        "camera": "medium or wide, eye-level, warm framing",
        "motion": (
            "The response unfolds (the seed sprouts, the friend smiles back, "
            "the door opens). Camera: gentle pull-back or static."
        ),
        "first_vs_last": "First: moment of response. Last: response fully landed.",
        "forbidden": "no new conflict, no ambiguity.",
    },
}

DEFAULT_BEAT = "hook"

# Beats that must render WITHOUT any character in frame.
NO_CHARACTER_BEATS = {"atmosphere"}


def _template_for(beat: str) -> dict:
    if not beat:
        return BEAT_TEMPLATES[DEFAULT_BEAT]
    return BEAT_TEMPLATES.get(beat.strip().lower(), BEAT_TEMPLATES[DEFAULT_BEAT])


def _scrub_characters_from_prompt(prompt: str, character_names: list[str]) -> str:
    """Remove any sentence that names a character. Used for breathing beats
    where the frame must be empty of characters. Last-line defence — if Qwen
    ignores the no-character rule, we strip the offending sentences here."""
    if not prompt or not character_names:
        return prompt
    sentences = re.split(r'(?<=[.!?])\s+', prompt.strip())
    kept = []
    dropped = []
    name_pattern = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in character_names if n) + r")\b",
        re.IGNORECASE,
    )
    for s in sentences:
        if name_pattern.search(s):
            dropped.append(s)
        else:
            kept.append(s)
    if dropped:
        log.warning(
            f"Atmosphere scrub: removed {len(dropped)} sentence(s) naming characters: "
            f"{dropped}"
        )
    result = " ".join(kept).strip()
    return result if result else prompt  # if we stripped everything, keep original


def _has_character_words(prompt: str, character_names: list[str]) -> bool:
    if not prompt or not character_names:
        return False
    name_pattern = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in character_names if n) + r")\b",
        re.IGNORECASE,
    )
    return bool(name_pattern.search(prompt))


# ==============================================================================
# SYSTEM PROMPT
# ==============================================================================

def _build_system_prompt(beat: str, template: dict) -> str:
    return f"""You output ONLY valid JSON. No prose, no preamble, no markdown fences.

You are a shot-tailor — a senior cinematographer rewriting THREE prompt fields for ONE shot in a kids' storybook video. Your job is to make the prompts match the shot's BEAT TYPE exactly. Nothing more, nothing less.

# THIS SHOT'S BEAT: "{beat}"

INTENT: {template['intent']}
FRAME FOCUS: {template['frame_focus']}
CAMERA: {template['camera']}
MOTION: {template['motion']}
FIRST vs LAST FRAME: {template['first_vs_last']}
FORBIDDEN: {template['forbidden']}

# OUTPUT — three fields, JSON only

{{
  "camera_angle": "<one short phrase, e.g. 'extreme close-up, eye-level, 3/4 face'>",
  "character_pose": "<one short phrase per main character in this shot, e.g. 'Rohan: kneeling, hands open, head tilted down'>",
  "first_frame_prompt": "<60-100 words. Match the BEAT exactly. Front-load main subject. Concrete present-tense verbs. Include every character's clothing colors and locked appearance verbatim from the context. State the camera angle and the character pose explicitly. NO abstract mood words.>",
  "last_frame_prompt": "<60-100 words. SAME beat rules. Same character appearance verbatim. Show the small visible change defined in FIRST vs LAST FRAME above. Same camera angle unless the template says otherwise.>",
  "motion_prompt": "<30-60 words. Match the BEAT's MOTION rule exactly. ONLY physical motion of bodies + camera. Banned verbs: says, speaks, talks, mouth, lips, voice, thinks, realizes, feels, decides. No emotion adverbs. Respect species anatomy (snails glide, frogs hop, birds flap — never humanoid verbs for animals).>"
}}

# HARD RULES — VIOLATING ANY MEANS REJECTION

1. The beat type "{beat}" rules ABOVE override everything. If FORBIDDEN says "no walking", do not write walking. If FRAME FOCUS says ECU on face, do not write wide shot.

2. Every character in the shot MUST appear with their FULL appearance verbatim from the context's `characters[].locked_visual_token`. Drop one clothing item and the diffusion model rolls a random color.

3. Mention each character's appearance ONCE per prompt — combine appearance + action in a single sentence per character. Never duplicate.

4. No character names without their appearance. Either both or neither (the storyboard renderer injects the locked sheet — but you must still describe the pose tied to that character).

5. Banned abstract words anywhere: serene, joyful, wonder, vibrant, lush, magical, whimsical, cinematic, expressive, captures, evokes, dreamlike, ethereal.

6. Camera framing as concrete spatial info, not film-school jargon. "the boy fills the lower-right of the frame, viewed from slightly above" not "cinematic close-up".

7. Lighting as direction + color. "warm yellow light from the upper right" not "soft golden glow".

8. No style suffix — style is appended later.

9. No character speech, no thought-bubbles, no internal-state description.

10. Anatomy lock for non-human characters:
    - Snail/slug: glides on underside, shell attached, antennae sway. Never stands/walks/runs.
    - Frog: low crouch, hops with both hind legs, lands on all four. Never stands on two legs.
    - Bird: body horizontal, wings flap symmetrically, hops with feet together on ground.
    - Fish: undulates side-to-side, fins steer, no limbs.
    - Insect/butterfly: wings flutter rapidly, body horizontal, legs grip.
    - Quadruped: moves on all four legs, head lifts independently.
    First verb of motion_prompt MUST be the species-natural locomotion verb.

Output ONLY the JSON object with exactly the five keys above. Start with {{ end with }}."""


# ==============================================================================
# OLLAMA CALL
# ==============================================================================

def _call_qwen(user_prompt: str, system_prompt: str) -> str:
    combined = f"SYSTEM INSTRUCTIONS (STRICTLY FOLLOW):\n{system_prompt}\n\nUSER REQUEST:\n{user_prompt}"
    payload = {
        "model": QWEN_MODEL,
        "prompt": combined,
        "stream": False,
        "options": {
            "temperature": 0.35,
            "top_p": 0.9,
            "num_ctx": 16384,
            "num_predict": 4096,
        },
    }
    log.info(f"Calling Qwen ({QWEN_MODEL}) for shot tailoring...")
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=900)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def _unload_model(model_name: str) -> None:
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model_name, "prompt": "", "keep_alive": 0},
            timeout=10,
        )
        log.info(f"Unloaded {model_name} from VRAM")
    except Exception as e:
        log.warning(f"Could not unload {model_name}: {e}")


# ==============================================================================
# JSON EXTRACTION
# ==============================================================================

def _extract_json(raw: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not extract JSON from Qwen response: {raw[:300]}")


# ==============================================================================
# PUBLIC API
# ==============================================================================

def tailor_script(script: dict, unload_qwen_after: bool = True) -> dict:
    """
    For every shot, run Qwen with a beat-specific system prompt that forces
    camera angle, character pose, and beat-discipline. Overwrites the three
    prompt fields in place. Stores camera_angle and character_pose on the
    shot for later inspection. Persists to the script JSON file.
    """
    _start = _t.time()
    tailored = json.loads(json.dumps(script))  # deep copy

    cleaned_chars = _sanitize_characters(tailored.get("characters", []))
    context = {
        "title": tailored.get("title"),
        "style": tailored.get("style"),
        "culture": tailored.get("culture"),
        "setting": tailored.get("setting"),
        "characters": cleaned_chars,
    }

    shots = tailored.get("shots", [])
    total = len(shots)
    log.info(f"Shot Tailor: processing {total} shots one-by-one...")

    character_names = [
        c.get("name", "").strip()
        for c in cleaned_chars
        if isinstance(c, dict) and c.get("name", "").strip()
    ]

    for idx, shot in enumerate(shots, start=1):
        n = shot.get("shot_number", idx)
        beat = (shot.get("beat") or DEFAULT_BEAT).strip().lower()
        template = _template_for(beat)
        is_breathing = beat in NO_CHARACTER_BEATS

        user_prompt = (
            f"STORY CONTEXT (consistency anchors — do not change):\n"
            f"```json\n{json.dumps(context, indent=2, ensure_ascii=False)}\n```\n\n"
            f"SHOT TO TAILOR (shot {n} of {total}, beat='{beat}'):\n"
            f"```json\n{json.dumps(shot, indent=2, ensure_ascii=False)}\n```\n\n"
            f"Rewrite the three prompt fields per the beat rules. Output a JSON "
            f"object with exactly: camera_angle, character_pose, first_frame_prompt, "
            f"last_frame_prompt, motion_prompt."
        )
        if is_breathing:
            user_prompt += (
                f"\n\nCRITICAL: beat='{beat}' is a BREATHING SHOT. The frame "
                f"is EMPTY of any character. DO NOT name any character. DO NOT "
                f"describe any body, clothing, pose, or expression. Set "
                f"character_pose to \"no characters in frame\". Setting + camera "
                f"+ light only."
            )

        log.info(f"[{idx}/{total}] Tailoring shot {n} (beat={beat}, breathing={is_breathing})...")
        try:
            raw = _call_qwen(user_prompt, _build_system_prompt(beat, template))
            tailored_fields = _extract_json(raw)

            # Store metadata fields (extra — never overwrite if missing)
            for meta_field in ("camera_angle", "character_pose"):
                v = tailored_fields.get(meta_field)
                if isinstance(v, str) and v.strip():
                    shot[meta_field] = v.strip()

            # Overwrite the three render fields
            for field in ("first_frame_prompt", "last_frame_prompt", "motion_prompt"):
                v = tailored_fields.get(field)
                if isinstance(v, str) and v.strip():
                    shot[field] = v.strip()
                else:
                    log.warning(f"Shot {n}: '{field}' missing/empty — keeping previous value")

            if is_breathing:
                # Hard scrub: strip any sentence that names a character.
                # Force character_pose marker so storyboard char-sheet skips this shot.
                for field in ("first_frame_prompt", "last_frame_prompt", "motion_prompt"):
                    if field in shot and isinstance(shot[field], str):
                        before = shot[field]
                        shot[field] = _scrub_characters_from_prompt(
                            shot[field], character_names
                        )
                        if shot[field] != before:
                            log.warning(
                                f"Shot {n} ({beat}): Qwen ignored no-character rule; "
                                f"scrubbed character mentions."
                            )
                shot["character_pose"] = "no characters in frame"
            else:
                # Mechanical clothing-anchor guard — only for character beats
                for char in cleaned_chars:
                    if not isinstance(char, dict):
                        continue
                    name = char.get("name", "")
                    appr = char.get("appearance", "")
                    if not appr:
                        continue
                    for field in ("first_frame_prompt", "last_frame_prompt"):
                        if field in shot and isinstance(shot[field], str):
                            shot[field] = _enforce_appearance_in_prompt(
                                shot[field], appr, character_name=name
                            )
        except Exception as e:
            log.error(f"Shot {n} tailor failed: {e} — keeping previous prompts")

    tailored["_tailored"] = True
    tailored["_tailored_at"] = datetime.now().isoformat()
    tailored["_tailor_model"] = QWEN_MODEL

    script_id = tailored.get("_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    tailored_path = OUTPUTS_DIR / f"script_{script_id}_tailored.json"
    tailored_path.write_text(json.dumps(tailored, indent=2, ensure_ascii=False), encoding="utf-8")

    main_path = OUTPUTS_DIR / f"script_{script_id}.json"
    main_path.write_text(json.dumps(tailored, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(f"Tailored script saved: {tailored_path} (and replaced {main_path}) ({_t.time() - _start:.1f}s)")

    if unload_qwen_after:
        _unload_model(QWEN_MODEL)
    return tailored


def format_tailor_summary_for_discord(tailored: dict) -> str:
    lines = [
        f"🎯 **Shots tailored by Qwen 2.5 14B (beat-aware)**",
        f"Script: `{tailored.get('_id', '?')}`",
        "",
    ]
    for shot in tailored.get("shots", []):
        n = shot.get("shot_number", "?")
        beat = shot.get("beat", "?")
        cam = shot.get("camera_angle", "?")
        pose = shot.get("character_pose", "?")
        lines.append(f"**Shot {n}** [`{beat}`] — cam: _{cam}_ · pose: _{pose[:80]}_")
    lines.append("")
    lines.append("React ✅ to start storyboard generation, or `!retailor <id>` to redo.")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import sys
    if len(sys.argv) < 2:
        print("Usage: python shot_tailor.py <path_to_script.json>")
        sys.exit(1)
    src = Path(sys.argv[1])
    original = json.loads(src.read_text(encoding="utf-8"))
    result = tailor_script(original)
    print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])
