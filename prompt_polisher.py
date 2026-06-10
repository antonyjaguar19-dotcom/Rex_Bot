"""
Claw Bot — Prompt Polisher

Second-pass LLM (Qwen 2.5 14B) that reads an approved script and rewrites
the visual + motion prompts for every shot. Story logic stays untouched.

Llama writes the story. Qwen polishes the visuals.
"""

import json
import logging
import re
import time as _t
from datetime import datetime
from pathlib import Path

import requests

log = logging.getLogger("claw_bot.prompt_polisher")


# ==============================================================================
# APPEARANCE SANITIZER — strips pose/location leaks from character descriptions
# ==============================================================================

# Phrases that indicate pose/location/action contamination in character.appearance
_POSE_LEAK_PATTERNS = [
    r",\s*(perched|sitting|sat|standing|stood|walking|walked|running|ran|"
    r"lying|laying|resting|crouching|crouched|kneeling|knelt|leaning|leaned|"
    r"hanging|hung|flying|flew|swimming|swam|holding|held|carrying|carried|"
    r"climbing|climbed|jumping|jumped|hopping|hopped|reaching|reached)\s+[^.]*",
    r",\s*(in|on|at|near|beside|next to|under|behind|in front of)\s+[^.]+",
    r"\s+(perched|sitting|standing|lying)\s+[^.]+\.?$",
]


def _clean_appearance(text: str) -> str:
    """Remove pose/location leaks from a character.appearance string.

    Examples:
      'A black crow with sharp eyes, perched on a tree branch.'
        -> 'A black crow with sharp eyes.'
      'A small red tomato, in a garden, with a determined look.'
        -> 'A small red tomato with a determined look.'
    """
    if not text:
        return text
    cleaned = text
    for pattern in _POSE_LEAK_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    # Collapse double spaces, fix trailing punctuation
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s*,\s*([.!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s*,\s*$", "", cleaned)
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _sanitize_characters(characters: list) -> list:
    """Apply _clean_appearance to every character's appearance field."""
    out = []
    for c in characters or []:
        if isinstance(c, dict):
            cc = dict(c)
            cc["appearance"] = _clean_appearance(cc.get("appearance", ""))
            out.append(cc)
        else:
            out.append(c)
    return out


# Color words that often anchor clothing — used by verifier below
_COLOR_KEYWORDS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "grey", "gray", "violet", "indigo", "turquoise",
    "cyan", "magenta", "maroon", "navy", "beige", "tan", "gold", "silver",
    "crimson", "scarlet", "olive", "teal",
}

# Clothing nouns we look for as anchors
_CLOTHING_KEYWORDS = {
    "shirt", "t-shirt", "tshirt", "blouse", "top", "dress", "skirt",
    "pants", "trousers", "shorts", "jeans", "leggings", "tights",
    "jacket", "coat", "vest", "sweater", "hoodie", "cardigan",
    "hat", "cap", "beanie", "scarf", "tie", "belt",
    "shoes", "boots", "sneakers", "sandals", "slippers",
    "outfit", "uniform", "robe", "cloak", "cape", "apron", "overalls",
    "fur", "feathers", "scales", "shell", "leaves",
}


def _extract_appearance_keywords(appearance: str) -> list[str]:
    """Pull out color+clothing pairs and other anchor phrases from appearance."""
    if not appearance:
        return []
    text = appearance.lower()
    anchors = []
    # Find color+clothing bigrams: "blue shorts", "red t-shirt", etc.
    words = re.findall(r"[a-z\-]+", text)
    for i, w in enumerate(words):
        if w in _COLOR_KEYWORDS and i + 1 < len(words):
            nxt = words[i + 1]
            if nxt in _CLOTHING_KEYWORDS:
                anchors.append(f"{w} {nxt}")
        # Also catch "blue shirt" with "and a" or "and" between
        if w in _COLOR_KEYWORDS and i + 2 < len(words):
            nxt2 = words[i + 2]
            if nxt2 in _CLOTHING_KEYWORDS and words[i + 1] in ("a", "an"):
                anchors.append(f"{w} {nxt2}")
    return anchors


def _enforce_appearance_in_prompt(
    prompt: str,
    appearance: str,
    character_name: str = "",
) -> str:
    """If clothing anchors from `appearance` are missing from `prompt`, append them.

    Mechanical guard against Qwen dropping clothing details despite instructions.
    """
    if not prompt or not appearance:
        return prompt
    anchors = _extract_appearance_keywords(appearance)
    if not anchors:
        return prompt

    prompt_lower = prompt.lower()
    missing = [a for a in anchors if a not in prompt_lower]
    if not missing:
        return prompt  # all clothing anchors already present

    # Append a clarifying sentence with the missing clothing
    name_clause = f" {character_name}" if character_name else " the character"
    missing_str = " and ".join(missing)
    addition = f"{name_clause.capitalize().strip()} is wearing {missing_str}."
    log.info(
        f"Appearance guard: injected missing clothing into prompt "
        f"(character={character_name or '?'}, missing={missing})"
    )
    # Insert before the trailing "Visually:" suffix if present, else append
    if " Visually:" in prompt:
        return prompt.replace(" Visually:", f" {addition} Visually:", 1)
    return prompt.rstrip(". ") + ". " + addition

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
OUTPUTS_DIR = PROJECT_ROOT / "04_Outputs" / "scripts"

# Qwen config — pinned, separate from Llama's model_registry entry
QWEN_MODEL = "qwen2.5:14b-instruct-q6_K"
OLLAMA_URL = "http://127.0.0.1:11434"


# ==============================================================================
# SYSTEM PROMPT — Qwen's polishing instructions
# ==============================================================================

def _build_system_prompt() -> str:
    return """You output ONLY valid JSON. No prose, no preamble, no markdown fences.

You are a senior storyboard prompt engineer writing prompts for Z-Image Turbo, a paragraph-style diffusion model. Your prompts must be CONCRETE, VISUAL, and PRESENT-TENSE.

# YOUR TASK

You receive a complete story script. For EACH shot, rewrite three fields:
- first_frame_prompt
- last_frame_prompt
- motion_prompt

You do NOT change: title, theme, culture, style, characters, setting, narration, visual_description, shot_number, moral, or any other field.

# Z-IMAGE PROMPT PHILOSOPHY (READ THIS — IT IS DIFFERENT FROM SDXL)

Z-Image is paragraph-driven, not tag-driven. It renders best from prose that reads like a museum description of a painting that ALREADY EXISTS. Concrete nouns + present-tense verbs + spatial relationships. Z-Image IGNORES vague atmospheric words and may DROP entities buried under adjectives.

GOOD: "A small green snail with a brown-and-white striped shell rests on a wide green leaf. The leaf bends slightly under its weight. Three orange marigold flowers grow behind it. Morning light comes from the upper left."

BAD (will lose entities): "A serene moment of curiosity unfolds as the brave little snail experiences wonder in a vibrant garden filled with the gentle dance of life and the soft glow of dawn."

# RULES FOR REWRITING

1. **Front-load the most important entities.** First sentence = main subject + key attribute. Second sentence = secondary entities. Third = setting. Fourth = light/camera. Z-Image weights early tokens more.

2. **EVERY entity from narration and visual_description must appear, in concrete form.** If narration says "spots a butterfly" — there must be a butterfly in the prompt. If narration says "blocks the path" — the obstacle must be physically described. Do NOT drop entities while paraphrasing.

2a. **GROUND every background character or activity with a CONCRETE physical anchor.** Z-Image will render "kids playing" as kids floating in mid-air unless you specify where they are and what they are touching. ALWAYS write background activity with anchored verbs:
- BAD: "other kids are playing in the background"
- GOOD: "other children sit on swings hanging from a wooden frame, slide down a yellow slide, and run across the grass"
- BAD: "people laughing around him"
- GOOD: "a small group of children stand on the grass nearby, laughing"
- BAD: "kids enjoying the park"
- GOOD: "two kids sit cross-legged on a picnic blanket, one child kicks a soccer ball on the path"
Every background character must be physically supported by something visible: ground, grass, slide, swing seat, bench, blanket, path. Never describe ambient motion ("playing", "laughing", "enjoying") without anchoring it to a surface or object.

2b. **NO floating, hovering, or unsupported figures EVER** — unless the script explicitly involves flying (birds, paper planes, kites). If a character appears in air, they MUST be: (a) on a swing with chains visible, (b) mid-jump with feet near a surface, (c) holding onto an object, or (d) explicitly flying. Never leave background figures suspended.

2c. **SUPPORT STRUCTURES must be explicitly described, not implied.** Children's playground equipment is the most common source of broken physics in renders. When describing playground furniture, ALWAYS specify the supporting structure:
- "swing" → write "swing with chains hanging from a wooden A-frame crossbar" (not just "swing")
- "slide" → write "slide attached to a tall climbing frame with metal rails" (not just "slide")
- "seesaw" → write "seesaw with a central wooden pivot post" (not just "seesaw")
- "monkey bars" → write "monkey bars supported by two metal end-posts"
Every playground structure must have its FRAME, POSTS, or SUPPORT explicitly named in the prompt. Z-Image will omit invisible supports otherwise, leading to floating swings.

3. **Character consistency anchors — ZERO TOLERANCE for dropped details.** In EVERY shot a character appears in, you MUST include their COMPLETE appearance description from the `characters` array — every single descriptor: clothing colors, garment types, hair, body, accessories, distinguishing features. The script's `characters[].appearance` is the GROUND TRUTH. If the script says "blue shorts and a red t-shirt", every shot featuring that character MUST mention "blue shorts and red t-shirt" — not "the boy", not "the curly-haired boy", but the FULL clothing. Z-Image needs this to render consistently — drop one clothing detail and the model rolls a random color.

3a. **CRITICAL CHECK before outputting each shot:** Before you finalize a `first_frame_prompt`, read your own output and verify: does it explicitly state every clothing item, color, and feature from the character's appearance description? If not, REWRITE the prompt to include them. This is non-negotiable.

3b. **CRITICAL: Mention each character's appearance ONCE per prompt — never twice.** Combine the character description AND the action into ONE sentence per character. Do NOT write "A red tomato. A red tomato lands on the pot." — that makes duplicate characters. Write: "A small red tomato with green leaves lands on the edge of the pot." One mention with full description, action attached.

3c. **Negative example (what NOT to do):**
Script says: `appearance: "small boy with curly brown hair, blue shorts and red t-shirt"`
BAD shot prompt: "A small boy with curly brown hair kneels beside the turtle." ← clothing dropped, Z-Image will pick random colors
GOOD shot prompt: "A small boy with curly brown hair, wearing blue shorts and a red t-shirt, kneels beside the turtle."

4. **First vs Last frame:** Same character appearance phrase, different POSE / POSITION / EXPRESSION / LOCATION-IN-FRAME. The frames must be visually distinct enough that interpolated motion between them looks natural.

5. **Use plain present-tense visual verbs.** Yes: stands, holds, leans, points, looks, reaches, sits, lies, faces, grips. No: "experiences", "feels", "embodies", "radiates".

6. **BANNED ABSTRACT WORDS** — Z-Image cannot render these and they waste tokens:
   - moods/emotions: "serene", "joyful", "wonder", "curiosity", "determination", "peaceful", "gentle"
   - vague qualifiers: "vibrant", "lush", "magical", "whimsical", "dreamlike", "ethereal"
   - meta-language: "cinematic", "expressive", "captures", "showcases", "evokes"
   Replace these with something concrete. "Curious snail" → "snail with head raised, antennae forward". "Vibrant garden" → "garden with red flowers, yellow flowers, and green leaves".

7. **Camera framing as concrete spatial info, not film-school terms.** Instead of "close-up shot" → "the snail fills most of the frame, viewed from slightly above". Instead of "wide shot" → "the boy stands small in the lower-left corner of a wide garden". Z-Image responds to spatial composition, not jargon.

8. **Lighting as direction + color, never "mood".** "Warm yellow light from the right" works. "Soft morning light" is too vague — specify direction and color.

9. **Frame prompts: 70-110 words.** Tighter than before. Every word must be doing visual work. Cut adjective stacks.

10. **Motion prompt rules — STRICT:**
    - Length: 40-70 words.
    - Describe ONLY physical motion of bodies + camera movement.
    - Be specific: "raises right hand slowly" not "moves arm".
    - Camera: "slow push in", "static hold", "slow pan left to right".
    - **FORBIDDEN VERBS — never use these:** says, speaks, talking, mouth opens, lips, voice, dialogue, smiles "as if", thinks, realizes, decides, understands, feels.
    - **FORBIDDEN VAGUE PHRASES:** "wobbles wildly", "moves with confidence", "looks determined", "shows emotion", "as if to say", "wonders", "appears to".
    - No emotion words ("happily", "sadly") — describe the visible action that conveys it.

11. **ANATOMY LOCK — non-human characters MUST move like their species. This is critical.**

    Diffusion video models default to humanoid bipedal motion when given vague instructions. You MUST prevent this by writing motion that respects the character's body.

    For each character, before writing motion, check `appearance` and `type` from the script's `characters` array. If type is "animal" or "creature", apply these locks:

    - **Snail / slug:** glides forward on its underside; the shell stays attached to the body; antennae sway slightly. NEVER write "stands", "walks", "runs", "lifts foot".
    - **Frog:** rests in a low crouch with hind legs folded; jumps by pushing off with both hind legs together; lands on all four limbs. NEVER write "stands up", "walks forward", "takes a step". A frog never stands on two legs.
    - **Bird:** body stays horizontal; wings flap symmetrically; head bobs separately from body; hops with feet together when on ground.
    - **Fish:** body undulates side-to-side; tail propels; fins steer; never has limbs.
    - **Insect / butterfly / bee:** wings flutter rapidly; body remains roughly horizontal; legs grip surfaces but do not "walk upright".
    - **Quadruped (dog, cat, mouse, rabbit, fox, etc.):** moves on all four legs; never "stands up to do something" unless explicitly part of the story; head lifts independently of body.

    For ANY non-human character, the FIRST verb in the motion_prompt must describe locomotion natural to that species ("crawls", "hops", "flies", "swims", "trots") — NOT a humanoid verb ("stands", "walks", "steps").

12. **Style suffix added later** — do NOT include style tags. Style is appended automatically.

13. **NO character names inside the prompt body** — names are not visible in pictures. Use descriptions: "the small green snail" not "Sally". The character array tracks names; the prompt describes appearance.
# OUTPUT FORMAT

Output the COMPLETE script JSON with the three fields rewritten in each shot. Keep all other fields byte-for-byte identical. Output ONLY the JSON object, starting with { and ending with }."""


# ==============================================================================
# OLLAMA CALL
# ==============================================================================

def _call_qwen(user_prompt: str, system_prompt: str) -> str:
    """Call Qwen via Ollama generate endpoint."""
    combined = f"SYSTEM INSTRUCTIONS (STRICTLY FOLLOW):\n{system_prompt}\n\nUSER REQUEST:\n{user_prompt}"

    payload = {
        "model": QWEN_MODEL,
        "prompt": combined,
        "stream": False,
        "options": {
            "temperature": 0.4,
            "top_p": 0.9,
            "num_ctx": 16384,
            "num_predict": 8192,
        },
    }

    log.info(f"Calling Qwen ({QWEN_MODEL}) for prompt polishing...")
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=900)
    r.raise_for_status()
    return r.json().get("response", "").strip()


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
# MERGE — keep Llama's structure, swap only the three prompt fields
# ==============================================================================

def _merge_polished(original: dict, polished: dict) -> dict:
    """
    Defensively merge: take Llama's script as the base of truth, only copy
    the three rewritten fields from Qwen's output. Protects against Qwen
    accidentally changing narration, shot_number, characters, etc.
    """
    merged = json.loads(json.dumps(original))  # deep copy

    polished_shots = {s.get("shot_number"): s for s in polished.get("shots", [])}

    for shot in merged.get("shots", []):
        n = shot.get("shot_number")
        if n in polished_shots:
            ps = polished_shots[n]
            for field in ("first_frame_prompt", "last_frame_prompt", "motion_prompt"):
                if field in ps and isinstance(ps[field], str) and ps[field].strip():
                    shot[field] = ps[field].strip()
                else:
                    log.warning(f"Shot {n}: Qwen did not return '{field}' — keeping Llama's version")
        else:
            log.warning(f"Shot {n}: not found in Qwen's output — keeping all Llama fields")

    return merged


# ==============================================================================
# UNLOAD HELPER — free Llama before loading Qwen
# ==============================================================================

def _unload_model(model_name: str) -> None:
    """Tell Ollama to evict a model from VRAM."""
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
# PUBLIC API
# ==============================================================================

def polish_script(script: dict, unload_llama: bool = True) -> dict:
    """
    Polish each shot's three prompt fields one at a time. More reliable than
    asking Qwen to rewrite the entire script in a single call.
    """
    _start = _t.time()

    if unload_llama:
        try:
            from modules import model_registry
            llm_cfg = model_registry.get_active("llm_backend")
            llama_name = (llm_cfg or {}).get("model_id") or "llama3.1:8b-instruct-q8_0"
            _unload_model(llama_name)
        except Exception as e:
            log.warning(f"Llama unload skipped: {e}")

    merged = json.loads(json.dumps(script))  # deep copy
    shots = merged.get("shots", [])
    total = len(shots)
    log.info(f"Polishing {total} shots one by one...")

    # Sanitize characters BEFORE Qwen sees them — strips pose/location leaks
    cleaned_chars = _sanitize_characters(merged.get("characters", []))

    # Compact context Qwen sees for every shot — keeps character/setting consistent
    context = {
        "title": merged.get("title"),
        "style": merged.get("style"),
        "culture": merged.get("culture"),
        "setting": merged.get("setting"),
        "characters": cleaned_chars,
    }

    for idx, shot in enumerate(shots, start=1):
        n = shot.get("shot_number", idx)
        user_prompt = (
            f"STORY CONTEXT (do not change, use for consistency):\n"
            f"```json\n{json.dumps(context, indent=2, ensure_ascii=False)}\n```\n\n"
            f"SHOT TO REWRITE (shot {n} of {total}):\n"
            f"```json\n{json.dumps(shot, indent=2, ensure_ascii=False)}\n```\n\n"
            f"Rewrite ONLY first_frame_prompt, last_frame_prompt, and motion_prompt "
            f"per the system rules. Output a JSON object with exactly these three keys, "
            f"nothing else."
        )

        log.info(f"[{idx}/{total}] Polishing shot {n}...")
        try:
            raw = _call_qwen(user_prompt, _build_system_prompt())
            polished_fields = _extract_json(raw)
            for field in ("first_frame_prompt", "last_frame_prompt", "motion_prompt"):
                v = polished_fields.get(field)
                if isinstance(v, str) and v.strip():
                    shot[field] = v.strip()
                else:
                    log.warning(f"Shot {n}: '{field}' missing/empty — keeping original")

            # Mechanical guard: ensure every character's clothing anchors made it into the prompt
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
            log.error(f"Shot {n} polish failed: {e} — keeping original")

    merged["_polished"] = True
    merged["_polished_at"] = datetime.now().isoformat()
    merged["_polish_model"] = QWEN_MODEL

    script_id = merged.get("_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save the polished version as audit copy
    polished_path = OUTPUTS_DIR / f"script_{script_id}_polished.json"
    polished_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    # ALSO overwrite the main script file so storyboard_workflow picks it up
    main_path = OUTPUTS_DIR / f"script_{script_id}.json"
    main_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(f"Polished script saved: {polished_path} (and replaced {main_path}) ({_t.time() - _start:.1f}s)")

    _unload_model(QWEN_MODEL)
    return merged


def format_polish_summary_for_discord(original: dict, polished: dict) -> str:
    """Short Discord message comparing what changed per shot."""
    lines = [
        f"✨ **Prompts polished by Qwen 2.5 14B**",
        f"Script: `{polished.get('_id', '?')}`",
        "",
    ]
    for shot in polished.get("shots", []):
        n = shot.get("shot_number", "?")
        ff = shot.get("first_frame_prompt", "")
        lines.append(f"**Shot {n}** — first frame: _{ff[:120]}{'...' if len(ff) > 120 else ''}_")
    lines.append("")
    lines.append("React ✅ to start storyboard generation, or `!repolish` to redo.")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import sys
    if len(sys.argv) < 2:
        print("Usage: python prompt_polisher.py <path_to_script.json>")
        sys.exit(1)
    src = Path(sys.argv[1])
    original = json.loads(src.read_text(encoding="utf-8"))
    result = polish_script(original)
    print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])
