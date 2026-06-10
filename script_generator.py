"""
Claw Bot — Script Generator (Session 1A)

Generates a 30-second kids' story script in JSON format.

Changes from previous version:
- LLM picks style ("storybook", "cartoon", "anime", "watercolor", "pixelart")
  based on story mood; user can override via revision feedback.
- LLM picks culture/setting (Indian, Western, Japanese, mixed, animal-kingdom,
  fantasy) based on theme — no longer hardcoded to Indian.
- LLM picks character types (human, animal, mixed) as fits the story.
- Safety filter still applies. Output schema extended with 'style' and 'culture'.
"""

import json
import logging
import re
import sys
import time as _t
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import model_registry
from modules import safety_filter as sf
from modules import runtime_settings as rs
from modules import generation_meta as gm
from modules import feedback_thinker as ft

log = logging.getLogger("claw_bot.script_generator")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
OUTPUTS_DIR = PROJECT_ROOT / "04_Outputs" / "scripts"
STYLES_PATH = PROJECT_ROOT / "05_Config" / "styles.json"


# ==============================================================================
# STYLE LIBRARY ACCESSOR
# ==============================================================================

def _load_styles() -> dict:
    if not STYLES_PATH.exists():
        return {"default": "storybook", "available": {}}
    return json.loads(STYLES_PATH.read_text(encoding="utf-8"))


def get_available_style_ids() -> list[str]:
    styles = _load_styles()
    return list(styles.get("available", {}).keys())


def get_style_description(style_id: str) -> dict:
    styles = _load_styles()
    return styles.get("available", {}).get(style_id, {})


def get_default_style() -> str:
    styles = _load_styles()
    return styles.get("default", "storybook")


# ==============================================================================
# SYSTEM PROMPT — the heart of the generator
# ==============================================================================

def _build_system_prompt() -> str:
    available_styles = get_available_style_ids()
    styles_data = _load_styles().get("available", {})
    style_guide_lines = []
    for sid, info in styles_data.items():
        style_guide_lines.append(
            f'  - "{sid}": {info.get("description", "")} '
            f'Best for: {info.get("best_for", "")}'
        )
    style_guide = "\n".join(style_guide_lines)

    return f"""You output ONLY valid JSON. No prose, no preamble, no markdown fences. Start with {{ end with }}.

You are Claw Bot, a master children's storyteller for ages 3-10. You write tiny illustrated stories that feel like Pixar shorts or Bluey episodes — short, but with a real heart.

# WHAT MAKES A GOOD STORY (read this carefully)

A story is NOT a list of things that happen. A story has a SHAPE:

1. **HOOK** — show the character WANTING something or FEELING something specific. Not "a boy walks down a path" — instead "a boy is hurrying home before the rain". The audience must care in the first 3 seconds.

2. **SPARK** — something unexpected happens. A problem, a discovery, an obstacle, a stranger, a small disaster. This is what makes the story start. Without a spark, you just have someone existing.

3. **STRUGGLE** — the character tries something, or hesitates, or has to figure it out. This is where the moral lives — not in narration, but in WHAT THE CHARACTER CHOOSES TO DO.

4. **CHOICE / TURN** — the character makes a clear, visible decision. They could have walked away. They didn't. THIS is the heart of the story.

5. **CONSEQUENCE** — the world responds to the choice. Something visibly changes — a smile, a transformation, a
reward, a small wonder. The audience FEELS the payoff.

If your story is missing the SPARK, the CHOICE, or the CONSEQUENCE — start over. A flat sequence of "and then...
and then... and then..." is a failure.

# A WORKED EXAMPLE OF GOOD STORY SHAPE (fable density, scaled to 30 seconds)

Theme: "a proud rabbit and a steady tortoise race." A strong 7-shot version reads like this (narration only shown):
  1. (atmosphere) "Morning dew sparkled on the meadow as the woodland animals gathered to watch the great race."
  2. (hook) "Rex the rabbit grinned, sure his fast legs would beat the slow little tortoise easily."
  3. (spark) "The whistle blew, and Rex shot ahead in a blur while Terry plodded calmly behind."
  4. (struggle) "Far in front, Rex yawned and curled beneath a shady tree for a quick little nap."
  5. (observation) "Step after patient step, Terry kept moving, never once stopping to rest or look back."
  6. (choice/turn) "Terry passed the sleeping rabbit quietly and pressed on toward the distant finish line."
  7. (consequence) "Rex woke too late — Terry had won, and the meadow cheered for the steady little tortoise."
  8. (Moral) Slow and steady wins the race.
  
Notice: ~84 words total, full sentences, clear arc, a real turn, an earned ending. THIS is the density and shape you aim for. Match this richness — never produce thin one-line shots.

# BREATHING SHOTS (cinematography that lets emotion land)

Children's stories for ages 3-10 NEED pauses to let kids absorb what's happening. A pure-plot sequence feels rushed
and emotionally flat. Use these shot TYPES alongside the 5 beats above:

- **atmosphere** — wide establishing shot setting the mood (an empty playground at dawn, a quiet kitchen). No
character action; pure setting. Often opens the story.
- **reaction** — a close-up on a character's face right after something happened. No new action — just feeling. STILL needs a narration line — observational, describing what we see (e.g. "The ball flies past Buddy's nose." or "She pauses, eyes wide.") — NEVER leave narration empty.
- **moment-of-decision** — a held beat BEFORE the choice. The character looks down, hesitates, takes a breath.
Tension is the point.
- **observation** — the character watches something happen to others. Quiet, contemplative.

These are NOT padding. They are STORYTELLING beats that earn their place when the story needs to slow down for
emotional weight. A 30-second story can absolutely have an atmosphere shot AND a reaction shot.

You may use any of these as the FIRST shot (before the hook) if it earns the opening — like a wide atmospheric
shot establishing the world before the character appears.

Mark these in the `beat` field exactly as listed: `"atmosphere"`, `"reaction"`, `"moment-of-decision"`, `"observation"`.

# RULES

1. **Shot count & length: hit the 30-second buffer.** The finished video is ~30 seconds. Narration is read aloud at ~2.5 words/second, so the WHOLE story needs roughly **70-85 words of narration total** across all shots. Shot count is whatever it takes to reach that — usually **6-9 shots**. Do NOT stop at 5 thin shots; that leaves the video half-empty. Each shot's narration should be a full, vivid sentence (10-14 words), not a fragment. Count your words: if the total is under 70, the story is too short — add a beat or enrich the narration until it lands between 70 and 85 words.

2. **Each shot serves a purpose — but purposes include EMOTION, not just plot.** A shot is justified if it
advances plot OR lets the audience feel something. "He looks down at the broken cup, eyes wide" is a valid shot
even though no plot moved — it gives the choice that comes next its emotional weight. Don't delete reaction or
atmosphere shots just because plot didn't move; delete shots that do NEITHER (a character walking from A to B
when arrival is what matters).

3. **Show, don't tell.** Narration describes what the audience SEES, never what the character feels internally. NOT "she felt sad" — instead "she sat alone, head in her hands". NOT "he learned to share" — instead "he held out his cookie".

4. **Narration: 10-14 words per shot. NEVER empty. NEVER a fragment.** Write FULL sentences a picture-book narrator would read aloud — like a classic fable. Specific verbs, warm simple language, no emotion-labels. Breathing shots (atmosphere/reaction/observation/moment-of-decision) ALSO need full narration — observational instead of action-driven. 

GOOD (full, fable-like): "The morning sun spilled across the meadow as the animals gathered to watch." 
GOOD: "Rex laughed and bounded ahead, certain the slow tortoise could never catch him." 
BAD (fragment, too thin): "Rex runs fast." 
BAD: "Terry is slow." 

Each shot should advance the story AND add texture — a detail of the world, a flicker of feeling, a turn of events. Aim the whole story at 70-85 narration words total.

5. **Camera variety (CRITICAL CAMERA RULE):** For every `visual_description`, `first_frame_prompt`, and `last_frame_prompt`, you MUST specify a dynamic cinematic camera angle. Never use flat, side-by-side theatrical staging. Alternate between Extreme Close-Ups (ECU), Over-The-Shoulder (OTS), High-Angle Drone shots, and Low-Angle tracking shots. If two characters are interacting, use alternating close-ups or OTS shots instead of placing them side-by-side.

6. **First & last frame coherence:** the locked_visual_token guarantees identical character appearance across ALL shots. Your first_frame_prompt and last_frame_prompt must NEVER re-describe the character's hair, clothes, glasses, age, species, or any appearance trait — refer to them by name only. Different pose/expression/position shows the change across the shot. If you describe the character's appearance inside a shot prompt, you are breaking the system.

7. **Motion prompt:** describes only physical motion + camera movement. Forbidden words: says, speaks, talking, mouth, lip, voice. Characters never speak — narration is voiceover.

8. **Style suffix added later** — do NOT include style tags ("anime style", "Pixar-style") in any prompt field.

9. **Characters need a clear feeling or want from shot 1.** Not "Rohan walks". Instead "Rohan, hurrying, almost steps on a tiny turtle". The want is built into the action.

10. **The moral is felt, not stated.** The "moral" field is a one-line summary FOR THE PARENT — but the narration must never say it directly.

11. **Pick a music mood that fits the story's emotional tone:**
- "cheerful" — upbeat happy stories, playful adventures
- "calm" — bedtime stories, quiet reflective moments
- "adventurous" — exploration, curiosity, journeys
- "tender" — emotional/heartfelt, family bonds, kindness
- "magical" — fantasy, fairy-tale, wonder, dreams
- "energetic" — fast-paced, action, excitement, silly fun

12. **VALID JSON ONLY:** Every field MUST be followed by a comma except the last one in its object. Every string MUST be in double quotes. NO comments. Test the JSON would parse before responding.

# OUTPUT FORMAT

{{
  "title": "<3-6 words>",
  "theme": "<the user's theme>",
  "culture": "indian | western | japanese | mixed | animal-kingdom | fantasy",
  "style": "storybook | cartoon | anime | watercolor | pixelart",
  "duration_seconds": <total — narration is read aloud at ~2.5 words/sec>,
  "characters": [
    {{
      "name": "...",
      "type": "human | animal | creature",
      "appearance": "<one specific sentence describing ONLY visual traits: species, age, color, body shape, clothing, distinctive features. NO poses, NO locations, NO actions. BAD: 'a black crow perched on a branch'. GOOD: 'a sleek black crow with sharp amber eyes and glossy black feathers'.>",
      "locked_visual_token": "<the SAME visual description as 'appearance' but compressed to a tight 15-30 word phrase that can be pasted verbatim into every shot prompt. Must include: age + species/race + hair + clothing colors + ONE distinctive feature. NO actions, NO emotions, NO setting. Example: 'a 6-year-old boy with curly brown hair and round glasses, wearing a red t-shirt, blue shorts, and white sneakers'.>"
    }}
  ],
  "setting": "<one sentence>",
  "shots": [
    {{
      "shot_number": 1,
      "beat": "atmosphere | hook | spark | reaction | observation | struggle | moment-of-decision | choice | consequence",
      "narration": "<8-15 words, specific actions, no emotion labels>",
      "visual_description": "<one sentence>",
      "first_frame_prompt": "<40-70 words: ONLY pose + setting + camera framing + lighting. DO NOT re-describe the character's hair, clothes, age, species, glasses, or any appearance trait — the locked_visual_token will be injected automatically. Refer to the character by NAME only. Example: 'Rohan stands at the edge of a dusty courtyard, head tilted down, hands clasped behind his back. Wide shot, late afternoon golden light, soft warm shadows.'>",
      "last_frame_prompt": "<40-70 words: SAME rules — pose + setting + framing + lighting only. Show a clearly different pose/expression/position from first_frame to convey the change across this shot. Refer to character by NAME only.>",
      "motion_prompt": "<30-60 words: only physical motion + camera, no speech>"
    }}
  ],
  "moral": "<one sentence summary for parents — never spoken in the story>",
  "music_mood": "cheerful | calm | adventurous | tender | magical | energetic"
}}

# VISUAL STYLES (pick one matching the story's mood):
{style_guide}

# SAFETY

Age-appropriate. No graphic violence, no fear-mongering, no heavy themes (death, illness, poverty) unless the user's theme explicitly asks. Conflict is good — danger and dread are not.

# OUTPUT

Output ONLY the JSON object. Start with {{ end with }}. Nothing else."""


# ==============================================================================
# OLLAMA CALL
# ==============================================================================

def _call_llm(prompt: str, system_prompt: str) -> str:
    """Call Ollama's generate endpoint and return raw text response."""
    cfg = model_registry.get_active("llm_backend")
    model_name = cfg.get("model_id") or cfg.get("model_name") or cfg.get("model") or "llama3.1:8b-instruct-q8_0"
    url = cfg.get("server_url", "http://127.0.0.1:11434") + "/api/generate"

    # Forcefully combine prompts to bypass missing Ollama Chat Templates
    combined_prompt = f"SYSTEM INSTRUCTIONS (STRICTLY FOLLOW THESE):\n{system_prompt}\n\nUSER REQUEST:\n{prompt}"

    payload = {
        "model": model_name,
        "prompt": combined_prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.75,
            "top_p": 0.95,
            "num_ctx": 8192,
            "num_predict": 8192,
        },
    }

    log.info(f"Calling LLM ({model_name}) with theme prompt: {prompt[:80]}...")
    log.info(f"System prompt starts with: {system_prompt[:50]}")
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


# ==============================================================================
# JSON EXTRACTION + VALIDATION
# ==============================================================================

def _extract_json(raw: str) -> dict:
    """Pull JSON out of an LLM response, tolerantly. Saves raw output for debugging."""
    # Save raw output for debugging
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUTS_DIR / "_last_raw_llm_output.txt").write_text(raw, encoding="utf-8")
    except Exception:
        pass

    # Strip <think>...</think> blocks (reasoning models)
    # Strip closed <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    # Strip UNCLOSED <think>... blocks (model got cut off before closing tag)
    # Cut everything from <think> to the first { (the JSON start)
    if "<think>" in cleaned.lower():
        json_start = cleaned.find("{")
        if json_start > 0:
            cleaned = cleaned[json_start:]
    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    # Find the JSON block (greedy, outermost braces)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    candidate = match.group(0) if match else cleaned

    # Attempt 1: strict JSON
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        log.warning(f"Strict JSON parse failed: {e}. Trying repair...")

    # Attempt 2: common LLM JSON mistakes
    repaired = candidate
    # Fix MISSING commas between JSON fields: `"...value"\n    "next_key":` (Llama 3.1 often drops these)
    repaired = re.sub(r'("\s*)\n(\s*")', r'\1,\n\2', repaired)
    # Fix unescaped newlines inside strings (very common)
    # Replace literal newlines that are inside "..." strings with \n
    def _escape_newlines_in_strings(s: str) -> str:
        out = []
        in_string = False
        escape = False
        for ch in s:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                out.append(ch)
                continue
            if in_string and ch == "\n":
                out.append("\\n")
                continue
            if in_string and ch == "\r":
                continue
            out.append(ch)
        return "".join(out)
    repaired = _escape_newlines_in_strings(repaired)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        log.error(f"Repair also failed: {e}")
        raise ValueError(
            f"Could not extract JSON from LLM response. "
            f"Raw output saved to: {OUTPUTS_DIR / '_last_raw_llm_output.txt'}. "
            f"First 300 chars: {raw[:300]}"
        )


def _validate_and_default(script: dict) -> dict:
    """Apply sensible defaults for missing fields. Does NOT cap shot count."""
    available_styles = get_available_style_ids()

    # Normalize characters: if LLM returned strings, wrap each in a dict
    raw_chars = script.get("characters", [])
    normalized = []
    for c in raw_chars:
        if isinstance(c, dict):
            normalized.append(c)
        elif isinstance(c, str):
            # Try to split on " - " or "—" to extract name+appearance
            parts = c.split(" - ", 1) if " - " in c else c.split(" — ", 1) if " — " in c else [c]
            if len(parts) == 2:
                normalized.append({"name": parts[0].strip(), "type": "character", "appearance": parts[1].strip()})
            else:
                normalized.append({"name": c.strip(), "type": "character", "appearance": ""})
    script["characters"] = normalized

    # Ensure every character has a locked_visual_token. If the LLM didn't
    # provide one, derive it from appearance (truncated). This is the SINGLE
    # source of truth for character look across all shots.
    for ch in script["characters"]:
        if not isinstance(ch, dict):
            continue
        token = (ch.get("locked_visual_token") or "").strip()
        if not token:
            appearance = (ch.get("appearance") or "").strip()
            if appearance:
                # Compress to ~30 words max for prompt efficiency
                words = appearance.split()
                token = " ".join(words[:30])
            else:
                token = ch.get("name", "a character")
            ch["locked_visual_token"] = token
            log.warning(
                f"Character '{ch.get('name')}' missing locked_visual_token; "
                f"derived from appearance: '{token[:80]}...'"
            )

    script.setdefault("title", "Untitled Story")
    script.setdefault("theme", "")
    script.setdefault("culture", "mixed")
    script.setdefault("style", get_default_style())
    script.setdefault("duration_seconds", 30)
    script.setdefault("characters", [])
    script.setdefault("setting", "")
    script.setdefault("shots", [])
    script.setdefault("moral", "")
    
    if script.get("music_mood") not in ("cheerful", "calm", "adventurous", "tender", "magical", "energetic"):
        script["music_mood"] = "cheerful"

    # Style must be one we know about; fall back to default
    if script["style"] not in available_styles:
        log.warning(
            f"LLM returned unknown style '{script['style']}'. Defaulting to '{get_default_style()}'."
        )
        script["style"] = get_default_style()

    # Minimum sanity floor only — let the model decide length
    shots = script.get("shots", [])
    if len(shots) < 3:
        raise ValueError(
            f"LLM produced only {len(shots)} shots — minimum is 3 "
            f"(need beginning/middle/end)"
        )
    if len(shots) > 20:
        log.warning(f"LLM produced {len(shots)} shots — capping at 20 for safety.")
        script["shots"] = shots[:20]

    # Detect duplicate narration (LLM padding)
    seen_narration = set()
    for s in script["shots"]:
        narr = s.get("narration", "").strip().lower()
        if narr and narr in seen_narration:
            log.warning(f"Duplicate narration detected: '{narr[:60]}...'")
        seen_narration.add(narr)

    return script


# ==============================================================================
# DURATION / WORD-BUDGET ENFORCEMENT
# ==============================================================================

WORDS_PER_SECOND = 2.5
TARGET_SECONDS = 30
MIN_NARRATION_WORDS = 65    # ≈26s — below this we expand
MAX_NARRATION_WORDS = 95    # ≈38s — above this we don't push further
MAX_EXPAND_ATTEMPTS = 2


def _count_narration_words(script: dict) -> int:
    total = 0
    for shot in script.get("shots", []):
        narr = (shot.get("narration") or "").strip()
        if narr:
            total += len(narr.split())
    return total


def _expand_script_to_buffer(script: dict) -> dict:
    """If narration is too short to fill ~30s, ask the LLM to enrich it.
    Repeats up to MAX_EXPAND_ATTEMPTS. Returns the (possibly) expanded script."""
    attempts = 0
    while attempts < MAX_EXPAND_ATTEMPTS:
        words = _count_narration_words(script)
        est_sec = words / WORDS_PER_SECOND
        if words >= MIN_NARRATION_WORDS:
            log.info(f"Narration {words} words (~{est_sec:.0f}s) — fills buffer, no expansion.")
            return script

        log.info(
            f"Narration only {words} words (~{est_sec:.0f}s) — under {MIN_NARRATION_WORDS}. "
            f"Expanding (attempt {attempts + 1}/{MAX_EXPAND_ATTEMPTS})."
        )

        system_prompt = _build_system_prompt()
        user_prompt = (
            f"Here is a children's story JSON that is TOO SHORT — its narration totals only "
            f"{words} words (~{est_sec:.0f} seconds), but the finished video must be ~{TARGET_SECONDS} "
            f"seconds, which needs {MIN_NARRATION_WORDS}-{MAX_NARRATION_WORDS} narration words total.\n\n"
            f"```\n{json.dumps(script, indent=2, ensure_ascii=False)}\n```\n\n"
            f"Enrich this story so its narration totals {MIN_NARRATION_WORDS}-{MAX_NARRATION_WORDS} words. "
            f"You may: lengthen thin narration lines into full vivid sentences, and/or add 1-3 NEW shots "
            f"(atmosphere, reaction, or struggle beats) where the story can breathe. Keep the SAME characters, "
            f"the SAME locked_visual_token for each (do not change appearance), the same arc and moral. "
            f"Renumber shots sequentially starting at 1. Output ONLY the JSON."
        )
        try:
            raw = _call_llm(user_prompt, system_prompt)
            expanded = _validate_and_default(_extract_json(raw))
            # Carry over locked tokens if the LLM dropped any
            orig_locks = {
                c.get("name", "").strip().lower(): c.get("locked_visual_token", "")
                for c in script.get("characters", []) if isinstance(c, dict)
            }
            for c in expanded.get("characters", []):
                if isinstance(c, dict):
                    nm = c.get("name", "").strip().lower()
                    if nm in orig_locks and orig_locks[nm] and not c.get("locked_visual_token"):
                        c["locked_visual_token"] = orig_locks[nm]
            new_words = _count_narration_words(expanded)
            # Only accept the expansion if it actually got longer
            if new_words > words:
                script = expanded
            else:
                log.warning("Expansion did not increase word count; keeping previous version.")
        except Exception as e:
            log.warning(f"Expansion attempt failed (keeping current script): {e}")
            return script
        attempts += 1

    final_words = _count_narration_words(script)
    log.info(f"Expansion done — final narration {final_words} words (~{final_words/WORDS_PER_SECOND:.0f}s).")
    return script


# ==============================================================================
# PUBLIC API
# ==============================================================================

def generate_script(theme: str, style_override: Optional[str] = None,
                    culture_override: Optional[str] = None) -> dict:
    """
    Generate a new script. Returns a dict with the parsed JSON + metadata.

    Args:
        theme: the creative prompt/theme for the story
        style_override: if given, forces a specific style (else LLM picks)
        culture_override: if given, forces a culture (else LLM picks)
    """
    # (Note: input theme safety is enforced post-generation via script-level check below)
    _job_start = _t.time()

    # Apply persistent user override if caller didn't specify one
    effective_style = style_override or rs.get_style_override()

    system_prompt = _build_system_prompt()
    user_prompt = f"Theme: {theme}\n\nWrite the story this theme deserves. Choose shot count based on what the story needs — short and punchy, or longer if the story has more emotional weight. Do NOT pad."

    # Detect explicit shot-count request in theme (e.g. "in 5 shots")
    shot_match = re.search(r"in (\d+)\s*shots?", theme, re.IGNORECASE)
    if shot_match:
        user_prompt += f"\n\nThe user explicitly requested {shot_match.group(1)} shots — match that exactly."

    if effective_style:
        user_prompt += f"\n\nYou MUST use style: {effective_style}"
    if culture_override:
        user_prompt += f"\n\nYou MUST use culture: {culture_override}"

    raw = _call_llm(user_prompt, system_prompt)
    script = _extract_json(raw)
    script = _validate_and_default(script)

    # Ensure the story is long enough to fill the ~30s buffer (TTS-first sync).
    # Honor an explicit shot-count request — don't expand if user pinned the count.
    if not shot_match:
        script = _expand_script_to_buffer(script)
        script = _validate_and_default(script)

    # Safety check on the generated script — filter returns (is_safe, blocked, warnings)
    is_safe, blocked_terms, warnings = sf.check_safety(script)
    if not is_safe:
        raise ValueError(
            f"Generated story contains hard-blocked terms: {blocked_terms}. "
            f"Try a different theme or word choice."
        )
    if warnings:
        log.info(f"Script generated with soft warnings: {warnings}")

    # Attach metadata
    now = datetime.now()
    script_id = now.strftime("%Y%m%d_%H%M%S")
    script["_id"] = script_id
    script["script_id"] = script_id  # legacy-compat
    script["_generated_at"] = now.isoformat()
    script["_revision_number"] = 1
    script["revision_number"] = 1  # legacy-compat

    # Apply style overrides post-hoc
    if effective_style:
        script["style"] = effective_style
    if culture_override:
        script["culture"] = culture_override

    # Save to disk
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / f"script_{script_id}.json"
    output_path.write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Script saved: {output_path}")

    # Record to generation_meta (dashboard auto-refresh picks it up)
    try:
        llm_cfg = model_registry.get_active("llm_backend")
        llm_id = llm_cfg.get("_id", "unknown") if llm_cfg else "unknown"
    except Exception:
        llm_id = "unknown"
    gm.record({
        "kind": "script",
        "script_id": script_id,
        "duration_sec": round(_t.time() - _job_start, 1),
        "vram_peak_mb": 0,
        "settings": {
            "backend_id": llm_id,
            "style": script.get("style"),
            "culture": script.get("culture"),
            "shots": len(script.get("shots", [])),
            "title": script.get("title", "")[:60],
        },
        "success": True,
    })
    return script


def revise_script(original_script: dict, feedback: str) -> dict:
    """
    Revise an existing script based on user feedback.
    Two-pass:
      1. Thinker (Qwen 2.5 14B) interprets the feedback into an edit plan.
      2. Script LLM rewrites the script guided by that plan.
    Thinker is optional — if it fails, we fall back to a plain rewrite.
    """
    log.info(f"Revising — running thinker pass on feedback: '{feedback[:80]}'")
    edit_plan = ft.think_about_feedback(original_script, feedback)
    plan_guidance = ft.plan_to_revision_prompt(edit_plan, feedback)

    system_prompt = _build_system_prompt()
    user_prompt = (
        f"Here is a previous version of the story JSON:\n\n"
        f"```\n{json.dumps(original_script, indent=2, ensure_ascii=False)}\n```\n\n"
        f"The user gave this feedback:\n\n"
        f'"{feedback}"\n\n'
    )
    if plan_guidance:
        user_prompt += (
            f"A story editor has reviewed the feedback and produced this edit plan. "
            f"Follow it carefully:\n\n{plan_guidance}\n\n"
        )
    user_prompt += (
        f"Rewrite the story as new JSON, applying the user's feedback (and the edit plan above if present). "
        f"Keep fields the feedback didn't mention. "
        f"IMPORTANT: Preserve each character's locked_visual_token EXACTLY as written "
        f"in the original — do not rephrase, expand, or shorten it, unless the user "
        f"specifically asked you to change the character's appearance. "
        f"Output ONLY the JSON."
    )

    raw = _call_llm(user_prompt, system_prompt)
    revised = _extract_json(raw)
    revised = _validate_and_default(revised)

    # Safety net: if the LLM dropped or mangled any locked_visual_token,
    # restore it from the original. The lock only changes when the user
    # explicitly asks to change appearance.
    feedback_lower = feedback.lower()
    appearance_keywords = (
        "appearance", "looks like", "look like", "wearing", "outfit",
        "clothes", "hair", "glasses", "age", "older", "younger",
        "redesign", "redraw character",
    )
    user_wants_appearance_change = any(k in feedback_lower for k in appearance_keywords)

    if not user_wants_appearance_change:
        original_locks = {
            c.get("name", "").strip().lower(): c.get("locked_visual_token", "")
            for c in original_script.get("characters", [])
            if isinstance(c, dict)
        }
        for c in revised.get("characters", []):
            if not isinstance(c, dict):
                continue
            name = c.get("name", "").strip().lower()
            if name in original_locks and original_locks[name]:
                if c.get("locked_visual_token", "") != original_locks[name]:
                    log.info(
                        f"Restoring locked_visual_token for '{c.get('name')}' "
                        f"(revision changed it, but user didn't ask for appearance change)"
                    )
                    c["locked_visual_token"] = original_locks[name]

    # Preserve identity + bump revision counter
    rev_num = int(original_script.get("_revision_number", 1)) + 1
    original_id = original_script.get("_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    # Revised scripts get _vN suffix if the original id already exists
    new_id = f"{original_id}_v{rev_num}"

    revised["_id"] = new_id
    revised["script_id"] = new_id  # legacy-compat
    revised["_generated_at"] = datetime.now().isoformat()
    revised["_revision_number"] = rev_num
    revised["revision_number"] = rev_num  # legacy-compat
    revised.setdefault("_parent_id", original_id)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / f"script_{new_id}.json"
    output_path.write_text(json.dumps(revised, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Revised script saved: {output_path}")
    return revised


def format_for_discord(script: dict) -> str:
    """Format a script into a readable Discord message."""
    title = script.get("title", "Untitled")
    theme = script.get("theme", "")
    style = script.get("style", "")
    culture = script.get("culture", "")
    moral = script.get("moral", "")
    rev = script.get("_revision_number", 1)
    rev_suffix = f" (revision {rev})" if rev > 1 else ""
    script_id = script.get("_id", "")

    style_info = get_style_description(style)
    style_label = style_info.get("display_name", style) if style_info else style

    lines = [
        f"**📖 {title}**{rev_suffix}",
        f"_Theme: {theme}_",
        f"🎨 Style: **{style_label}**  ·  🌍 Culture: **{culture}**  ·  🎬 Shots: **{len(script.get('shots', []))}**",
        f"⏱️ Est. narration: **~{_count_narration_words(script) / WORDS_PER_SECOND:.0f}s** ({_count_narration_words(script)} words)",
        f"ID: `{script_id}`",
        "",
        "**Characters:**",
    ]
    for c in script.get("characters", []):
        if isinstance(c, dict):
            name = c.get('name', 'Someone')
            ctype = c.get('type', 'character')
            appearance = c.get('appearance', '')
            lock = c.get('locked_visual_token', '')
            lines.append(f"• **{name}** _({ctype})_ — {appearance}")
            if lock and lock != appearance:
                lines.append(f"   🔒 _Locked: {lock[:200]}_")
        elif isinstance(c, str):
            lines.append(f"• {c}")
    lines.append("")
    lines.append(f"**Setting:** {script.get('setting', '')}")
    lines.append("")
    lines.append("**Shots:**")
    for shot in script.get("shots", []):
        n = shot.get("shot_number", "?")
        narration = shot.get("narration", "")
        vd = shot.get("visual_description", "")
        lines.append(f"**{n}.** 🎙️ *{narration}*")
        lines.append(f"       🖼️ {vd}")
    lines.append("")
    lines.append(f"✨ **Moral:** {moral}")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    test_theme = "a shy little mouse learning to speak up"
    result = generate_script(test_theme)
    print(json.dumps(result, indent=2))