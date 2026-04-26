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

    return f"""You are Claw Bot, a creative children's story writer. You write short picture-book stories for kids (ages 3-10) that teach good habits, emotional intelligence, or gentle life lessons.

## YOUR JOB

Given a theme, write a complete story and output it as structured JSON. You make creative decisions about:

1. **Cultural setting** — pick what fits the theme best:
   - "indian" — names like Arjun, Priya, Kavya; settings like Indian courtyards, festivals, joint families
   - "western" — names like Emma, Oliver, Sophie; suburban homes, parks, schools
   - "japanese" — names like Yuki, Haruto, Sakura; rice-paper screens, cherry blossoms, gentle minimalism
   - "mixed" — a diverse cast; multicultural setting like an international school
   - "animal-kingdom" — the story is about animals only (no humans); forest, savanna, ocean, burrow
   - "fantasy" — dragons, fairies, robots, sentient clouds; no real-world location

2. **Character types** — humans, animals, or mixed (e.g., "a girl and her turtle"). Animals should talk if story needs it.

3. **Visual style** — pick ONE that matches the story's mood:
{style_guide}

4. **Shot count** — usually 4 shots for a 30-second story. You may use 3-6 shots if the story clearly needs it. Default to 4 unless the story really benefits from more/fewer.

## OUTPUT SCHEMA (strict JSON, no markdown)

{{
  "title": "Short evocative title (3-6 words)",
  "theme": "the theme given to you, rephrased slightly if needed",
  "culture": "indian" | "western" | "japanese" | "mixed" | "animal-kingdom" | "fantasy",
  "style": "storybook" | "cartoon" | "anime" | "watercolor" | "pixelart",
  "duration_seconds": 30,
  "characters": [
    {{
      "name": "Character's name",
      "type": "human" | "animal" | "creature",
      "appearance": "Vivid sentence describing their visual look, clothing, distinguishing features. Be specific enough that an illustrator could draw them consistently."
    }}
  ],
  "setting": "One sentence describing where this story takes place, specific and sensory.",
  "shots": [
    {{
      "shot_number": 1,
      "narration": "Single short sentence (12-18 words) the narrator says during this shot.",
      "visual_description": "What's happening in this shot, in one vivid sentence.",
      "first_frame_prompt": "A detailed paragraph (80-150 words) describing the OPENING image of this shot. Written as descriptive prose, not tags. Include composition, character poses, expressions, lighting, atmosphere. Do NOT include style tags — those come from the style field. Mention characters by name so they stay consistent.",
      "last_frame_prompt": "A detailed paragraph describing the ENDING image of this shot — what changed from the first_frame. Same style rules as first_frame_prompt."
    }}
  ],
  "moral": "One sentence stating what the child learns from this story."
}}

## RULES

- The story MUST be safe, positive, age-appropriate. No violence, fear content, or scary themes.
- Narration lines together should sum to ~30 seconds at normal reading speed (roughly 65-80 words total across all shots).
- Each shot should advance the story — a clear emotional arc: setup → tension → choice → resolution.
- Character descriptions must be SPECIFIC enough to keep them recognizable across shots.
- Frame prompts are prose paragraphs, not comma-separated tags. Natural language.
- Output ONLY the JSON object. No preamble, no "```json" fence, no post-text. Just the JSON.

Now write the story."""


# ==============================================================================
# OLLAMA CALL
# ==============================================================================

def _call_llm(prompt: str, system_prompt: str) -> str:
    """Call Ollama's generate endpoint and return raw text response."""
    cfg = model_registry.get_active("llm_backend")
    model_name = cfg.get("model_name") or cfg.get("model") or "llama3.1:8b-instruct-q4_K_M"
    url = cfg.get("server_url", "http://127.0.0.1:11434") + "/api/generate"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.8,
            "top_p": 0.9,
            "num_ctx": 8192,
        },
    }

    log.info(f"Calling Llama ({model_name}) with theme prompt: {prompt[:80]}...")
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


# ==============================================================================
# JSON EXTRACTION + VALIDATION
# ==============================================================================

def _extract_json(raw: str) -> dict:
    """Pull JSON out of an LLM response, tolerantly."""
    # Strip markdown fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not extract JSON from LLM response: {raw[:300]}")


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

    script.setdefault("title", "Untitled Story")
    script.setdefault("theme", "")
    script.setdefault("culture", "mixed")
    script.setdefault("style", get_default_style())
    script.setdefault("duration_seconds", 30)
    script.setdefault("characters", [])
    script.setdefault("setting", "")
    script.setdefault("shots", [])
    script.setdefault("moral", "")

    # Style must be one we know about; fall back to default
    if script["style"] not in available_styles:
        log.warning(
            f"LLM returned unknown style '{script['style']}'. Defaulting to '{get_default_style()}'."
        )
        script["style"] = get_default_style()

    # No validation on shot count — LLM decides
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

    system_prompt = _build_system_prompt()
    user_prompt = f"Theme: {theme}"
    if style_override:
        user_prompt += f"\n\nYou MUST use style: {style_override}"
    if culture_override:
        user_prompt += f"\n\nYou MUST use culture: {culture_override}"

    raw = _call_llm(user_prompt, system_prompt)
    script = _extract_json(raw)
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
    if style_override:
        script["style"] = style_override
    if culture_override:
        script["culture"] = culture_override

    # Save to disk
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / f"script_{script_id}.json"
    output_path.write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Script saved: {output_path}")
    return script


def revise_script(original_script: dict, feedback: str) -> dict:
    """
    Revise an existing script based on user feedback.
    Reuses system prompt + adds feedback + original as context.
    """
    system_prompt = _build_system_prompt()
    user_prompt = (
        f"Here is a previous version of the story JSON:\n\n"
        f"```\n{json.dumps(original_script, indent=2, ensure_ascii=False)}\n```\n\n"
        f"The user gave this feedback:\n\n"
        f'"{feedback}"\n\n'
        f"Rewrite the story as new JSON, applying the user's feedback. "
        f"Keep fields the feedback didn't mention. Output ONLY the JSON."
    )

    raw = _call_llm(user_prompt, system_prompt)
    revised = _extract_json(raw)
    revised = _validate_and_default(revised)

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
        f"ID: `{script_id}`",
        "",
        "**Characters:**",
    ]
    for c in script.get("characters", []):
        if isinstance(c, dict):
            lines.append(f"• **{c.get('name', 'Someone')}** _({c.get('type', 'character')})_ — {c.get('appearance', '')}")
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