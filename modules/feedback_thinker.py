"""
Claw Bot — Feedback Thinker
Pre-revision reasoning pass. Qwen 2.5 14B reads user feedback + current script,
returns a structured edit plan that revise_script uses as guidance.
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import requests

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import model_registry

log = logging.getLogger("claw_bot.feedback_thinker")

OLLAMA_URL = "http://127.0.0.1:11434"
THINKER_MODEL = "qwen3.6-thinker"  # registered via Modelfile, points at the 35B A3B IQ3_XXS GGUF


def _thinker_model_name() -> str:
    """Use a dedicated thinker_backend if registered, else fall back to polish backend."""
    try:
        cfg = model_registry.get_active("thinker_backend")
        if cfg:
            return cfg.get("model_id") or cfg.get("model_name") or THINKER_MODEL
    except Exception:
        pass
    return THINKER_MODEL


THINK_SYSTEM_PROMPT = """You output ONLY valid JSON. No prose, no preamble. Start with { end with }.

You are a story editor. Your job is to read user feedback on a children's story script and produce a structured EDIT PLAN — NOT a rewritten script.

Think step by step:

1. WHAT does the user actually want changed? Be specific. Distinguish between:
   - Pose/action change (visible motion only) → motion_prompt + first/last_frame_prompt
   - Emotional/narrative change (different feeling) → narration + frame_prompts
   - Structural change (shot order, count, beats) → shots array
   - Style/tone change → style field or whole rewrite
   - Character appearance change → locked_visual_token (rare; only if explicitly asked)

2. WHICH shots are affected? Surgical (1-2 shots) is almost always better than rewriting everything.

3. WHAT fields specifically should change in each affected shot?

4. DOES the change risk breaking continuity with adjacent shots? If yes, note it.

Output this exact JSON shape:

{
  "interpretation": "<one sentence — what the user wants in plain words>",
  "scope": "surgical | full_rewrite",
  "target_shots": [<shot_numbers>],
  "field_changes": {
    "<shot_number>": {
      "narration": "<new value OR null if unchanged>",
      "first_frame_prompt": "<new value OR null>",
      "last_frame_prompt": "<new value OR null>",
      "motion_prompt": "<new value OR null>",
      "visual_description": "<new value OR null>",
      "beat": "<new value OR null>"
    }
  },
  "global_changes": {
    "title": "<new value OR null>",
    "theme": "<new value OR null>",
    "moral": "<new value OR null>",
    "style": "<new value OR null>",
    "culture": "<new value OR null>",
    "music_mood": "<new value OR null>",
    "setting": "<new value OR null>"
  },
  "preserve_locked_tokens": true,
  "continuity_notes": "<one sentence — anything to watch for, or 'none'>"
}

RULES:
- NEVER change locked_visual_token unless user explicitly asked to change character appearance.
- If scope is surgical, leave non-target shots completely untouched.
- field_changes values are FINAL text, ready to drop into the JSON — not instructions.
- frame_prompts must stay in the locked-token style: pose + framing + setting only, refer to character by NAME, no appearance words.
- Output ONLY the JSON object. Nothing else."""


def _call_thinker(user_prompt: str) -> str:
    model = _thinker_model_name()
    url = OLLAMA_URL + "/api/generate"
    combined = f"SYSTEM:\n{THINK_SYSTEM_PROMPT}\n\nUSER:\n{user_prompt}"
    payload = {
        "model": model,
        "prompt": combined,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_ctx": 16384,
            "num_predict": 4096,
        },
    }
    log.info(f"Thinker calling {model}...")
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def _extract_json(raw: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    if "<think>" in cleaned.lower():
        json_start = cleaned.find("{")
        if json_start > 0:
            cleaned = cleaned[json_start:]
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    candidate = match.group(0) if match else cleaned
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Repair: drop trailing commas
        repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)
        return json.loads(repaired)


def think_about_feedback(original_script: dict, feedback: str) -> Optional[dict]:
    """
    Run the thinking pass. Returns an edit plan dict, or None if it fails.
    The caller (revise_script) uses the plan as guidance, not as a hard contract.
    """
    user_prompt = (
        f"ORIGINAL SCRIPT:\n```\n{json.dumps(original_script, indent=2, ensure_ascii=False)}\n```\n\n"
        f"USER FEEDBACK:\n\"{feedback}\"\n\n"
        f"Produce the EDIT PLAN JSON."
    )
    try:
        raw = _call_thinker(user_prompt)
        plan = _extract_json(raw)
        log.info(
            f"Edit plan: scope={plan.get('scope')} | "
            f"shots={plan.get('target_shots')} | "
            f"interpretation: {plan.get('interpretation', '')[:80]}"
        )
        return plan
    except Exception as e:
        log.warning(f"Thinker pass failed (continuing without plan): {e}")
        return None


def plan_to_revision_prompt(plan: dict, feedback: str) -> str:
    """Turn an edit plan into a guidance string that revise_script appends to its LLM call."""
    if not plan:
        return ""
    lines = [
        "EDIT PLAN (follow this exactly):",
        f"- Interpretation: {plan.get('interpretation', '')}",
        f"- Scope: {plan.get('scope', 'surgical')}",
        f"- Target shots: {plan.get('target_shots', [])}",
    ]
    field_changes = plan.get("field_changes") or {}
    if field_changes:
        lines.append("- Per-shot changes:")
        for shot_n, changes in field_changes.items():
            non_null = {k: v for k, v in (changes or {}).items() if v is not None}
            if non_null:
                lines.append(f"  Shot {shot_n}:")
                for k, v in non_null.items():
                    v_short = (str(v)[:200] + "...") if len(str(v)) > 200 else v
                    lines.append(f"    {k}: {v_short}")
    global_changes = plan.get("global_changes") or {}
    non_null_globals = {k: v for k, v in global_changes.items() if v is not None}
    if non_null_globals:
        lines.append("- Global changes:")
        for k, v in non_null_globals.items():
            lines.append(f"  {k}: {v}")
    if plan.get("preserve_locked_tokens", True):
        lines.append("- PRESERVE every character's locked_visual_token EXACTLY (do not rephrase).")
    notes = plan.get("continuity_notes", "")
    if notes and notes.lower() != "none":
        lines.append(f"- Continuity: {notes}")
    if plan.get("scope") == "surgical":
        lines.append("- For shots NOT in target_shots: copy them verbatim from the original. Do not touch them.")
    lines.append(f"- Original user feedback (for reference): \"{feedback}\"")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    fake_script = {
        "title": "The Shared Stick",
        "characters": [{"name": "Leo", "locked_visual_token": "a 6yo boy in orange shirt"}],
        "shots": [
            {"shot_number": 1, "narration": "Leo finds a stick.", "motion_prompt": "Leo picks up a stick."},
            {"shot_number": 2, "narration": "Tommy sees it.", "motion_prompt": "Tommy walks over."},
            {"shot_number": 3, "narration": "Leo hesitates.", "motion_prompt": "Leo holds the stick."},
        ],
    }
    plan = think_about_feedback(fake_script, "make Leo's hesitation feel like real reluctance, not just standing still")
    print(json.dumps(plan, indent=2))
    print("\n--- Revision prompt ---")
    print(plan_to_revision_prompt(plan, "make Leo's hesitation feel like real reluctance"))