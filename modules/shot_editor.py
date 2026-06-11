"""
Claw Bot — Shot Editor

Insert a brand-new shot into an EXISTING script at any position (before /
after / between shots), even after the storyboard or videos were rendered.

What it does, in order:
  1. Load the script JSON and ask Qwen to write the new shot (beat, shot_type,
     narration, visual_description, frame prompts, motion prompt) using the
     surrounding shots as context, plus an optional user brief.
  2. Renumber: every existing shot at/after the insert position shifts +1.
  3. Shift on-disk artifacts DESCENDING (highest shot first, so renames never
     collide) so already-rendered work stays attached to its original shot:
       - approved_prompts/{sid}.json   prompt keys
       - storyboards/{sid}/shot{N}_*.png  renames + storyboard.json manifest
       - clips/clip_{sid}_shot{N}[*].mp4  renames
       - clips/clip_{sid}_seeds.json   seed keys
  4. Generate an UNAPPROVED prompts entry for the new shot (same pattern as
     prompt_approval.generate_all_prompts) so the user can edit/approve it,
     then regenerate just that shot's frame + clip.

Only the new shot needs rendering afterwards — everything else is untouched.
"""

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import prompt_assembler as pa
from modules import script_generator as sg
from modules.file_utils import atomic_write_json

log = logging.getLogger("claw_bot.shot_editor")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SCRIPTS_DIR = PROJECT_ROOT / "04_Outputs" / "scripts"
STORYBOARDS_DIR = PROJECT_ROOT / "04_Outputs" / "storyboards"
CLIPS_DIR = PROJECT_ROOT / "04_Outputs" / "clips"
APPROVED_PROMPTS_DIR = PROJECT_ROOT / "04_Outputs" / "approved_prompts"

_VALID_SHOT_TYPES = ("wide", "medium", "closeup", "insert")
_VALID_BEATS = (
    "atmosphere", "hook", "spark", "reaction", "observation",
    "struggle", "moment-of-decision", "choice", "consequence",
)


# ==============================================================================
# NEW-SHOT GENERATION (LLM with fallback)
# ==============================================================================

def _shot_context_line(shot: Optional[dict], label: str) -> str:
    if not shot:
        return f"{label}: (none — this is the {'first' if 'BEFORE' in label else 'last'} shot)"
    return (
        f"{label} (shot {shot.get('shot_number')}, beat={shot.get('beat')}, "
        f"shot_type={shot.get('shot_type', '?')}):\n"
        f"  narration: {shot.get('narration', '')}\n"
        f"  visual: {shot.get('visual_description', '')}"
    )


def _build_new_shot_system_prompt() -> str:
    return """You output ONLY valid JSON. No prose, no preamble, no markdown fences. Start with { end with }.

You write ONE new shot to be inserted into an existing 30-second animated kids' story. The story is already written and approved — your shot must slot between the surrounding shots WITHOUT changing the plot. It adds drama, breathing room, or visual texture (an establishing wide, a reaction closeup, an insert of hands on an object).

# RULES
1. Narration: ONE full sentence, 8-15 words, picture-book narrator voice, present or past tense matching the surrounding narration. Show, don't tell — no emotion labels. NEVER empty.
2. shot_type: exactly one of "wide" (whole location visible, character small), "medium" (waist-up/full body action), "closeup" (face fills frame), "insert" (extreme close-up of hands/paws + object, face out of frame). If a character touches or picks up an object, prefer "insert". For establishing a place, use "wide".
3. beat: one of atmosphere | hook | spark | reaction | observation | struggle | moment-of-decision | choice | consequence — pick what fits this position in the arc.
4. first_frame_prompt / last_frame_prompt: 30-60 words each — ONLY pose + setting + camera framing + lighting. NEVER re-describe a character's appearance (hair, clothes, age, species) — refer to characters by NAME only. last_frame shows a clearly different pose/position than first_frame.
5. motion_prompt: 25-50 words — ONE main physical action plus ONE camera move. No speech, no mouth movement, no style words.
6. visual_description: one short sentence of what is on screen.
7. Do NOT introduce new characters or new locations unless the user brief explicitly asks.

# OUTPUT FORMAT
{
  "beat": "...",
  "shot_type": "wide | medium | closeup | insert",
  "narration": "...",
  "visual_description": "...",
  "first_frame_prompt": "...",
  "last_frame_prompt": "...",
  "motion_prompt": "..."
}

Output ONLY the JSON object."""


def _generate_shot_llm(script: dict, position: int, brief: str) -> dict:
    shots = script.get("shots", []) or []
    by_num = {s.get("shot_number"): s for s in shots}
    prev_shot = by_num.get(position - 1)
    next_shot = by_num.get(position)  # the shot currently holding this slot

    char_lines = []
    for c in script.get("characters", []) or []:
        if isinstance(c, dict):
            char_lines.append(
                f"- {c.get('name', '?')} ({c.get('type', 'character')}): "
                f"{c.get('locked_visual_token') or c.get('appearance', '')}"
            )
    user_prompt = (
        f"STORY TITLE: {script.get('title', '')}\n"
        f"SETTING: {script.get('setting', '')}\n"
        f"MORAL: {script.get('moral', '')}\n"
        f"CHARACTERS:\n" + ("\n".join(char_lines) or "(none)") + "\n\n"
        f"{_shot_context_line(prev_shot, 'SHOT BEFORE the new one')}\n\n"
        f"{_shot_context_line(next_shot, 'SHOT AFTER the new one')}\n\n"
        f"The new shot becomes shot {position}.\n"
    )
    if brief.strip():
        user_prompt += f"\nUSER'S BRIEF for the new shot (follow it): {brief.strip()}\n"
    else:
        user_prompt += (
            "\nNo brief given — invent the most dramatic connective beat: an "
            "establishing wide, a reaction closeup, or an insert detail shot.\n"
        )
    user_prompt += "\nNow output the JSON for the new shot."

    raw = sg._call_llm(user_prompt, _build_new_shot_system_prompt())
    shot = sg._extract_json(raw)
    if not isinstance(shot, dict):
        raise ValueError("LLM did not return a JSON object")
    return shot


def _fallback_shot(brief: str) -> dict:
    """Minimal hand-built shot when the LLM is down. Brief becomes the content."""
    text = brief.strip() or "a quiet beat as the scene settles"
    return {
        "beat": "observation",
        "shot_type": "medium",
        "narration": text if len(text.split()) <= 18 else " ".join(text.split()[:18]),
        "visual_description": text,
        "first_frame_prompt": text,
        "last_frame_prompt": text,
        "motion_prompt": "Gentle, subtle movement. Slow push-in. No speech. No mouth movement.",
    }


def _sanitize_new_shot(shot: dict, position: int) -> dict:
    """Clamp fields to valid values + required keys."""
    out = {
        "shot_number": position,
        "beat": (shot.get("beat") or "").strip().lower(),
        "shot_type": (shot.get("shot_type") or "").strip().lower(),
        "narration": (shot.get("narration") or "").strip(),
        "visual_description": (shot.get("visual_description") or "").strip(),
        "first_frame_prompt": (shot.get("first_frame_prompt") or "").strip(),
        "last_frame_prompt": (shot.get("last_frame_prompt") or "").strip(),
        "motion_prompt": (shot.get("motion_prompt") or "").strip(),
    }
    if out["beat"] not in _VALID_BEATS:
        out["beat"] = "observation"
    if out["shot_type"] not in _VALID_SHOT_TYPES:
        out["shot_type"] = "medium"
    if not out["narration"]:
        out["narration"] = out["visual_description"] or "The moment lingers quietly."
    if not out["first_frame_prompt"]:
        out["first_frame_prompt"] = out["visual_description"] or out["narration"]
    if not out["last_frame_prompt"]:
        out["last_frame_prompt"] = out["first_frame_prompt"]
    if not out["motion_prompt"]:
        out["motion_prompt"] = "Gentle, subtle movement. Slow push-in."
    return out


# ==============================================================================
# ARTIFACT SHIFTING (descending renames, highest shot first)
# ==============================================================================

def _shift_json_keys(path: Path, position: int, what: str) -> bool:
    """Shift integer-string keys >= position up by 1 in a flat dict file."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"shift {what}: could not parse {path}: {e}")
        return False
    if not isinstance(data, dict):
        return False
    new = {}
    for k, v in data.items():
        try:
            n = int(k)
        except (TypeError, ValueError):
            new[k] = v
            continue
        new[str(n + 1) if n >= position else k] = v
    atomic_write_json(path, new)
    return True


def _shift_approved_prompts(script_id: str, position: int) -> Optional[dict]:
    """Shift prompts keys >= position. Returns the updated state dict or None."""
    path = APPROVED_PROMPTS_DIR / f"{script_id}.json"
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"shift approved_prompts: could not parse {path}: {e}")
        return None
    prompts = state.get("prompts") or {}
    new_prompts = {}
    for k, v in prompts.items():
        try:
            n = int(k)
        except (TypeError, ValueError):
            new_prompts[k] = v
            continue
        new_prompts[str(n + 1) if n >= position else k] = v
    state["prompts"] = new_prompts
    atomic_write_json(path, state)
    return state


def _shift_storyboard(script_id: str, position: int, max_shot: int) -> int:
    """Rename shot{N}_*.png descending + fix manifest. Returns renames done."""
    story_dir = STORYBOARDS_DIR / script_id
    renamed = 0
    if story_dir.exists():
        for n in range(max_shot, position - 1, -1):
            for frame_type in ("first", "last"):
                src = story_dir / f"shot{n}_{frame_type}.png"
                if src.exists():
                    dst = story_dir / f"shot{n + 1}_{frame_type}.png"
                    if dst.exists():
                        dst.unlink()  # stale leftover from an older edit
                    src.rename(dst)
                    renamed += 1

    man_path = story_dir / "storyboard.json"
    if man_path.exists():
        try:
            md = json.loads(man_path.read_text(encoding="utf-8"))
            for fr in md.get("frames", []):
                sn = fr.get("shot_number")
                if isinstance(sn, int) and sn >= position:
                    fr["shot_number"] = sn + 1
                    ip = fr.get("image_path") or ""
                    fr["image_path"] = ip.replace(f"shot{sn}_", f"shot{sn + 1}_")
            md["total_frames"] = len(md.get("frames", []))
            atomic_write_json(man_path, md)
        except Exception as e:
            log.warning(f"shift storyboard manifest failed for {script_id}: {e}")
    return renamed


_CLIP_RE_TMPL = r"^clip_{sid}_shot(\d+)(.*)\.mp4$"


def _shift_clips(script_id: str, position: int) -> int:
    """Rename clip_{sid}_shot{N}[suffix].mp4 descending. Returns renames done."""
    if not CLIPS_DIR.exists():
        return 0
    pat = re.compile(_CLIP_RE_TMPL.format(sid=re.escape(script_id)))
    matches = []  # (shot_num, suffix, path)
    for p in CLIPS_DIR.glob(f"clip_{script_id}_shot*.mp4"):
        m = pat.match(p.name)
        if m:
            matches.append((int(m.group(1)), m.group(2), p))
    renamed = 0
    # Highest shot first so the rename target never exists yet
    for n, suffix, p in sorted(matches, key=lambda t: t[0], reverse=True):
        if n < position:
            continue
        dst = CLIPS_DIR / f"clip_{script_id}_shot{n + 1}{suffix}.mp4"
        if dst.exists():
            dst.unlink()
        p.rename(dst)
        renamed += 1
    return renamed


# ==============================================================================
# NEW-SHOT PROMPTS ENTRY (same pattern as prompt_approval.generate_all_prompts)
# ==============================================================================

def _make_prompts_entry(script: dict, shot: dict) -> dict:
    style_id = script.get("style") or sg.get_default_style()
    style_suffix = sg.get_style_description(style_id).get("prompt_suffix", "")
    shot_num = shot.get("shot_number")

    try:
        img_p = pa.assemble_image_prompt(
            script=script, shot=shot,
            style_suffix=style_suffix, frame_type="first",
        )
    except Exception as e:
        log.warning(f"new shot {shot_num} image prompt LLM failed: {e}; using raw")
        img_p = (shot.get("first_frame_prompt") or shot.get("visual_description") or "").strip()

    try:
        mot_p = pa.assemble_motion_prompt(
            script=script, shot=shot, approved_image_prompt=img_p,
        )
    except Exception as e:
        log.warning(f"new shot {shot_num} motion prompt LLM failed: {e}; using raw")
        mot_p = (shot.get("motion_prompt") or shot.get("visual_description") or "").strip()
        if "no speech" not in mot_p.lower():
            mot_p = mot_p.rstrip(".") + ". No speech. No mouth movement."

    return {
        "image_prompt": img_p,
        "image_seed": -1,
        "motion_prompt": mot_p,
        "motion_seed": -1,
        "beat": (shot.get("beat") or "").strip().lower() or "observation",
        "narration": (shot.get("narration") or "").strip(),
        "approved": False,
    }


# ==============================================================================
# PUBLIC API
# ==============================================================================

def insert_shot(script_id: str, position: int, brief: str = "") -> dict:
    """Insert a new LLM-written shot so it becomes shot `position` (1-based);
    every existing shot at/after that position shifts +1, and all on-disk
    artifacts (approved prompts, storyboard PNGs + manifest, clips, seeds)
    are renamed to follow their shots.

    Returns: {"script", "new_shot", "shifted_shots", "renamed_frames",
              "renamed_clips", "prompts_entry_added"}
    """
    script_path = SCRIPTS_DIR / f"script_{script_id}.json"
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    script = json.loads(script_path.read_text(encoding="utf-8"))

    shots = script.get("shots", []) or []
    if not shots:
        raise ValueError(f"Script {script_id} has no shots")
    max_shot = max(int(s.get("shot_number", 0)) for s in shots)
    position = max(1, min(int(position), max_shot + 1))

    # 1. Write the new shot (LLM, fallback to brief)
    try:
        raw_shot = _generate_shot_llm(script, position, brief)
    except Exception as e:
        log.warning(f"insert_shot: LLM generation failed ({e}); using fallback shot")
        raw_shot = _fallback_shot(brief)
    new_shot = _sanitize_new_shot(raw_shot, position)

    # 2. Renumber script shots and insert
    shifted = 0
    for s in shots:
        n = int(s.get("shot_number", 0))
        if n >= position:
            s["shot_number"] = n + 1
            shifted += 1
    shots.append(new_shot)
    shots.sort(key=lambda s: int(s.get("shot_number", 0)))
    script["shots"] = shots
    script["_edited_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(script_path, script)
    log.info(
        f"insert_shot: {script_id} new shot {position} "
        f"({new_shot['beat']}/{new_shot['shot_type']}), {shifted} shots shifted"
    )

    # 3. Shift artifacts (all best-effort; renames are descending = collision-free)
    state = _shift_approved_prompts(script_id, position)
    renamed_frames = _shift_storyboard(script_id, position, max_shot)
    renamed_clips = _shift_clips(script_id, position)
    _shift_json_keys(CLIPS_DIR / f"clip_{script_id}_seeds.json", position, "seeds")

    # 4. Add an unapproved prompts entry for the new shot (if a prompts file exists)
    prompts_added = False
    if state is not None:
        try:
            state["prompts"][str(position)] = _make_prompts_entry(script, new_shot)
            atomic_write_json(APPROVED_PROMPTS_DIR / f"{script_id}.json", state)
            prompts_added = True
        except Exception as e:
            log.warning(f"insert_shot: could not add prompts entry: {e}")

    # Invalidate prompt_approval's in-memory cache so its views reload from disk
    try:
        from modules import prompt_approval as pap
        pap._ACTIVE_STATES.pop(script_id, None)
    except Exception:
        pass

    return {
        "script": script,
        "new_shot": new_shot,
        "shifted_shots": shifted,
        "renamed_frames": renamed_frames,
        "renamed_clips": renamed_clips,
        "prompts_entry_added": prompts_added,
    }
