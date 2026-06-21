"""
Claw Bot — Song Generator (music-video pipeline)

Qwen writes a complete song from a theme: lyrics (section-tagged for ACE-Step),
a musical style, a vocal type, ACE-Step style tags, bpm/key/duration, AND a
storyboard of mood-fit visual scenes (Ken Burns stills).

Mirrors script_generator.py: same Ollama call (_call_llm), JSON extraction,
safety filter, atomic save. Output saved to 04_Outputs/songs/song_{id}.json.

Visual style values map to styles.json entries:
  cartoon -> cartoon, doodle -> stickman, spectrum -> spectrum, photoreal -> photoreal
"""

import json
import logging
import re
import sys
import time as _t
from datetime import datetime
from typing import Optional

from pathlib import Path

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import model_registry
from modules import safety_filter as sf
from modules import runtime_settings as rs
from modules import generation_meta as gm
from modules.file_utils import atomic_write_json
# Reuse the battle-tested LLM call + JSON extraction from script_generator.
from modules.script_generator import _call_llm, _extract_json

log = logging.getLogger("claw_bot.song_generator")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
OUTPUTS_DIR = PROJECT_ROOT / "04_Outputs" / "songs"

# ~8s of music per Ken Burns still. 120s song -> ~15 scenes.
SECONDS_PER_SCENE = 8.0
DEFAULT_DURATION = 120

SONG_STYLES = list(rs.VALID_SONG_STYLES)
VOCAL_TYPES = list(rs.VALID_VOCAL_TYPES)
VISUAL_STYLES = list(rs.VALID_VISUAL_STYLES)

# visual_style (user-facing) -> styles.json id used to pull a prompt_suffix.
VISUAL_STYLE_TO_STYLE_ID = {
    "cartoon": "cartoon",
    "doodle": "stickman",
    "spectrum": "spectrum",
    "photoreal": "photoreal",
}


def _build_system_prompt(duration_sec: int, n_scenes: int) -> str:
    return f"""You output ONLY valid JSON. No prose, no preamble, no markdown fences. Start with {{ end with }}.

You are a songwriter + music-video director. From a THEME you write a complete, original song and plan its music video.

# AUDIENCE
This is for BOTH kids and adults. Keep it tasteful and appropriate: no explicit sex, no slurs, no graphic violence, no drugs/alcohol glorification, no hate. Emotion, energy, heartbreak, hope, fun — all fine. Think radio-clean.

# SONG
- Write ORIGINAL lyrics (never copy real songs). Structure them with section tags ACE-Step understands, each on its own line in the lyrics string:
  [Intro] [Verse 1] [Pre-Chorus] [Chorus] [Verse 2] [Bridge] [Outro]
- Total song length is about {duration_sec} seconds — write enough lyrics to fill it (roughly 2 verses + repeated chorus). Use real line breaks inside the lyrics string (\\n).
- Pick a `song_style` from: {', '.join(SONG_STYLES)}.
- Pick a `vocal_type` from: {', '.join(VOCAL_TYPES)} (use "instrumental" only if the song truly has no singing).
- Write `ace_tags`: a short comma-separated ACE-Step style prompt describing genre + instrumentation + vocal, e.g. "upbeat pop, female vocal, bright synths, punchy drums". Include the vocal_type in the tags.
- Pick `bpm` (integer 60-180 fitting the style) and `keyscale` (e.g. "C major", "A minor").

# VISUALS
- Pick `visual_style` from: {', '.join(VISUAL_STYLES)} — choose the look that FITS the song mood (soft/romantic -> photoreal or cartoon; energetic/electronic -> spectrum; playful/funny -> doodle or cartoon). NO character consistency is needed; scenes do not need recurring people.
- Write `visual_world`: ONE sentence describing the shared look/palette so the scenes feel coherent.
- Write exactly {n_scenes} `scenes`. Each scene covers ~{SECONDS_PER_SCENE:.0f} seconds of the song and has:
  - `section`: which song part it illustrates (e.g. "verse 1", "chorus").
  - `image_prompt`: 30-50 words describing ONE vivid still image that fits the lyric mood. Concrete subject + setting + lighting + composition. Do NOT name a style ("cartoon"/"photo") — the style is added automatically. Keep it appropriate for all ages.
- Scenes should track the song's emotional arc start to finish.

# OUTPUT FORMAT
{{
  "title": "<3-6 words>",
  "theme": "<the user's theme>",
  "song_style": "{SONG_STYLES[0]}",
  "vocal_type": "{VOCAL_TYPES[1]}",
  "ace_tags": "<comma-separated style tags incl. vocal>",
  "bpm": 120,
  "keyscale": "C major",
  "language": "en",
  "duration_sec": {duration_sec},
  "visual_style": "{VISUAL_STYLES[0]}",
  "visual_world": "<one sentence>",
  "lyrics": "[Verse 1]\\n...\\n[Chorus]\\n...",
  "scenes": [
    {{"section": "intro", "image_prompt": "..."}}
  ]
}}

Output ONLY the JSON object. Start with {{ end with }}. Nothing else."""


def _validate_and_default(song: dict, duration_sec: int, n_scenes: int) -> dict:
    song.setdefault("title", "Untitled Song")
    song.setdefault("theme", "")
    song.setdefault("language", "en")
    song.setdefault("visual_world", "")
    song.setdefault("lyrics", "")

    # song_style
    style = (song.get("song_style") or "").strip().lower()
    song["song_style"] = style if style in SONG_STYLES else "pop"

    # vocal_type
    vocal = (song.get("vocal_type") or "").strip().lower()
    song["vocal_type"] = vocal if vocal in VOCAL_TYPES else "auto"

    # visual_style
    vis = (song.get("visual_style") or "").strip().lower()
    song["visual_style"] = vis if vis in VISUAL_STYLES else "cartoon"

    # numeric fields
    try:
        song["bpm"] = int(song.get("bpm", 120))
    except (ValueError, TypeError):
        song["bpm"] = 120
    song["bpm"] = max(50, min(200, song["bpm"]))
    if not (song.get("keyscale") or "").strip():
        song["keyscale"] = "C major"
    song["duration_sec"] = int(duration_sec)

    if not (song.get("ace_tags") or "").strip():
        song["ace_tags"] = f"{song['song_style']}, {song['vocal_type']} vocal"

    # scenes — must be a non-empty list of dicts with image_prompt
    scenes = song.get("scenes") or []
    clean = []
    for sc in scenes:
        if isinstance(sc, dict) and (sc.get("image_prompt") or "").strip():
            clean.append({
                "section": (sc.get("section") or "").strip() or "scene",
                "image_prompt": sc["image_prompt"].strip(),
            })
    if not clean:
        raise ValueError("Song has no usable scenes (need at least one image_prompt).")
    # Pad/trim toward n_scenes so timing maths is stable.
    if len(clean) > n_scenes:
        clean = clean[:n_scenes]
    song["scenes"] = clean

    # Even per-scene seconds; remainder padded onto the last scene.
    per = round(duration_sec / len(clean), 2)
    for sc in clean:
        sc["seconds"] = per
    drift = round(duration_sec - per * len(clean), 2)
    if abs(drift) > 0.01:
        clean[-1]["seconds"] = round(clean[-1]["seconds"] + drift, 2)

    return song


def generate_song(theme: str, duration_sec: Optional[int] = None) -> dict:
    """Generate a song JSON (lyrics + style + scenes) and save it to disk."""
    _job_start = _t.time()
    duration_sec = int(duration_sec or DEFAULT_DURATION)
    n_scenes = max(4, round(duration_sec / SECONDS_PER_SCENE))

    # User overrides steer the LLM but don't replace its creativity.
    style_ov = rs.get_song_style_override()
    vocal_ov = rs.get_vocal_type_override()
    visual_ov = rs.get_visual_style_override()

    user_prompt = f"Theme: {theme}"
    if style_ov:
        user_prompt += f"\nUse song_style: {style_ov}"
    if vocal_ov:
        user_prompt += f"\nUse vocal_type: {vocal_ov}"
    if visual_ov:
        user_prompt += f"\nUse visual_style: {visual_ov}"
    user_prompt += "\n\nNow output the song JSON."

    system_prompt = _build_system_prompt(duration_sec, n_scenes)
    log.info(f"Generating song for theme: '{theme[:80]}' ({duration_sec}s, {n_scenes} scenes)")
    raw = _call_llm(user_prompt, system_prompt, role="creative")
    song = _extract_json(raw)
    song = _validate_and_default(song, duration_sec, n_scenes)

    # Apply hard overrides post-hoc (user choice wins over LLM pick).
    if style_ov:
        song["song_style"] = style_ov
    if vocal_ov:
        song["vocal_type"] = vocal_ov
    if visual_ov:
        song["visual_style"] = visual_ov

    # Safety: scan lyrics + scene prompts. Reuse the script safety filter (it
    # walks string values of the dict).
    is_safe, blocked, warnings = sf.check_safety({
        "title": song.get("title", ""),
        "lyrics": song.get("lyrics", ""),
        "scenes": song.get("scenes", []),
    })
    if not is_safe:
        raise ValueError(
            f"Generated song contains hard-blocked terms: {blocked}. Try a different theme."
        )
    if warnings:
        log.info(f"Song generated with soft warnings: {warnings}")

    now = datetime.now()
    song_id = now.strftime("%Y%m%d_%H%M%S")
    song["_id"] = song_id
    song["song_id"] = song_id
    song["_generated_at"] = now.isoformat()
    song["_kind"] = "song"

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / f"song_{song_id}.json"
    atomic_write_json(output_path, song)
    log.info(f"Song saved: {output_path}")

    try:
        llm_cfg = model_registry.get_active("llm_backend")
        llm_id = llm_cfg.get("_id", "unknown") if llm_cfg else "unknown"
    except Exception:
        llm_id = "unknown"
    gm.record({
        "kind": "song",
        "script_id": song_id,
        "duration_sec": round(_t.time() - _job_start, 1),
        "vram_peak_mb": 0,
        "settings": {
            "backend_id": llm_id,
            "song_style": song.get("song_style"),
            "vocal_type": song.get("vocal_type"),
            "visual_style": song.get("visual_style"),
            "scenes": len(song.get("scenes", [])),
            "title": song.get("title", "")[:60],
        },
        "success": True,
    })
    return song


def load_song(song_id: str) -> Optional[dict]:
    path = OUTPUTS_DIR / f"song_{song_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_song(song: dict) -> Path:
    song_id = song.get("song_id") or song.get("_id")
    path = OUTPUTS_DIR / f"song_{song_id}.json"
    atomic_write_json(path, song)
    return path


def rewrite_lyrics(song: dict, instruction: str = "") -> dict:
    """AI-rewrite the song's lyrics (and refresh tags) per a free-text instruction.
    Returns the updated song dict (also saved to disk). Falls back to the original
    on LLM failure."""
    system_prompt = (
        "You rewrite the LYRICS of a song. Output ONLY valid JSON: "
        '{"lyrics": "...", "ace_tags": "..."}. Keep section tags like [Verse 1], '
        "[Chorus] on their own lines (use \\n). Audience is kids AND adults — keep "
        "it appropriate: no explicit content, slurs, or graphic violence."
    )
    parts = [
        f"Title: {song.get('title','')}",
        f"Song style: {song.get('song_style','')}",
        f"Vocal type: {song.get('vocal_type','')}",
        f"Current lyrics:\n{song.get('lyrics','')}",
        f"Rewrite instruction: {instruction.strip() or 'rephrase freshly, keep theme, structure and length'}",
        "\nOutput the JSON now.",
    ]
    try:
        raw = _call_llm("\n".join(parts), system_prompt, role="creative")
        out = _extract_json(raw)
    except Exception as e:
        log.warning(f"rewrite_lyrics failed: {e}")
        return song
    if isinstance(out.get("lyrics"), str) and out["lyrics"].strip():
        song["lyrics"] = out["lyrics"].strip()
    if isinstance(out.get("ace_tags"), str) and out["ace_tags"].strip():
        song["ace_tags"] = out["ace_tags"].strip()
    save_song(song)
    return song


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import sys as _sys
    theme = _sys.argv[1] if len(_sys.argv) > 1 else "a rainy night drive through the city"
    s = generate_song(theme, duration_sec=60)
    print(json.dumps(s, indent=2, ensure_ascii=False)[:2000])
