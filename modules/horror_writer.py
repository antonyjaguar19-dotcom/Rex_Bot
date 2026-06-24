"""
Claw Bot — Horror Story Writer (long-form, chunked)

Writes a ~30-minute horror narration with Qwen. A single LLM call can't produce
~4500 coherent words, so generation is CHUNKED:
  Stage 1: outline  → title, logline, ordered chapters (summaries).
  Stage 2: per chapter → narration split into image-scenes, each with a
           photorealistic image_prompt (the cadence = one image per narration beat).

Output JSON (04_Outputs/horror/horror_{id}.json):
  { title, logline, theme, voice_design, target_minutes,
    beats: [ {narration, image_prompt}, ... ] }

`beats` are the unit the pipeline renders: VoxCPM voices each beat's narration
(its real audio duration becomes the scene length) and one photoreal still is
generated per beat. No character consistency — coherence comes from the prose.
"""

import json
import logging
import sys
import time as _t
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import model_registry
from modules import safety_filter as sf
from modules import generation_meta as gm
from modules.file_utils import atomic_write_json
from modules.script_generator import _call_llm, _extract_json

log = logging.getLogger("claw_bot.horror_writer")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
OUTPUTS_DIR = PROJECT_ROOT / "04_Outputs" / "horror"

# Pacing: narration ~150 wpm. Default target is SHORT + punchy (5 min ≈ 750 words).
WORDS_PER_MINUTE = 150
DEFAULT_MINUTES = 2        # max ~2-min horror videos (tight, scary, animated)
# Per-shot beat sizing: ~45-60 words ≈ ~20-26s of narration per shot.
WORDS_PER_BEAT = 52
MERGE_MIN_WORDS = 30       # merge consecutive beats below this into one
MAX_BEATS = 10             # ~2-min cap (each beat -> a Wan clip, so keep it tight)
MIN_BEAT_WORDS = 8         # drop empty/too-short fragments

# One designed deep narrator voice, reused for the whole story (VoxCPM2).
# Kept CLEAR + calm — "gravelly/hushed" muddied intelligibility. The pipeline
# anchors this once and clones it for every beat so the voice never drifts.
DEFAULT_VOICE_DESIGN = (
    "a deep, calm, clear adult male voice with measured, deliberate pacing and a "
    "low ominous horror-narrator tone"
)


def _premise_system() -> str:
    return """You output ONLY valid JSON. No prose, no preamble, no markdown fences. Start with { end with }.

You are a master horror writer (think creepypasta / r-nosleep / Junji Ito) creating the premise + cast for a SHORT (~5 minute) narrated horror story for ADULTS. It must be GENUINELY FRIGHTENING — not mild, not whimsical. Build real dread: an ordinary moment that turns deeply wrong, a creeping sense of threat, a horrifying revelation, a bleak or gut-punch ending. Visceral, disturbing, original fiction.

Output:
{
  "title": "<3-7 word eerie title>",
  "logline": "<one-sentence hook>",
  "setting": "<where/when, one sentence>",
  "characters": [
    {
      "name": "<ONE short proper name, e.g. Mara — no title/species>",
      "gender": "male | female",
      "age": "<explicit, e.g. '34-year-old' or 'elderly'>",
      "appearance": "<one sentence: face, hair, build, typical clothing — concrete + photoreal; NO actions, NO setting>",
      "locked_token": "<tight 15-25 word phrase: age + gender + face + hair + clothing colors + ONE distinctive feature, pasteable into every image prompt>"
    }
  ],
  "locations": [
    {"name": "<short, e.g. The Lantern Room>", "description": "<one sentence: concrete photoreal look of this place + its lighting/mood>"}
  ]
}

Rules:
- 1 to 2 recurring named characters (tiny cast = scarier + consistent visuals).
- 2 to 4 distinct locations.
- Genuinely scary psychological/supernatural horror: fear, dread, death, monsters, ghosts, body-horror, blood, fictional violence are ALL allowed and encouraged for real scares.
- Keep it ORIGINAL fiction — invent the people and events; never depict real, named, living people.
- FORBIDDEN: sexual content, sexualizing minors, real self-harm/suicide how-to, glorifying real drugs.
- Output ONLY the JSON object."""


def _chapters_system(n_chapters: int) -> str:
    return f"""You output ONLY valid JSON. No prose, no preamble, no markdown fences. Start with {{ end with }}.

Given a horror premise + cast, plot the CHAPTERS of a SHORT, intense ~5-minute horror story:
{{
  "chapters": [
    {{"title": "<short>", "summary": "<1-2 sentences of what happens; ratchets the dread; name which cast members appear>"}}
  ]
}}

Rules:
- Exactly {n_chapters} chapters, ordered: ordinary-but-wrong opening → first scare → escalation → horrifying revelation → bleak/gut-punch ending. Tight and relentless — every chapter raises the fear.
- Use the given character + location names.
- Output ONLY the JSON object."""


def _chapter_system(cast: str, locs: str, lo: int, hi: int) -> str:
    return f"""You output ONLY valid JSON. No prose, no preamble, no markdown fences. Start with {{ end with }}.

You are writing ONE chapter of a narrated horror story, broken into IMAGE BEATS for a video. A narrator reads the `narration`; each beat shows ONE photorealistic still.

CAST (use these exact names): {cast}
LOCATIONS (use these exact names): {locs}

Output:
{{
  "beats": [
    {{
      "narration": "<{WORDS_PER_BEAT-12}-{WORDS_PER_BEAT+13} words of vivid, atmospheric horror narration: 2-4 COMPLETE flowing sentences, third person, no dialogue tags. Read ALOUD by one narrator — smooth and clear, never a choppy fragment.>",
      "characters": ["<names of cast members visible in this beat, [] if none>"],
      "location": "<exactly one location NAME from the list above>",
      "image_prompt": "<25-45 words: which CHARACTER (by exact name) is present and what they do + what we see + camera + lighting/mood. No on-screen text. Do NOT name an art style — handled downstream.>",
      "motion_prompt": "<15-30 words: ONE subtle eerie motion + ONE slow camera move for this shot (e.g. 'shadow creeps across the wall as the camera pushes in slowly'). Restrained, dread-building, no fast action.>"
    }}
  ]
}}

Rules:
- {lo} to {hi} beats for this chapter — this is a SHORT, punchy ~5-minute story, so keep it tight.
- Each beat's narration continues smoothly from the previous; together they fully narrate this chapter.
- Make it genuinely scary: build dread, then hit hard. Real fear, threat, disturbing imagery.
- Refer to characters by their EXACT cast name in narration AND image_prompt (identity is locked downstream by name).
- `location` MUST be one of the given location names.
- Fictional horror (fear, death, monsters, body-horror, blood) is allowed; NO sexual content, NO real self-harm instructions, NO sexualizing minors.
- Output ONLY the JSON object."""


def _gen_outline(theme: str, n_chapters: int = 5) -> dict:
    # Two calls — one prompt can't reliably emit bible AND chapters in one go.
    # Premise+cast first, then chapters.
    data = _extract_json(_call_llm(
        f"Theme: {theme}\n\nWrite the premise JSON.", _premise_system(), role="creative"))
    cast_names = ", ".join(c.get("name", "") for c in data.get("characters", []) if isinstance(c, dict))
    loc_names = ", ".join(l.get("name", "") for l in data.get("locations", []) if isinstance(l, dict))
    ch_user = (
        f"Theme: {theme}\n"
        f"Title: {data.get('title','')}\n"
        f"Logline: {data.get('logline','')}\n"
        f"Setting: {data.get('setting','')}\n"
        f"Characters: {cast_names or '(none)'}\n"
        f"Locations: {loc_names or '(none)'}\n\n"
        f"Now write the chapters JSON."
    )
    ch_data = _extract_json(_call_llm(ch_user, _chapters_system(n_chapters), role="creative"))
    chapters = [c for c in ch_data.get("chapters", []) if isinstance(c, dict) and c.get("summary")]
    if len(chapters) < 3:
        raise ValueError(f"Outline too short ({len(chapters)} chapters).")
    data["chapters"] = chapters
    data.setdefault("title", "Untitled Horror")
    data.setdefault("logline", "")
    data.setdefault("setting", "")
    # Normalize bible
    chars = []
    for c in data.get("characters", []):
        if isinstance(c, dict) and (c.get("name") or "").strip():
            c["name"] = c["name"].strip()
            c.setdefault("locked_token", c.get("appearance", "")[:150])
            chars.append(c)
    data["characters"] = chars
    locs = []
    for lc in data.get("locations", []):
        if isinstance(lc, dict) and (lc.get("name") or "").strip():
            lc["name"] = lc["name"].strip()
            locs.append(lc)
    data["locations"] = locs
    return data


def _gen_chapter_beats(theme: str, logline: str, setting: str, prev_summary: str,
                       chapter: dict, cast: list, locations: list,
                       beats_lo: int = 2, beats_hi: int = 3) -> list[dict]:
    cast_str = ", ".join(c["name"] for c in cast) or "(none — atmospheric)"
    loc_str = ", ".join(lc["name"] for lc in locations) or "(unnamed)"
    user = (
        f"Story theme: {theme}\n"
        f"Logline: {logline}\n"
        f"Setting: {setting}\n"
        f"Previous chapter: {prev_summary or '(this is the opening)'}\n\n"
        f"THIS chapter — {chapter.get('title','')}: {chapter.get('summary','')}\n\n"
        f"Write this chapter's beats JSON now."
    )
    raw = _call_llm(user, _chapter_system(cast_str, loc_str, beats_lo, beats_hi), role="creative")
    data = _extract_json(raw)
    valid_locs = {lc["name"] for lc in locations}
    beats = []
    for b in data.get("beats", []):
        if not isinstance(b, dict):
            continue
        narr = (b.get("narration") or "").strip()
        img = (b.get("image_prompt") or "").strip()
        if len(narr.split()) >= MIN_BEAT_WORDS and img:
            loc = (b.get("location") or "").strip()
            chs = [c for c in (b.get("characters") or []) if isinstance(c, str)]
            beats.append({
                "narration": narr, "image_prompt": img,
                "motion_prompt": (b.get("motion_prompt") or "").strip()
                                 or "slow ominous camera push-in, faint shadows shifting",
                "location": loc if loc in valid_locs else "",
                "characters": chs,
            })
    return beats


def _merge_short_beats(beats: list[dict], min_words: int = MERGE_MIN_WORDS) -> list[dict]:
    """Merge consecutive too-short beats so each spoken beat is a full, clear unit
    (choppy fragments hurt TTS clarity). Merged narration joins; first image_prompt
    is kept (one still per merged beat)."""
    merged: list[dict] = []
    for b in beats:
        if merged and len(merged[-1]["narration"].split()) < min_words:
            prev = merged[-1]
            prev["narration"] = (prev["narration"].rstrip() + " " + b["narration"].strip()).strip()
            # keep prev image_prompt; drop this beat's (fewer, longer holds)
        else:
            merged.append(dict(b))
    return merged


def generate_horror_story(
    theme: str,
    target_minutes: int = DEFAULT_MINUTES,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Write a short, scary horror story (chunked) and save it. Returns the dict."""
    def _p(msg: str):
        log.info(msg)
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    _job_start = _t.time()
    target_minutes = max(1, min(int(target_minutes or DEFAULT_MINUTES), 3))
    target_words = int(target_minutes * WORDS_PER_MINUTE)
    # Scale structure to length: ~1 chapter/min, ~2-3 beats each, capped tight.
    n_chapters = max(2, min(target_minutes + 1, 4))
    total_beats_target = max(5, min(round(target_words / WORDS_PER_BEAT), MAX_BEATS))
    beats_lo = max(2, total_beats_target // n_chapters)
    beats_hi = beats_lo + 1

    _p("✍️ plotting outline...")
    outline = _gen_outline(theme, n_chapters)
    chapters = outline["chapters"]
    cast = outline.get("characters", [])
    locations = outline.get("locations", [])
    _p(f"📖 outline: '{outline['title']}' — {len(chapters)} chapters, "
       f"{len(cast)} characters, {len(locations)} locations")

    beats: list[dict] = []
    prev = ""
    for i, ch in enumerate(chapters):
        _p(f"✍️ writing chapter {i+1}/{len(chapters)}: {ch.get('title','')}")
        try:
            ch_beats = _gen_chapter_beats(
                theme, outline["logline"], outline["setting"], prev, ch,
                cast, locations, beats_lo, beats_hi,
            )
        except Exception as e:
            log.warning(f"Chapter {i+1} failed ({e}); skipping.")
            ch_beats = []
        beats.extend(ch_beats)
        prev = ch.get("summary", "")
        if len(beats) >= MAX_BEATS:
            log.warning(f"Hit MAX_BEATS={MAX_BEATS}; stopping chapter expansion.")
            beats = beats[:MAX_BEATS]
            break

    beats = _merge_short_beats(beats)
    if len(beats) > MAX_BEATS:
        beats = beats[:MAX_BEATS]
    if len(beats) < 5:
        raise ValueError(f"Horror story produced only {len(beats)} beats — too short.")

    word_total = sum(len(b["narration"].split()) for b in beats)
    _p(f"📝 story written: {len(beats)} beats, ~{word_total} words "
       f"(~{word_total/WORDS_PER_MINUTE:.0f} min). Target ~{target_words}.")

    story = {
        "title": outline["title"],
        "logline": outline["logline"],
        "setting": outline["setting"],
        "theme": theme,
        "voice_design": DEFAULT_VOICE_DESIGN,
        "target_minutes": target_minutes,
        "characters": cast,        # bible: recurring chars (LoRA + look)
        "locations": locations,    # bible: environments
        "beats": beats,
    }

    # Adult safety profile — horror needs violence/dread vocabulary.
    is_safe, blocked, warnings = sf.check_safety(story, profile="adult")
    if not is_safe:
        raise ValueError(f"Horror story hit adult hard-blocks: {blocked}. Adjust the theme.")
    if warnings:
        log.info(f"Horror soft warnings: {warnings}")

    now = datetime.now()
    horror_id = now.strftime("%Y%m%d_%H%M%S")
    story["_id"] = horror_id
    story["horror_id"] = horror_id
    story["_generated_at"] = now.isoformat()
    story["_kind"] = "horror"

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(OUTPUTS_DIR / f"horror_{horror_id}.json", story)
    _p(f"💾 saved horror_{horror_id}.json")

    try:
        llm_cfg = model_registry.get_active("llm_backend")
        llm_id = llm_cfg.get("_id", "unknown") if llm_cfg else "unknown"
    except Exception:
        llm_id = "unknown"
    gm.record({
        "kind": "horror", "script_id": horror_id,
        "duration_sec": round(_t.time() - _job_start, 1), "vram_peak_mb": 0,
        "settings": {"backend_id": llm_id, "beats": len(beats),
                     "words": word_total, "title": story["title"][:60]},
        "success": True,
    })
    return story


def load_horror(horror_id: str) -> Optional[dict]:
    path = OUTPUTS_DIR / f"horror_{horror_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import sys as _sys
    theme = _sys.argv[1] if len(_sys.argv) > 1 else "an abandoned lighthouse that calls people into the sea"
    mins = int(_sys.argv[2]) if len(_sys.argv) > 2 else 5
    s = generate_horror_story(theme, target_minutes=mins, progress_cb=print)
    print(f"\n{s['title']} — {len(s['beats'])} beats")
    print(s['beats'][0])
