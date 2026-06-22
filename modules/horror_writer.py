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

# Pacing: narration ~150 wpm. 30 min ≈ 4500 words.
WORDS_PER_MINUTE = 150
# Per-image beat sizing: ~35-45 words ≈ ~15-18s of slow narration per still.
WORDS_PER_BEAT = 40
MAX_BEATS = 130            # hard cap so render time can't run away
MIN_BEAT_WORDS = 8         # drop empty/too-short fragments

# One designed deep narrator voice, reused for the whole story (VoxCPM2).
DEFAULT_VOICE_DESIGN = (
    "a deep, slow, gravelly older male voice, ominous and hushed, with deliberate "
    "pacing and a haunting, suspenseful horror-narrator tone"
)


def _outline_system() -> str:
    return """You output ONLY valid JSON. No prose, no preamble, no markdown fences. Start with { end with }.

You are a master horror writer plotting a ~30-minute narrated horror story for a MATURE audience (teens and adults). You design dread, suspense, and a real arc — slow build, escalating wrongness, a turn, a chilling payoff.

Write an OUTLINE only (no full prose yet):
{
  "title": "<3-7 word eerie title>",
  "logline": "<one-sentence hook>",
  "setting": "<where/when, one sentence>",
  "chapters": [
    {"title": "<short>", "summary": "<2-3 sentences of what happens in this chapter; advances the dread>"}
  ]
}

Rules:
- 9 to 11 chapters, ordered: ordinary-but-uneasy opening → first wrong sign → investigation/descent → escalation → revelation/turn → horrifying climax → bleak or twist ending.
- Keep it grounded psychological/supernatural horror. Allowed: fear, death, monsters, ghosts, blood, violence in a FICTIONAL horror context.
- FORBIDDEN: sexual content, sexualizing minors, real self-harm/suicide how-to, glorifying real drugs.
- Output ONLY the JSON object."""


def _chapter_system() -> str:
    return f"""You output ONLY valid JSON. No prose, no preamble, no markdown fences. Start with {{ end with }}.

You are writing ONE chapter of a narrated horror story, broken into short IMAGE BEATS for a video. A narrator reads the `narration`; each beat shows ONE photorealistic still.

Output:
{{
  "beats": [
    {{
      "narration": "<{WORDS_PER_BEAT-10}-{WORDS_PER_BEAT+15} words of vivid, atmospheric horror narration, complete sentences, third person, present or past tense, no dialogue tags. This is read ALOUD — make it flow.>",
      "image_prompt": "<25-45 words describing ONE photorealistic, cinematic horror still that matches THIS beat: concrete subject + location + lighting + mood + camera. No on-screen text. Do NOT name an art style — 'photorealistic' is added automatically.>"
    }}
  ]
}}

Rules:
- 6 to 9 beats for this chapter.
- Each beat's narration continues smoothly from the previous; together the beats fully narrate this chapter.
- Dread over gore. Fictional horror (fear, death, monsters, blood) is allowed; NO sexual content, NO real self-harm instructions, NO sexualizing minors.
- image_prompt must be appropriate (suggestive horror, not extreme gore).
- Output ONLY the JSON object."""


def _gen_outline(theme: str) -> dict:
    raw = _call_llm(f"Theme: {theme}\n\nWrite the outline JSON.", _outline_system(), role="creative")
    data = _extract_json(raw)
    chapters = [c for c in data.get("chapters", []) if isinstance(c, dict) and c.get("summary")]
    if len(chapters) < 4:
        raise ValueError(f"Outline too short ({len(chapters)} chapters).")
    data["chapters"] = chapters
    data.setdefault("title", "Untitled Horror")
    data.setdefault("logline", "")
    data.setdefault("setting", "")
    return data


def _gen_chapter_beats(theme: str, logline: str, setting: str,
                       prev_summary: str, chapter: dict) -> list[dict]:
    user = (
        f"Story theme: {theme}\n"
        f"Logline: {logline}\n"
        f"Setting: {setting}\n"
        f"Previous chapter: {prev_summary or '(this is the opening)'}\n\n"
        f"THIS chapter — {chapter.get('title','')}: {chapter.get('summary','')}\n\n"
        f"Write this chapter's beats JSON now."
    )
    raw = _call_llm(user, _chapter_system(), role="creative")
    data = _extract_json(raw)
    beats = []
    for b in data.get("beats", []):
        if not isinstance(b, dict):
            continue
        narr = (b.get("narration") or "").strip()
        img = (b.get("image_prompt") or "").strip()
        if len(narr.split()) >= MIN_BEAT_WORDS and img:
            beats.append({"narration": narr, "image_prompt": img})
    return beats


def generate_horror_story(
    theme: str,
    target_minutes: int = 30,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Write a long-form horror story (chunked) and save it. Returns the dict."""
    def _p(msg: str):
        log.info(msg)
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    _job_start = _t.time()
    target_words = int(target_minutes * WORDS_PER_MINUTE)

    _p("✍️ plotting outline...")
    outline = _gen_outline(theme)
    chapters = outline["chapters"]
    _p(f"📖 outline: '{outline['title']}' — {len(chapters)} chapters")

    beats: list[dict] = []
    prev = ""
    for i, ch in enumerate(chapters):
        _p(f"✍️ writing chapter {i+1}/{len(chapters)}: {ch.get('title','')}")
        try:
            ch_beats = _gen_chapter_beats(
                theme, outline["logline"], outline["setting"], prev, ch
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

    if len(beats) < 6:
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
