"""
Claw Bot — Audio-First Pipeline (Phase 2)

The reorder: voiceover is rendered FIRST, then the natural pauses in the speech
decide where the film cuts. Shot count FLOATS with the spoken rhythm.

Flow:
  theme
   → story_writer.write_story()            # prose (narrator + dialogue)
   → voice_and_segment()                   # 1 master VO wav + pause-bound segments
   → _structure_segments()                 # LLM assigns beat/shot_type/visuals per
                                            #   segment; narration stays VERBATIM
   → standard script JSON on disk          # downstream pipeline unchanged

MVP = single narrator voice (honours "pauses dictate count"). Per-character
voices + S2V lip-sync are a later layer — they fight a single master track.

TTS: VoxCPM primary (local, offline, Apache-2.0 — voiced per breath-group for
exact spans, no cloud/key). Falls back to local Kokoro + ffmpeg silencedetect if
VoxCPM is unavailable. Either way the pipeline keeps running, fully contained.

The script JSON gains:
  script["_audio_first"] = True
  script["_master_audio"] = "<abs path to master VO wav>"
  shot["win_start"/"win_end"]        — timeline slice this shot owns (seconds)
  shot["audio_t_start"/"audio_t_end"]— where its words sit in the master track
Assembly (Phase 3) reads these to lay the one master track over the cut timeline.
"""

import logging
import re
import sys
import time as _t
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

_AGENT_DIR = Path(__file__).parent.parent.resolve()
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from modules import story_writer as sw
from modules import narration_segmenter as nseg
from modules import script_generator as sg
from modules import runtime_settings as rs
from modules import safety_filter as sf
from modules import generation_meta as gm
from modules import model_registry
from modules.file_utils import atomic_write_json

log = logging.getLogger("claw_bot.audio_first")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SCRIPTS_DIR = PROJECT_ROOT / "04_Outputs" / "scripts"
AUDIO_DIR = PROJECT_ROOT / "04_Outputs" / "audio"


# ==============================================================================
# TTS SELECTION — ElevenLabs primary, local fallback
# ==============================================================================

def _split_sentences(prose: str) -> list[str]:
    """Rough sentence split for the silence-fallback path (text → chunk map)."""
    parts = re.split(r"(?<=[.!?])\s+", prose.strip())
    return [p.strip() for p in parts if p.strip()]


# Chars per breath-group ceiling. VoxCPM voices ≈ 0.075s/char, so ~70 chars
# ≈ 5s — Wan's native clip length. Longer groups get freeze-padded at assembly
# (visible frozen tail), so keep groups at/under the video ceiling.
_GROUP_MAX_CHARS = 70


def _breath_groups(prose: str, max_chars: int = _GROUP_MAX_CHARS) -> list[str]:
    """Split prose into breath-groups for per-group voicing (VoxCPM path).

    Each group becomes one shot. Sentences are the unit; a sentence longer than
    max_chars is split at its commas/clause breaks so no single voiced group
    exceeds the video model's ~5s clip ceiling. Never splits mid-clause.
    """
    groups: list[str] = []
    for sent in _split_sentences(prose):
        if len(sent) <= max_chars:
            groups.append(sent)
            continue
        # split long sentence at clause punctuation, greedily packing to max_chars
        clauses = re.split(r"(?<=[,;:])\s+", sent)
        buf = ""
        for cl in clauses:
            cl = cl.strip()
            if not cl:
                continue
            if buf and len(buf) + 1 + len(cl) > max_chars:
                groups.append(buf.strip())
                buf = cl
            else:
                buf = (buf + " " + cl).strip()
        if buf:
            groups.append(buf.strip())
    return [g for g in groups if g]


def voice_and_segment(
    prose: str,
    master_wav: Optional[Path] = None,
    voice_id: Optional[str] = None,
    progress_cb: Optional[Callable] = None,
) -> tuple[Path, list[nseg.Segment], str]:
    """Render the whole narration to one wav and split it at its pauses.

    Returns (master_wav_path, segments, engine_name). Primary path is VoxCPM
    (local, offline, Apache-2.0): each breath-group is voiced separately and
    stitched, giving EXACT spans (no detection). On any failure falls back to
    the local Kokoro engine + ffmpeg silencedetect so the pipeline never stops.
    """
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    if master_wav is None:
        stub = datetime.now().strftime("%Y%m%d_%H%M%S")
        master_wav = AUDIO_DIR / f"master_{stub}.wav"

    # --- VoxCPM (preferred): per-group voicing → exact spans ---
    try:
        from modules.tts_voxcpm import VoxCPMTTS
        vox = VoxCPMTTS()
        ok, msg = vox.health_check(load_model=False)
        if not ok:
            raise RuntimeError(msg)
        if progress_cb:
            progress_cb("voicing narration (VoxCPM)...")
        groups = _breath_groups(prose)
        wav, spans = vox.synthesize_segments(groups, output_path=master_wav)
        segs = nseg.segment_by_spans(spans)
        log.info(f"Audio-first VO via VoxCPM — {len(segs)} segments.")
        return wav, segs, "voxcpm"
    except Exception as e:
        log.warning(f"VoxCPM path unavailable ({e}); falling back to local Kokoro.")

    # --- Local fallback: Kokoro + silencedetect ---
    if progress_cb:
        progress_cb("voicing narration (local fallback)...")
    from modules.tts_engine import TTSEngine
    tts = TTSEngine(voice=rs.get_effective_voice())
    wav = tts.synthesize(prose, output_path=master_wav)
    segs = nseg.segment_by_silence(wav, sentence_texts=_split_sentences(prose))
    # Silence segmentation has no text mapping guarantee — if counts diverge,
    # fall back to even sentence-per-segment text assignment.
    sents = _split_sentences(prose)
    if len([s for s in segs if not s.text]) and len(sents) == len(segs):
        for s, txt in zip(segs, sents):
            s.text = txt
    log.info(f"Audio-first VO via local fallback — {len(segs)} segments.")
    return wav, segs, "kokoro"


# ==============================================================================
# STRUCTURER — assign beat/shot_type/speaker/visuals PER segment (no re-split)
# ==============================================================================

def _build_segment_structurer_prompt(n: int) -> str:
    """System prompt: the narration is ALREADY split + voiced. The model only
    annotates each fixed segment. It MUST NOT change the narration text."""
    styles = ", ".join(sg.get_available_style_ids())
    return f"""You output ONLY valid JSON. No prose, no preamble, no markdown fences. Start with {{ end with }}.

The story's narration has ALREADY been written, voiced, and split into {n} numbered SEGMENTS (in order). The audio is locked — you must NOT rewrite, merge, split, reorder, or rephrase any segment's words. Your ONLY job is to annotate each segment with camera + character metadata and extract the cast.

# RULES
1. Output EXACTLY {n} shots, one per segment, in the SAME order. Do NOT add or drop shots.
2. Do NOT include the narration text in your output — it is supplied separately. Just annotate.
3. For each shot pick:
   - `beat`: one of atmosphere | hook | spark | reaction | observation | struggle | moment-of-decision | choice | consequence
   - `shot_type`: wide | medium | closeup | insert (wide for new locations/atmosphere; insert for hands+object; closeup for reactions/decisions; medium otherwise). Avoid two identical consecutive types unless it's a dialogue back-and-forth.
   - `speaker`: "narrator" or the EXACT name of a character who speaks this segment's words. If the segment's words are spoken dialogue (a question/command/exclamation or first-person "I/we"), set the speaking character; otherwise "narrator".
   - `visual_description`: one short sentence — what we SEE this segment.
   - `first_frame_prompt` / `last_frame_prompt`: ~25-45 words, pose + setting + camera framing + lighting ONLY. Refer to characters by NAME — never re-describe their appearance (a locked token is injected later).
   - `motion_prompt`: ~25-45 words, physical motion + ONE camera move. No speech.
4. Extract every named character into `characters` with `name` (short proper name, no species), `type` (human|animal|creature), `appearance` (one sentence, MUST start with explicit age — number for humans, life stage for animals — then species/colors/clothing/feature; NO poses/actions), and `locked_visual_token` (tight 15-30 words starting with the age, pasteable verbatim into every shot).
5. Pick `style` from: {styles}. Pick `culture` from indian|western|japanese|mixed|animal-kingdom|fantasy. Pick `music_mood` from cheerful|calm|adventurous|tender|magical|energetic. Write a one-line `moral` (for parents, never spoken).

# OUTPUT FORMAT
{{
  "title": "<keep the given title>",
  "culture": "...",
  "style": "...",
  "characters": [{{"name":"...","type":"...","appearance":"...","locked_visual_token":"..."}}],
  "setting": "<one sentence>",
  "shots": [
    {{"segment": 1, "beat":"...", "shot_type":"...", "speaker":"...", "visual_description":"...", "first_frame_prompt":"...", "last_frame_prompt":"...", "motion_prompt":"..."}}
  ],
  "moral": "<one sentence>",
  "music_mood": "..."
}}

Output ONLY the JSON. Exactly {n} shots. Start with {{ end with }}."""


def _structure_segments(
    title: str,
    theme: str,
    prose: str,
    segments: list[nseg.Segment],
    style_override: Optional[str] = None,
    culture_override: Optional[str] = None,
) -> dict:
    """Ask the LLM to annotate the fixed segments; assemble the script dict.

    Narration is injected VERBATIM from the segments (never from the LLM) so the
    text always matches what was voiced. Falls back to a deterministic annotation
    if the LLM is down or returns the wrong shot count.
    """
    n = len(segments)
    seg_lines = "\n".join(f"{i+1}. {s.text}" for i, s in enumerate(segments))
    system_prompt = _build_segment_structurer_prompt(n)
    user_prompt = (
        f"Original theme: {theme}\nTitle: {title}\n\n"
        f"Full story (for context only — do not rewrite):\n{prose}\n\n"
        f"The {n} locked narration segments, in order:\n{seg_lines}\n\n"
        f"Annotate all {n} segments now. Output ONLY the JSON."
    )
    if style_override:
        user_prompt += f"\n\nUse style: {style_override}"
    if culture_override:
        user_prompt += f"\n\nUse culture: {culture_override}"

    meta = None
    try:
        raw = sg._call_llm(user_prompt, system_prompt, role="structurer")
        meta = sg._extract_json(raw)
    except Exception as e:
        log.warning(f"Segment structurer LLM failed ({e}); using deterministic fallback.")

    if not isinstance(meta, dict):
        meta = {}
    shots_meta = meta.get("shots") if isinstance(meta.get("shots"), list) else []

    # Build the final shots: narration VERBATIM from segments, metadata from LLM
    # (or fallback). Length is forced to n — extra LLM shots dropped, missing ones
    # filled deterministically.
    shots = []
    for i, seg in enumerate(segments):
        m = shots_meta[i] if i < len(shots_meta) and isinstance(shots_meta[i], dict) else {}
        shots.append(_assemble_shot(i, seg, m))

    script = {
        "title": title or meta.get("title") or "Untitled Story",
        "theme": theme,
        "culture": culture_override or meta.get("culture") or "mixed",
        "style": style_override or meta.get("style") or sg.get_default_style(),
        "duration_seconds": round(segments[-1].win_end) if segments else 30,
        "characters": meta.get("characters") or [],
        "setting": meta.get("setting") or "",
        "shots": shots,
        "moral": meta.get("moral") or "",
        "music_mood": meta.get("music_mood") or "cheerful",
        "_audio_first": True,
    }

    # Reuse the canonical validator: fills char tokens, normalizes speakers,
    # shot_type defaults, light direction, voice casting. We pass through it but
    # the narration is already final (validator's scrub only trims stray tags).
    script = sg._validate_and_default(script)
    return script


_BEAT_BY_POSITION = ["hook", "spark", "struggle", "observation",
                     "moment-of-decision", "choice", "consequence"]


def _assemble_shot(i: int, seg: nseg.Segment, m: dict) -> dict:
    """One shot dict: narration verbatim from the segment, metadata from the LLM
    annotation `m` (or position-based deterministic defaults)."""
    beat = (m.get("beat") or "").strip().lower()
    if beat not in ("atmosphere", "hook", "spark", "reaction", "observation",
                    "struggle", "moment-of-decision", "choice", "consequence"):
        beat = "atmosphere" if i == 0 else _BEAT_BY_POSITION[min(i, len(_BEAT_BY_POSITION) - 1)]
    shot_type = (m.get("shot_type") or "").strip().lower()
    if shot_type not in ("wide", "medium", "closeup", "insert"):
        shot_type = "wide" if i == 0 else "medium"
    return {
        "shot_number": i + 1,
        "beat": beat,
        "shot_type": shot_type,
        "speaker": (m.get("speaker") or "narrator").strip() or "narrator",
        "narration": seg.text,                      # VERBATIM — matches the voiced audio
        "visual_description": (m.get("visual_description") or seg.text)[:200],
        "first_frame_prompt": m.get("first_frame_prompt") or "",
        "last_frame_prompt": m.get("last_frame_prompt") or "",
        "motion_prompt": m.get("motion_prompt") or "",
        # audio-first timing carried for assembly
        "win_start": seg.win_start,
        "win_end": seg.win_end,
        "win_dur": round(seg.window_dur, 3),
        "audio_t_start": seg.t_start,
        "audio_t_end": seg.t_end,
    }


# ==============================================================================
# PUBLIC API
# ==============================================================================

def generate_script_audio_first(
    theme: str,
    style_override: Optional[str] = None,
    culture_override: Optional[str] = None,
    voice_id: Optional[str] = None,
    progress_cb: Optional[Callable] = None,
) -> dict:
    """Audio-first script generation. Writes a standard script_<id>.json (plus
    audio-first fields) and returns the script dict."""
    _job_start = _t.time()
    effective_style = style_override or rs.get_style_override()

    # 1) prose
    if progress_cb:
        progress_cb("writing story...")
    story = sw.write_story(theme=theme, style_hint=effective_style,
                           culture_hint=culture_override)
    log.info(f"Audio-first stage 1: '{story['title']}' ({story['word_count']} words)")

    # 2) voice + segment (pauses dictate shot count)
    wav, segments, engine = voice_and_segment(
        story["prose"], voice_id=voice_id, progress_cb=progress_cb
    )
    log.info(f"Voiced via {engine}; {len(segments)} segments (shots).")

    # 3) annotate segments → script schema (narration stays verbatim)
    if progress_cb:
        progress_cb(f"structuring {len(segments)} shots...")
    script = _structure_segments(
        title=story["title"], theme=theme, prose=story["prose"],
        segments=segments, style_override=effective_style,
        culture_override=culture_override,
    )
    script["_stage1_prose"] = story["prose"]
    script["_stage1_word_count"] = story["word_count"]
    script["_master_audio"] = str(wav)
    script["_tts_engine"] = engine

    # 4) safety
    is_safe, blocked_terms, warnings = sf.check_safety(script)
    if not is_safe:
        raise ValueError(
            f"Generated story contains hard-blocked terms: {blocked_terms}."
        )
    if warnings:
        log.info(f"Audio-first script soft warnings: {warnings}")

    # 5) metadata + save
    now = datetime.now()
    script_id = now.strftime("%Y%m%d_%H%M%S")
    script["_id"] = script_id
    script["script_id"] = script_id
    script["_generated_at"] = now.isoformat()
    script["_revision_number"] = 1
    script["revision_number"] = 1
    if effective_style:
        script["style"] = effective_style
    if culture_override:
        script["culture"] = culture_override

    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCRIPTS_DIR / f"script_{script_id}.json"
    atomic_write_json(out_path, script)
    log.info(f"Audio-first script saved: {out_path} ({len(script['shots'])} shots)")

    try:
        gm.record({
            "kind": "script",
            "script_id": script_id,
            "duration_sec": round(_t.time() - _job_start, 1),
            "vram_peak_mb": 0,
            "settings": {
                "backend_id": f"audio_first/{engine}",
                "style": script.get("style"),
                "shots": len(script.get("shots", [])),
                "title": script.get("title", "")[:60],
            },
            "success": True,
        })
    except Exception:
        pass
    return script


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=PROJECT_ROOT / "05_Config" / "secrets.env")
    theme = " ".join(sys.argv[1:]) or "a shy little mouse learning to speak up"
    s = generate_script_audio_first(theme, progress_cb=lambda m: print(f"  · {m}"))
    print(f"\nTitle: {s['title']}  shots={len(s['shots'])}  engine={s.get('_tts_engine')}")
    for sh in s["shots"]:
        print(f"  [{sh['shot_number']}] {sh['beat']}/{sh['shot_type']} "
              f"win={sh['win_start']}-{sh['win_end']} :: {sh['narration']}")
