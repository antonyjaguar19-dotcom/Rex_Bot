"""
Claw Bot — Runtime Settings

Persistent user overrides for style, resolution, steps, cfg.
Stored as JSON so they survive bot restarts.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import model_registry
from modules import voices
from modules.file_utils import atomic_write_json

log = logging.getLogger("claw_bot.runtime_settings")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SETTINGS_PATH = PROJECT_ROOT / "05_Config" / "runtime_settings.json"


# ==============================================================================
# LOAD / SAVE
# ==============================================================================

def _load() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"Could not read runtime_settings.json: {e}")
        return {}


def _save(data: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(SETTINGS_PATH, data)


# ==============================================================================
# INDIVIDUAL ACCESSORS
# ==============================================================================

def get_style_override() -> Optional[str]:
    """Return user-set default style, or None if not overridden."""
    return _load().get("style")


def set_style_override(style_id: str) -> None:
    data = _load()
    data["style"] = style_id
    _save(data)
    log.info(f"Style override set: {style_id}")


def clear_style_override() -> None:
    data = _load()
    data.pop("style", None)
    _save(data)
    log.info("Style override cleared (reverting to LLM choice)")


def get_resolution_override() -> Optional[str]:
    return _load().get("aspect_ratio")


def set_resolution_override(aspect: str) -> None:
    data = _load()
    data["aspect_ratio"] = aspect
    _save(data)
    log.info(f"Aspect ratio override set: {aspect}")


def get_steps_override() -> Optional[int]:
    val = _load().get("steps")
    return int(val) if val is not None else None


def set_steps_override(steps: int) -> None:
    data = _load()
    data["steps"] = int(steps)
    _save(data)
    log.info(f"Steps override set: {steps}")


def clear_steps_override() -> None:
    data = _load()
    data.pop("steps", None)
    _save(data)


def get_cfg_override() -> Optional[float]:
    val = _load().get("cfg")
    return float(val) if val is not None else None


def set_cfg_override(cfg: float) -> None:
    data = _load()
    data["cfg"] = float(cfg)
    _save(data)
    log.info(f"CFG override set: {cfg}")


def clear_cfg_override() -> None:
    data = _load()
    data.pop("cfg", None)
    _save(data)


def reset_overrides_for_model_switch() -> None:
    """Clear generation-tuning overrides (cfg + steps) so a freshly-switched model
    runs at ITS own recommended settings from models.json instead of inheriting
    the previous model's tuning. cfg is consumed by both image + video paths."""
    data = _load()
    cleared = [k for k in ("cfg", "steps") if k in data]
    for k in cleared:
        data.pop(k, None)
    _save(data)
    if cleared:
        log.info(f"Reset overrides for model switch: cleared {cleared}")


def get_all_overrides() -> dict:
    """Return all active overrides."""
    return _load()


def clear_all_overrides() -> None:
    _save({})
    log.info("All runtime overrides cleared")

# --- Voice (Kokoro TTS) ---

def get_voice_override() -> Optional[str]:
    """Self-healing read — an unknown stored id is treated as 'no override'
    rather than being handed to a UI select that will reject it."""
    raw = _load().get("voice")
    if not raw:
        return None
    canon = voices.normalize(raw)
    if canon is None:
        log.warning(f"Stored voice {raw!r} is not a valid voice; ignoring override")
        return None
    return canon


def set_voice_override(voice_id: str) -> None:
    """Validated write — an unknown voice never reaches disk."""
    canon = voices.normalize(voice_id)
    if canon is None:
        raise ValueError(f"'{voice_id}' is not a Kokoro voice. {voices.suggest(voice_id)}")
    data = _load()
    data["voice"] = canon
    _save(data)
    log.info(f"Voice override set: {canon}")


def clear_voice_override() -> None:
    data = _load()
    data.pop("voice", None)
    _save(data)


# --- Clip length (seconds) ---

def get_clip_length_override() -> Optional[float]:
    val = _load().get("clip_length")
    return float(val) if val is not None else None


def set_clip_length_override(seconds: float) -> None:
    data = _load()
    data["clip_length"] = float(seconds)
    _save(data)
    log.info(f"Clip length override set: {seconds}s")


def clear_clip_length_override() -> None:
    data = _load()
    data.pop("clip_length", None)
    _save(data)


# --- Sync mode (strict | loose) ---

def get_sync_mode_override() -> Optional[str]:
    return _load().get("sync_mode")


def set_sync_mode_override(mode: str) -> None:
    if mode not in ("strict", "loose"):
        raise ValueError(f"Invalid sync_mode: {mode!r}. Must be 'strict' or 'loose'.")
    data = _load()
    data["sync_mode"] = mode
    _save(data)
    log.info(f"Sync mode override set: {mode}")


def clear_sync_mode_override() -> None:
    data = _load()
    data.pop("sync_mode", None)
    _save(data)


# --- Transition mode (crossfade | cut) ---
# How adjacent shot clips join in final assembly:
#   crossfade — 0.3s dissolve; clips get a silent tail so narration of one shot
#               never mixes into the next during the overlap.
#   cut       — immediate hard cut at each clip's full length; zero narration loss.

def get_transition_mode_override() -> Optional[str]:
    return _load().get("transition_mode")


def set_transition_mode_override(mode: str) -> None:
    if mode not in ("crossfade", "cut"):
        raise ValueError(f"Invalid transition_mode: {mode!r}. Must be 'crossfade' or 'cut'.")
    data = _load()
    data["transition_mode"] = mode
    _save(data)
    log.info(f"Transition mode override set: {mode}")


def clear_transition_mode_override() -> None:
    data = _load()
    data.pop("transition_mode", None)
    _save(data)


def get_effective_transition_mode() -> str:
    return get_transition_mode_override() or "crossfade"


# --- Upscale toggle ---

def get_upscale_enabled() -> bool:
    """Whether upscale runs in auto-assembly. Default True."""
    val = _load().get("upscale_enabled")
    return True if val is None else bool(val)


def set_upscale_enabled(enabled: bool) -> None:
    data = _load()
    data["upscale_enabled"] = bool(enabled)
    _save(data)
    log.info(f"Upscale enabled: {enabled}")


# --- Music mood ---

def get_music_mood_override() -> Optional[str]:
    return _load().get("music_mood")


def set_music_mood_override(mood: str) -> None:
    from modules.music_generator import VALID_MOODS
    if mood not in VALID_MOODS:
        raise ValueError(f"Invalid music_mood: {mood!r}. Must be one of {VALID_MOODS}.")
    data = _load()
    data["music_mood"] = mood
    _save(data)
    log.info(f"Music mood override set: {mood}")


def clear_music_mood_override() -> None:
    data = _load()
    data.pop("music_mood", None)
    _save(data)


# --- Pipeline mode (story | music_video) ---
# Selects which production pipeline the "Generate" entrypoints drive.
#   story        — theme → kids story → storyboard → Wan video → TTS → assemble.
#   music_video  — theme → song lyrics (Qwen) → ACE-Step song → Ken Burns stills.
# Default story so existing behavior is unchanged until the user opts in.

VALID_PIPELINE_MODES = ("story", "music_video", "horror_story", "facts")


def get_pipeline_mode() -> str:
    """Active pipeline mode. Default 'story'."""
    val = _load().get("pipeline_mode")
    return val if val in VALID_PIPELINE_MODES else "story"


def set_pipeline_mode(mode: str) -> None:
    mode = (mode or "").strip().lower()
    if mode not in VALID_PIPELINE_MODES:
        raise ValueError(
            f"Invalid pipeline_mode: {mode!r}. Must be one of {VALID_PIPELINE_MODES}."
        )
    data = _load()
    data["pipeline_mode"] = mode
    _save(data)
    log.info(f"Pipeline mode set: {mode}")


# --- Music-video overrides (None = let the LLM choose) ---
# song_style   — musical genre fed to ACE-Step tags.
# vocal_type   — singer voice fed to ACE-Step tags.
# visual_style — look of the Ken Burns stills (maps to a styles.json entry).

VALID_SONG_STYLES = (
    "soft", "hard", "rap", "pop", "jazz", "metal", "romantic",
    "lofi", "edm", "acoustic", "cinematic", "folk",
)
VALID_VOCAL_TYPES = (
    # Vocal type = WHO sings. 'rap' removed (it's a Song Style, not a voice).
    # 'choir' dropped (ACE-Step rarely delivers it). instrumental works only when
    # lyrics are cleared (see musicvideo_pipeline._effective_lyrics).
    "auto", "female", "male", "duet", "instrumental",
)
VALID_VISUAL_STYLES = ("cartoon", "doodle", "spectrum", "photoreal")


def get_song_style_override() -> Optional[str]:
    return _load().get("song_style")


def set_song_style_override(style: str) -> None:
    style = (style or "").strip().lower()
    if style not in VALID_SONG_STYLES:
        raise ValueError(f"Invalid song_style: {style!r}. Must be one of {VALID_SONG_STYLES}.")
    data = _load()
    data["song_style"] = style
    _save(data)
    log.info(f"Song style override set: {style}")


def clear_song_style_override() -> None:
    data = _load()
    data.pop("song_style", None)
    _save(data)


# --- Tempo override (bpm fed to ACE-Step; None = let the LLM/song choose) ---
VALID_TEMPOS = {"slow": 75, "medium": 100, "fast": 130, "very fast": 160}


def get_tempo_override():
    v = _load().get("tempo_bpm")
    return int(v) if v is not None else None


def set_tempo_override(tempo: str) -> int:
    t = (tempo or "").strip().lower()
    if t not in VALID_TEMPOS:
        raise ValueError(f"Invalid tempo: {tempo!r}. Must be one of {list(VALID_TEMPOS)} or auto.")
    data = _load()
    data["tempo_bpm"] = VALID_TEMPOS[t]
    _save(data)
    log.info(f"Tempo override set: {t} ({VALID_TEMPOS[t]} bpm)")
    return VALID_TEMPOS[t]


def clear_tempo_override() -> None:
    data = _load()
    data.pop("tempo_bpm", None)
    _save(data)


def get_music_image_backend() -> str:
    """Music scene stills: 'zturbo' (Z-Image Turbo — fast + reliable, no cross-scene
    consistency; DEFAULT) or 'uso' (character consistency via scene-0 anchor, slower
    + intermittent hangs)."""
    v = _load().get("music_image_backend")
    return v if v in ("uso", "zturbo") else "zturbo"


def set_music_image_backend(mode: str) -> None:
    mode = (mode or "").strip().lower()
    if mode not in ("uso", "zturbo"):
        raise ValueError(f"Invalid music_image_backend: {mode!r} (uso|zturbo).")
    data = _load()
    data["music_image_backend"] = mode
    _save(data)
    log.info(f"Music image backend: {mode}")


def get_tempo_label() -> str:
    bpm = get_tempo_override()
    if bpm is None:
        return "auto"
    for k, v in VALID_TEMPOS.items():
        if v == bpm:
            return k
    return f"{bpm}bpm"


def get_vocal_type_override() -> Optional[str]:
    return _load().get("vocal_type")


def set_vocal_type_override(vocal: str) -> None:
    vocal = (vocal or "").strip().lower()
    if vocal not in VALID_VOCAL_TYPES:
        raise ValueError(f"Invalid vocal_type: {vocal!r}. Must be one of {VALID_VOCAL_TYPES}.")
    data = _load()
    data["vocal_type"] = vocal
    _save(data)
    log.info(f"Vocal type override set: {vocal}")


def clear_vocal_type_override() -> None:
    data = _load()
    data.pop("vocal_type", None)
    _save(data)


def get_visual_style_override() -> Optional[str]:
    return _load().get("visual_style")


def set_visual_style_override(style: str) -> None:
    style = (style or "").strip().lower()
    if style not in VALID_VISUAL_STYLES:
        raise ValueError(f"Invalid visual_style: {style!r}. Must be one of {VALID_VISUAL_STYLES}.")
    data = _load()
    data["visual_style"] = style
    _save(data)
    log.info(f"Visual style override set: {style}")


def clear_visual_style_override() -> None:
    data = _load()
    data.pop("visual_style", None)
    _save(data)


# --- Horror story narrator voice ---
# engine: 'kokoro' (preset, DETERMINISTIC — identical voice every call, no drift,
#         the consistency fix) or 'voxcpm' (designed/anchored, can still drift).
# voice: a Kokoro voice id (am_michael / am_adam cached locally).
# speed: <1.0 slows for dread.

_HORROR_ENGINES = ("kokoro", "voxcpm", "chatterbox", "qwen")


def get_horror_voice_engine() -> str:
    v = _load().get("horror_voice_engine")
    return v if v in _HORROR_ENGINES else "kokoro"


def set_horror_voice_engine(engine: str) -> None:
    engine = (engine or "").strip().lower()
    if engine not in _HORROR_ENGINES:
        raise ValueError(f"Invalid horror_voice_engine: {engine!r} ({'|'.join(_HORROR_ENGINES)}).")
    data = _load()
    data["horror_voice_engine"] = engine
    _save(data)
    log.info(f"Horror voice engine: {engine}")


def get_horror_voice() -> str:
    return _load().get("horror_voice") or "am_michael"


def set_horror_voice(voice: str) -> None:
    data = _load()
    data["horror_voice"] = voice.strip()
    _save(data)
    log.info(f"Horror voice: {voice}")


def get_horror_voice_speed() -> float:
    val = _load().get("horror_voice_speed")
    return float(val) if val is not None else 0.88


def set_horror_voice_speed(speed: float) -> None:
    data = _load()
    data["horror_voice_speed"] = float(speed)
    _save(data)
    log.info(f"Horror voice speed: {speed}")


# --- Facts Shorts voice: bright + slightly fast = energetic, "alive" delivery
# (kokoro voice ids installed: af_bella / af_nicole / af_sky / am_adam / am_michael)
def get_facts_voice() -> str:
    """Self-healing read: a bad stored id degrades to the default rather than
    crashing the dashboard's select ("Invalid value: Af_bella" -> HTTP 500)."""
    raw = _load().get("facts_voice")
    voice = voices.coerce(raw, voices.DEFAULT_FACTS_VOICE, voices.FACTS_VOICES)
    if raw and voice != raw:
        log.warning(f"Stored facts_voice {raw!r} is not a valid voice; using {voice!r}")
    return voice


def set_facts_voice(voice: str) -> None:
    """Validated write. Raises ValueError on an unknown voice so the bad value
    never reaches disk — that is what bricked the dashboard."""
    canon = voices.normalize(voice, voices.FACTS_VOICES)
    if canon is None:
        raise ValueError(
            f"'{voice}' is not a facts voice. "
            f"{voices.suggest(voice, voices.FACTS_VOICES)}")
    data = _load()
    data["facts_voice"] = canon
    _save(data)
    log.info(f"Facts voice: {canon}")


def get_facts_voice_speed() -> float:
    val = _load().get("facts_voice_speed")
    return float(val) if val is not None else 1.08


def set_facts_voice_speed(speed: float) -> None:
    data = _load()
    data["facts_voice_speed"] = float(speed)
    _save(data)
    log.info(f"Facts voice speed: {speed}")


def get_facts_video_mode() -> str:
    """'wan' = animate each cut with Wan I2V (cinematic, ~2-4 min per clip).
    'kenburns' = pan/zoom stills (fast, stable).

    Default 'wan': the animated cut is what makes a facts reel look made rather
    than assembled, and gpu_memory now keeps Wan from sharing the card with
    Ollama or the thumbnail model. The pipeline still falls back to Ken Burns on
    an OOM, so the reel always lands.
    """
    v = _load().get("facts_video_mode")
    return v if v in ("kenburns", "wan") else "wan"


def set_facts_video_mode(mode: str) -> None:
    mode = (mode or "").strip().lower()
    if mode not in ("kenburns", "wan"):
        raise ValueError(f"Invalid facts_video_mode: {mode!r} (kenburns|wan).")
    data = _load()
    data["facts_video_mode"] = mode
    _save(data)
    log.info(f"Facts video mode: {mode}")


# --- Mascot voice (facts mascot-presenter mode) ---------------------------
# Kokoro is fast but flat, and pitch-shifting it to fake a kid sounds artificial.
# Qwen3-TTS with an emotion "instruct" gives a natural, expressive, deterministic
# (= consistent) read. Vivian was the chosen timbre for the jaguar cub.
# Chatterbox is the third option: it CLONES a reference clip. No local TTS ships
# a genuine child voice — Eric+2st is an adult timbre pitch-shifted, and it never
# read as cute. A 5-15s reference recording is the only route to a real one.
VALID_MASCOT_TTS = ("qwen", "kokoro", "chatterbox")
VALID_QWEN_SPEAKERS = ("Vivian", "Serena", "Dylan", "Eric", "Uncle_Fu")
DEFAULT_MASCOT_SPEAKER = "Eric"
# Happy, not hyper. The first cut ("excited, high-energy, bouncy") read as
# shouty — a friendly explainer should sound warm and relaxed, not caffeinated.
DEFAULT_MASCOT_INSTRUCT = ("happy young boy about ten years old, bright youthful "
                           "child voice, warm and friendly, relaxed and natural, "
                           "gently upbeat, EVEN steady delivery throughout, "
                           "calm and conversational, never excited, never shouting, "
                           "no dramatic emphasis")


def get_mascot_tts_engine() -> str:
    v = _load().get("mascot_tts_engine")
    return v if v in VALID_MASCOT_TTS else "qwen"


def set_mascot_tts_engine(engine: str) -> None:
    engine = (engine or "").strip().lower()
    if engine not in VALID_MASCOT_TTS:
        raise ValueError(f"engine must be one of {VALID_MASCOT_TTS}")
    data = _load(); data["mascot_tts_engine"] = engine; _save(data)
    log.info(f"Mascot TTS engine: {engine}")


def get_mascot_voice() -> str:
    v = _load().get("mascot_voice")
    return v if v in VALID_QWEN_SPEAKERS else DEFAULT_MASCOT_SPEAKER


def set_mascot_voice(speaker: str) -> None:
    speaker = (speaker or "").strip()
    match = next((s for s in VALID_QWEN_SPEAKERS if s.lower() == speaker.lower()), None)
    if not match:
        raise ValueError(f"voice must be one of {VALID_QWEN_SPEAKERS}")
    data = _load(); data["mascot_voice"] = match; _save(data)
    log.info(f"Mascot voice: {match}")


def get_mascot_voice_speed() -> float:
    """Playback speed for the mascot's narration (pitch preserved).

    Qwen's instruct steers tone reliably but not pace, so the pace is a post
    step. 1.0 read a touch slow; 1.20 was the chosen pace.
    """
    try:
        v = float(_load().get("mascot_voice_speed", 1.20))
    except (TypeError, ValueError):
        return 1.20
    return v if 0.5 <= v <= 2.0 else 1.20


def set_mascot_voice_speed(speed: float) -> None:
    speed = float(speed)
    if not 0.5 <= speed <= 2.0:
        raise ValueError("speed must be between 0.5 and 2.0")
    data = _load(); data["mascot_voice_speed"] = speed; _save(data)
    log.info(f"Mascot voice speed: {speed}")


def get_mascot_voice_pitch() -> float:
    """Semitones to raise the mascot's voice (duration preserved).

    No local TTS has a genuine child voice. Eric (adult male) lifted +2 semitones
    reads as the young boy the jaguar cub should be; more than that starts to
    sound like a chipmunk.
    """
    try:
        v = float(_load().get("mascot_voice_pitch", 2.0))
    except (TypeError, ValueError):
        return 2.0
    return v if -6.0 <= v <= 6.0 else 2.0


def set_mascot_voice_pitch(semitones: float) -> None:
    semitones = float(semitones)
    if not -6.0 <= semitones <= 6.0:
        raise ValueError("pitch must be between -6 and +6 semitones")
    data = _load(); data["mascot_voice_pitch"] = semitones; _save(data)
    log.info(f"Mascot voice pitch: {semitones:+.1f} st")


def get_mascot_voice_instruct() -> str:
    return _load().get("mascot_voice_instruct") or DEFAULT_MASCOT_INSTRUCT


def set_mascot_voice_instruct(text: str) -> None:
    data = _load()
    data["mascot_voice_instruct"] = (text or "").strip() or DEFAULT_MASCOT_INSTRUCT
    _save(data)
    log.info(f"Mascot voice instruct: {data['mascot_voice_instruct'][:60]}")


# The clip Chatterbox clones for the mascot. Default lives beside the mascot art.
DEFAULT_MASCOT_VOICE_REF = (
    Path(__file__).parent.parent / "assets" / "mascot_voice.wav"
)


def get_mascot_voice_ref() -> Optional[Path]:
    """Reference clip Chatterbox clones the mascot's voice from, or None.

    Timbre comes from this file, so the pitch/speed knobs above are NOT applied
    on top of a clone — shifting a cloned voice just undoes the cloning.
    """
    raw = _load().get("mascot_voice_ref")
    p = Path(raw) if raw else DEFAULT_MASCOT_VOICE_REF
    return p if p.exists() else None


def set_mascot_voice_ref(path) -> None:
    if path in (None, "", "none"):
        data = _load(); data.pop("mascot_voice_ref", None); _save(data)
        log.info("Mascot voice ref cleared.")
        return
    p = Path(path).resolve()
    if not p.exists():
        raise ValueError(f"reference clip not found: {p}")
    data = _load(); data["mascot_voice_ref"] = str(p); _save(data)
    log.info(f"Mascot voice ref: {p.name}")


# --- Which mascot ----------------------------------------------------------
# A folder name under 02_Agent/assets/mascots/ (see modules/mascot_library.py).
# Kept as a plain string here, deliberately: runtime_settings is the bottom of the
# import graph and must not reach up into the library to validate. The library
# resolves a stale id to the first mascot on the shelf.
def get_active_mascot() -> str:
    return (_load().get("active_mascot") or "").strip()


def set_active_mascot(mascot_id: str) -> None:
    data = _load()
    mid = (mascot_id or "").strip()
    if mid:
        data["active_mascot"] = mid
    else:
        data.pop("active_mascot", None)
    _save(data)
    log.info(f"Active mascot: {mid or '(none)'}")


# --- Facts background music ------------------------------------------------
# facts_assembly has always been able to MIX a bed (at 0.16, well under the voice)
# — nothing ever generated one, so every facts reel shipped with silence under the
# narration. MusicGen writes it; "cheerful" suits the mascot.
def get_facts_music_enabled() -> bool:
    v = _load().get("facts_music_enabled")
    return True if v is None else bool(v)


def set_facts_music_enabled(enabled: bool) -> None:
    data = _load(); data["facts_music_enabled"] = bool(enabled); _save(data)
    log.info(f"Facts background music: {'on' if enabled else 'off'}")


def get_facts_music_mood() -> str:
    from modules.music_generator import VALID_MOODS
    v = _load().get("facts_music_mood")
    return v if v in VALID_MOODS else "cheerful"


def set_facts_music_mood(mood: str) -> None:
    from modules.music_generator import VALID_MOODS
    mood = (mood or "").strip().lower()
    if mood not in VALID_MOODS:
        raise ValueError(f"mood must be one of {VALID_MOODS}")
    data = _load(); data["facts_music_mood"] = mood; _save(data)
    log.info(f"Facts music mood: {mood}")


# Shorts custom thumbnails are not offered in every region (Jeffy's included), and
# where they are not, the platform grabs the FIRST FRAME. So the thumbnail is held
# on the front of the reel: a poster frame, not an animated shot. 0 = off.
def get_facts_thumb_hold_sec() -> float:
    try:
        v = float(_load().get("facts_thumb_hold_sec", 0.5))
    except (TypeError, ValueError):
        return 0.5
    return v if 0.0 <= v <= 3.0 else 0.5


def set_facts_thumb_hold_sec(seconds: float) -> None:
    seconds = float(seconds)
    if not 0.0 <= seconds <= 3.0:
        raise ValueError("hold must be between 0 and 3 seconds (0 = off)")
    data = _load(); data["facts_thumb_hold_sec"] = seconds; _save(data)
    log.info(f"Facts thumbnail hold: {seconds:.2f}s")


def get_facts_max_seconds() -> float:
    """Hard ceiling on a facts reel. Shorts die on length — 40s is the target.

    Enforced by measurement, not by hope: the beats are voiced, the real total is
    measured, and the pace is trimmed to fit (see facts_pipeline._fit_to_budget).
    """
    try:
        v = float(_load().get("facts_max_seconds", 40.0))
    except (TypeError, ValueError):
        return 40.0
    return v if 10.0 <= v <= 180.0 else 40.0


def set_facts_max_seconds(seconds: float) -> None:
    seconds = float(seconds)
    if not 10.0 <= seconds <= 180.0:
        raise ValueError("length must be between 10 and 180 seconds")
    data = _load(); data["facts_max_seconds"] = seconds; _save(data)
    log.info(f"Facts reel ceiling: {seconds:.0f}s")


def get_mascot_voice_exaggeration() -> float:
    """Chatterbox emotion knob. 0.5 neutral; higher = more theatrical.

    0.35 was chosen by ear. Every earlier voice failed the same way — too excited,
    tone spiking between beats — so the presenter reads calm and lets the fact land.
    """
    try:
        v = float(_load().get("mascot_voice_exaggeration", 0.35))
    except (TypeError, ValueError):
        return 0.35
    return v if 0.0 <= v <= 1.5 else 0.35


def set_mascot_voice_exaggeration(v: float) -> None:
    v = float(v)
    if not 0.0 <= v <= 1.5:
        raise ValueError("exaggeration must be between 0.0 and 1.5")
    data = _load(); data["mascot_voice_exaggeration"] = v; _save(data)
    log.info(f"Mascot voice exaggeration: {v}")



def get_facts_mascot_lipsync() -> bool:
    """Whether the mascot's mouth is animated to the narration (Wan S2V).

    Default OFF. S2V is a TALKING-HEAD model: it lip-syncs well but it does not
    understand hands, props or legs, and it dissolved them — a paw melted mid-clip
    and a leg bent backwards. It is also ~4 min a clip. With it off the mascot is
    animated by the normal I2V backend (clean motion, much faster) and the
    narration plays as a voice-over, which is what a presenter reel wants anyway.
    """
    return bool(_load().get("facts_mascot_lipsync", False))


def set_facts_mascot_lipsync(enabled: bool) -> None:
    data = _load()
    data["facts_mascot_lipsync"] = bool(enabled)
    _save(data)
    log.info(f"Facts mascot lip-sync (S2V): {'ON' if enabled else 'OFF'}")


def get_facts_mascot_mode() -> bool:
    """Whether facts reels star the mascot in every shot (costumed, explaining
    each fact, lip-synced via S2V) instead of abstract backdrops.

    Off by default: it is the slow, heavy path (per-shot Qwen still + S2V clip).
    When on, the pipeline ignores the abstract-backdrop backend and the I2V/Ken
    Burns animate stage. [[baked-thumbnail-headline]] shares the mascot ref.
    """
    return bool(_load().get("facts_mascot_mode", False))


def set_facts_mascot_mode(enabled: bool) -> None:
    data = _load()
    data["facts_mascot_mode"] = bool(enabled)
    _save(data)
    log.info(f"Facts mascot mode: {'ON' if enabled else 'OFF'}")


def get_facts_thumbnail_enabled() -> bool:
    """Whether a facts reel gets a generated thumbnail at all.

    The thumbnail is a whole extra stage — a title from the LLM plus one or two
    ~25 s Qwen renders. On by default (a Short wants a cover), but skippable when
    you only care about the video. The upload title and description are still
    written; only the thumbnail image is skipped.
    """
    val = _load().get("facts_thumbnail")
    return True if val is None else bool(val)


def set_facts_thumbnail_enabled(enabled: bool) -> None:
    data = _load()
    data["facts_thumbnail"] = bool(enabled)
    _save(data)
    log.info(f"Facts thumbnail: {'ON' if enabled else 'OFF'}")


_HORROR_QWEN_INSTRUCT = (
    "Read this aloud as a serious adult male narrator telling a horror story: "
    "slow, deep, calm and grave, with an unhurried, measured, steady delivery "
    "like a documentary voice recounting something truly disturbing. Low pitch, "
    "no theatrics, no cheerfulness, no exaggeration."
)


def get_horror_qwen_instruct() -> str:
    """Tone/emotion instruction for Qwen3-TTS horror narration."""
    return _load().get("horror_qwen_instruct") or _HORROR_QWEN_INSTRUCT


def set_horror_qwen_instruct(text: str) -> None:
    data = _load()
    data["horror_qwen_instruct"] = text
    _save(data)
    log.info(f"Horror qwen instruct set: {text[:60]}")


def get_horror_qwen_pitch() -> float:
    """Post-process pitch/tempo factor for the horror narrator. <1.0 deepens via
    ffmpeg asetrate (tempo restored, no slow-mo). 1.0 = OFF (raw TTS, never sped
    up or slowed). Default 1.0 — keep the narrator's natural voice untouched."""
    val = _load().get("horror_qwen_pitch")
    return float(val) if val is not None else 1.0


def set_horror_qwen_pitch(factor: float) -> None:
    data = _load()
    data["horror_qwen_pitch"] = float(factor)
    _save(data)
    log.info(f"Horror qwen pitch factor: {factor}")


def get_horror_qwen_speaker() -> str:
    """Qwen3-TTS preset speaker for the horror narrator (eric/ryan/dylan/...)."""
    return _load().get("horror_qwen_speaker") or "eric"


def set_horror_qwen_speaker(spk: str) -> None:
    data = _load()
    data["horror_qwen_speaker"] = spk.strip()
    _save(data)
    log.info(f"Horror qwen speaker: {spk}")


def get_horror_chatterbox_exaggeration() -> float:
    """Chatterbox emotion intensity for the horror narrator (0.5 calm .. 0.8 intense)."""
    val = _load().get("horror_cb_exaggeration")
    return float(val) if val is not None else 0.6


def set_horror_chatterbox_exaggeration(v: float) -> None:
    data = _load()
    data["horror_cb_exaggeration"] = float(v)
    _save(data)
    log.info(f"Horror chatterbox exaggeration: {v}")


# --- Horror visuals: animated Wan clips vs Ken Burns stills ---

def get_horror_video_mode() -> str:
    """'wan' = animated Wan I2V clip per shot (cinematic, slow). 'kenburns' =
    photoreal stills with pan/zoom (fast). Default 'wan'."""
    v = _load().get("horror_video_mode")
    return v if v in ("wan", "kenburns") else "wan"


def set_horror_video_mode(mode: str) -> None:
    mode = (mode or "").strip().lower()
    if mode not in ("wan", "kenburns"):
        raise ValueError(f"Invalid horror_video_mode: {mode!r} (wan|kenburns).")
    data = _load()
    data["horror_video_mode"] = mode
    _save(data)
    log.info(f"Horror video mode: {mode}")


# --- Horror visuals: fixed STYLE LoRA (no per-character LoRA) ---

def get_horror_style_lora() -> str:
    """ComfyUI loras/ filename of the horror STYLE LoRA stacked on every horror
    still (flux_lora backend). Identity consistency comes from prompt tokens, not
    a character LoRA. '' = no style LoRA. Default 'Horrorstyle.safetensors'."""
    val = _load().get("horror_style_lora")
    return "Horrorstyle.safetensors" if val is None else str(val)


def set_horror_style_lora(name: str) -> None:
    data = _load()
    data["horror_style_lora"] = (name or "").strip()
    _save(data)
    log.info(f"Horror style LoRA: {name}")


def get_horror_style_lora_weight() -> float:
    val = _load().get("horror_style_lora_weight")
    return float(val) if val is not None else 0.8


def set_horror_style_lora_weight(w: float) -> None:
    data = _load()
    data["horror_style_lora_weight"] = float(w)
    _save(data)
    log.info(f"Horror style LoRA weight: {w}")


# --- Horror story mode: ambient drone bed under narration ---

def get_horror_ambient_enabled() -> bool:
    """Whether a low ambient drone is mixed under the horror narration. Default True."""
    val = _load().get("horror_ambient")
    return True if val is None else bool(val)


def set_horror_ambient_enabled(enabled: bool) -> None:
    data = _load()
    data["horror_ambient"] = bool(enabled)
    _save(data)
    log.info(f"Horror ambient bed: {enabled}")


# --- Character reference mode (cast-sheet anchoring) ---

def get_reference_mode_enabled() -> bool:
    """Whether storyboards use a cast-sheet reference latent for character consistency.
    Default False — opt in with !set_reference_mode on."""
    val = _load().get("reference_mode")
    return False if val is None else bool(val)


def set_reference_mode_enabled(enabled: bool) -> None:
    data = _load()
    data["reference_mode"] = bool(enabled)
    _save(data)
    log.info(f"Reference mode: {enabled}")


# --- USO character-consistency backend (kids + music) ---

def get_mascot_thumbnails_enabled() -> bool:
    """Whether finished videos get a branded MASCOT thumbnail (USO renders the
    mascot in a scene about the video) instead of a frame from the video itself.
    Default True — it no-ops harmlessly until a mascot image exists at
    02_Agent/assets/mascot.png."""
    val = _load().get("mascot_thumbnails")
    return True if val is None else bool(val)


def set_mascot_thumbnails_enabled(enabled: bool) -> None:
    data = _load()
    data["mascot_thumbnails"] = bool(enabled)
    _save(data)
    log.info(f"Mascot thumbnails: {'ON' if enabled else 'OFF'}")


def get_uso_mode_enabled() -> bool:
    """Whether kids storyboards + music videos render via the USO backend
    (Flux.1-dev + USO LoRA, single-image subject consistency). Default True.
    Turn off with set_uso_mode_enabled(False) to fall back to the previous
    backends (SDXL+IP-Adapter for kids, active backend for music)."""
    val = _load().get("uso_mode")
    return True if val is None else bool(val)


def set_uso_mode_enabled(enabled: bool) -> None:
    data = _load()
    data["uso_mode"] = bool(enabled)
    _save(data)
    log.info(f"USO mode: {enabled}")


# --- Music-video lyric captions (burned-in subtitles) ---

def get_music_captions_enabled() -> bool:
    """Whether music videos burn word-synced lyric captions. Default False —
    music ships clean (watermark only). Turn on with
    set_music_captions_enabled(True). When off, the WhisperX alignment pass is
    skipped entirely (faster render)."""
    val = _load().get("music_captions")
    return False if val is None else bool(val)


def set_music_captions_enabled(enabled: bool) -> None:
    data = _load()
    data["music_captions"] = bool(enabled)
    _save(data)
    log.info(f"Music lyric captions: {enabled}")


# --- Lip-sync (Wan S2V for character dialogue shots) ---

def get_lipsync_enabled() -> bool:
    """Whether CHARACTER dialogue shots render via the Wan-S2V lip-sync backend
    instead of the silent I2V backend. Default False — opt in once S2V weights +
    workflow are installed. Narrator shots always use I2V regardless."""
    val = _load().get("lipsync_enabled")
    return False if val is None else bool(val)


def set_lipsync_enabled(enabled: bool) -> None:
    data = _load()
    data["lipsync_enabled"] = bool(enabled)
    _save(data)
    log.info(f"Lip-sync (S2V): {enabled}")


def get_lipsync_backend_id() -> str:
    """Registry id of the S2V backend used for character shots."""
    return _load().get("lipsync_backend_id") or "comfyui_wan22_s2v"


# ==============================================================================
# RESOLVED VALUES — what the pipeline actually uses
# ==============================================================================

def get_effective_style() -> str:
    """User override > registry default."""
    override = get_style_override()
    if override:
        return override
    return "pixar"  # fallback default


def get_effective_aspect_ratio() -> str:
    override = get_resolution_override()
    if override:
        return override
    defaults = model_registry.get_image_defaults()
    return defaults.get("aspect_ratio", "16:9")


def get_effective_steps() -> int:
    """User override > backend config > 8."""
    override = get_steps_override()
    if override is not None:
        return override
    cfg = model_registry.get_active("image_backend")
    return int(cfg.get("steps", 8))


def get_effective_cfg() -> float:
    override = get_cfg_override()
    if override is not None:
        return override
    cfg = model_registry.get_active("image_backend")
    return float(cfg.get("cfg", 1.0))


# ==============================================================================
# VIDEO RESOLUTION  (named presets → width×height per aspect)
# ==============================================================================
# Dims are multiples of 16 (Wan/LTX latent-stride friendly). Higher res = sharper,
# less noise, but slower + more VRAM. 1080p I2V is heavy on 16GB — use sparingly.
VIDEO_RES_PRESETS = {
    "480p":  {"16:9": (832, 480),  "9:16": (480, 832),  "1:1": (512, 512)},
    "720p":  {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (768, 768)},
    "1080p": {"16:9": (1920, 1072), "9:16": (1072, 1920), "1:1": (1024, 1024)},
}


def get_video_resolution_override() -> Optional[str]:
    return _load().get("video_resolution")


def set_video_resolution_override(preset: str) -> None:
    preset = preset.strip().lower()
    if preset not in VIDEO_RES_PRESETS:
        raise ValueError(
            f"Unknown video resolution '{preset}'. Choose: {', '.join(VIDEO_RES_PRESETS)}"
        )
    data = _load()
    data["video_resolution"] = preset
    _save(data)
    log.info(f"Video resolution override -> {preset}")


def clear_video_resolution_override() -> None:
    data = _load()
    data.pop("video_resolution", None)
    _save(data)


def get_effective_video_resolution() -> Optional[tuple]:
    """Return (width, height) for video gen, or None to let the backend use its
    own models.json default dims. Resolves the named preset against the effective
    aspect ratio."""
    preset = get_video_resolution_override()
    if not preset:
        return None
    aspect = get_effective_aspect_ratio()
    table = VIDEO_RES_PRESETS.get(preset, {})
    return table.get(aspect) or table.get("16:9")

def get_effective_voice() -> str:
    """User override > Kokoro default ('af_heart'). Always a valid voice id."""
    return get_voice_override() or voices.DEFAULT_VOICE


def get_effective_sync_mode() -> str:
    override = get_sync_mode_override()
    if override:
        return override
    return "strict"