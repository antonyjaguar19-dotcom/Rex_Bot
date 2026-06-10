"""
Claw Bot — Background Music Generator (Phase 6B)

Two-tier source strategy:
  1. PRIMARY: MusicGen-medium via HuggingFace transformers (~6GB VRAM)
  2. FALLBACK: Local library at 03_Assets/music/<mood>/*.mp3

Mood decided by:
  1. !set_music_mood override (if active)
  2. LLM-picked mood from the script (default)
  3. DEFAULT_MOOD ('cheerful')
"""

import logging
import random
import subprocess
import sys
import time as _t
from pathlib import Path
from typing import Optional, Callable

# Import torch at module load so OOM exception class is always defined.
# If torch itself is missing, _TORCH_OK guards the model load.
try:
    import torch
    _TORCH_OK = True
except ImportError:
    torch = None
    _TORCH_OK = False

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import gpu_utils  # noqa
from modules import runtime_settings as rs  # noqa

log = logging.getLogger("claw_bot.music_generator")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FFMPEG_EXE = PROJECT_ROOT / "00_Tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE_EXE = PROJECT_ROOT / "00_Tools" / "ffmpeg" / "bin" / "ffprobe.exe"
MUSIC_LIB_DIR = PROJECT_ROOT / "03_Assets" / "music"
MUSIC_CACHE_DIR = PROJECT_ROOT / "04_Outputs" / "music_cache"
HF_CACHE_DIR = PROJECT_ROOT / "03_Models" / "hf_cache"

# Force HuggingFace to download inside the project (no leaks to C:\Users\...\.cache)
import os as _os
_os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
_os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
_os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "hub"))
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

VALID_MOODS = (
    "cheerful",
    "calm",
    "adventurous",
    "tender",
    "magical",
    "energetic",
)

DEFAULT_MOOD = "cheerful"

# Suffix appended to every prompt to enforce kid-safe instrumental output
_PROMPT_SUFFIX = ", instrumental, no vocals"

MOOD_PROMPTS = {
    "cheerful":    "happy children's music, upbeat, light and playful, major key, warm" + _PROMPT_SUFFIX,
    "calm":        "peaceful children's lullaby, soft and gentle, slow tempo, dreamy and soothing" + _PROMPT_SUFFIX,
    "adventurous": "adventure music for children, playful orchestral, curious and exploratory, light and bouncy" + _PROMPT_SUFFIX,
    "tender":      "heartwarming children's music, gentle and emotional, warm and sweet" + _PROMPT_SUFFIX,
    "magical":     "magical fairy-tale music for children, sparkling and dreamy, whimsical and enchanted" + _PROMPT_SUFFIX,
    "energetic":   "energetic happy children's music, fast and fun, bouncy and bright" + _PROMPT_SUFFIX,
}


# ==============================================================================
# MOOD RESOLUTION
# ==============================================================================

def get_effective_mood(script_data: Optional[dict] = None) -> str:
    """Override > LLM-picked from script > default."""
    override = rs.get_all_overrides().get("music_mood")
    if override and override in VALID_MOODS:
        log.info(f"Music mood from override: {override}")
        return override

    if script_data and "music_mood" in script_data:
        picked = script_data["music_mood"]
        if picked in VALID_MOODS:
            log.info(f"Music mood from script: {picked}")
            return picked

    log.info(f"Music mood fallback: {DEFAULT_MOOD}")
    return DEFAULT_MOOD


# ==============================================================================
# MUSICGEN (HuggingFace transformers)
# ==============================================================================

_MUSICGEN_MODEL = None         # tuple: (model, processor)
_MUSICGEN_LOAD_FAILED = False


def _load_musicgen():
    """Load MusicGen-medium via HF transformers. Returns (model, processor) or None."""
    global _MUSICGEN_MODEL, _MUSICGEN_LOAD_FAILED
    if _MUSICGEN_MODEL is not None:
        return _MUSICGEN_MODEL
    if _MUSICGEN_LOAD_FAILED:
        return None

    try:
        log.info("Freeing VRAM before MusicGen load...")
        try:
            gpu_utils.cleanup_after_job(free_comfy=True, free_ollama=True)
        except Exception as e:
            log.warning(f"VRAM cleanup failed (continuing anyway): {e}")

        log.info("Loading MusicGen-medium via HF transformers (~6GB VRAM, first run downloads ~3GB)...")
        if not _TORCH_OK:
            log.error("torch not installed in this venv — MusicGen unavailable.")
            _MUSICGEN_LOAD_FAILED = True
            return None
        from transformers import MusicgenForConditionalGeneration, AutoProcessor

        processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
        model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-medium",
            torch_dtype=torch.float16,
        ).to("cuda")
        model.eval()

        _MUSICGEN_MODEL = (model, processor)
        log.info("MusicGen-medium loaded.")
        return _MUSICGEN_MODEL
    except Exception as e:
        log.exception(f"MusicGen load failed: {e}")
        _MUSICGEN_LOAD_FAILED = True
        return None


def _generate_with_musicgen(
    mood: str,
    duration_sec: float,
    output_path: Path,
    progress_cb: Optional[Callable] = None,
) -> Optional[Path]:
    """Try MusicGen via HF transformers. Returns output path on success, None on failure."""
    loaded = _load_musicgen()
    if loaded is None:
        return None
    model, processor = loaded

    try:
        prompt = MOOD_PROMPTS.get(mood, MOOD_PROMPTS[DEFAULT_MOOD])
        duration = min(duration_sec + 2.0, 30.0)

        max_new_tokens = int(duration * 50)  # MusicGen ~50 tokens/sec
        log.info(f"MusicGen generating: mood={mood}, duration={duration:.1f}s, tokens={max_new_tokens}")
        if progress_cb:
            progress_cb(f"MusicGen rendering {duration:.0f}s ({mood})...")

        inputs = processor(text=[prompt], padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                guidance_scale=3.0,   # MusicGen sweet spot
            )
            
        wav = audio_values[0].cpu().float()
        sample_rate = model.config.audio_encoder.sampling_rate

        wav_path = output_path.with_suffix(".wav")
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        import soundfile as sf
        wav_np = wav.numpy().T  # (channels, samples) -> (samples, channels)
        sf.write(str(wav_path), wav_np, samplerate=sample_rate)
        log.info(f"MusicGen WAV saved: {wav_path}")

        if output_path.suffix.lower() == ".mp3":
            cmd = [
                str(FFMPEG_EXE), "-y", "-loglevel", "error",
                "-i", str(wav_path),
                "-c:a", "libmp3lame", "-b:a", "192k",
                str(output_path),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                log.warning(f"MP3 transcode failed, keeping WAV: {r.stderr}")
                return wav_path
            wav_path.unlink(missing_ok=True)
            return output_path

        return wav_path
    except torch.cuda.OutOfMemoryError as e:
        log.error(f"MusicGen OOM: {e}")
        global _MUSICGEN_MODEL
        _MUSICGEN_MODEL = None
        try:
            import gc
            gc.collect()
            import torch as _torch
            _torch.cuda.empty_cache()
        except Exception:
            pass
        return None
    except Exception as e:
        log.exception(f"MusicGen generation failed: {e}")
        return None


# ==============================================================================
# LIBRARY FALLBACK
# ==============================================================================

def _pick_library_track(mood: str) -> Optional[Path]:
    """Random track from 03_Assets/music/<mood>/. Cascades to DEFAULT_MOOD then any."""
    extensions = ("*.mp3", "*.wav", "*.m4a", "*.flac")
    search_order = [mood]
    if mood != DEFAULT_MOOD:
        search_order.append(DEFAULT_MOOD)

    for m in search_order:
        folder = MUSIC_LIB_DIR / m
        if folder.is_dir():
            candidates = []
            for ext in extensions:
                candidates.extend(folder.glob(ext))
            if candidates:
                pick = random.choice(candidates)
                log.info(f"Library pick ({m}): {pick.name}")
                return pick

    if MUSIC_LIB_DIR.is_dir():
        all_tracks = []
        for ext in extensions:
            all_tracks.extend(MUSIC_LIB_DIR.rglob(ext))
        if all_tracks:
            pick = random.choice(all_tracks)
            log.warning(f"No tracks for '{mood}'/'{DEFAULT_MOOD}'. Random: {pick}")
            return pick

    log.warning(f"No library tracks found in {MUSIC_LIB_DIR}")
    return None


def _trim_or_loop_to_duration(
    source: Path,
    duration_sec: float,
    output_path: Path,
) -> Path:
    """Trim or loop source audio to match target duration, with 1s fade in/out."""
    fade_out_start = max(duration_sec - 1.0, 0.0)
    cmd = [
        str(FFMPEG_EXE), "-y", "-loglevel", "error",
        "-stream_loop", "-1",
        "-i", str(source),
        "-t", f"{duration_sec:.3f}",
        "-af", f"afade=t=in:st=0:d=1,afade=t=out:st={fade_out_start:.3f}:d=1",
        "-c:a", "libmp3lame", "-b:a", "192k",
        str(output_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg trim/loop failed:\n{r.stderr.strip()}")
    return output_path


# ==============================================================================
# PUBLIC API
# ==============================================================================

def generate_music(
    mood: str,
    duration_sec: float,
    script_id: Optional[str] = None,
    progress_cb: Optional[Callable] = None,
) -> Optional[Path]:
    """Produce a music track of the requested duration matching mood.

    Strategy: MusicGen → library → None.
    """
    if mood not in VALID_MOODS:
        log.warning(f"Invalid mood '{mood}', using {DEFAULT_MOOD}")
        mood = DEFAULT_MOOD

    MUSIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = script_id or f"adhoc_{int(_t.time())}"
    final_path = MUSIC_CACHE_DIR / f"music_{tag}_{mood}.mp3"

    # Tier 1: MusicGen
    if progress_cb:
        progress_cb("trying MusicGen...")
    musicgen_raw = MUSIC_CACHE_DIR / f"_raw_{tag}_{mood}.mp3"
    mg_result = _generate_with_musicgen(mood, duration_sec, musicgen_raw, progress_cb)

    if mg_result is not None and mg_result.exists():
        try:
            _trim_or_loop_to_duration(mg_result, duration_sec, final_path)
            mg_result.unlink(missing_ok=True)
            log.info(f"✅ Music ready (MusicGen): {final_path.name}")
            return final_path
        except Exception as e:
            log.exception(f"Post-process of MusicGen output failed: {e}")

    # Tier 2: Library
    log.info("Falling back to local library...")
    if progress_cb:
        progress_cb("falling back to library...")
    pick = _pick_library_track(mood)
    if pick is not None:
        try:
            _trim_or_loop_to_duration(pick, duration_sec, final_path)
            log.info(f"✅ Music ready (library): {final_path.name}")
            return final_path
        except Exception as e:
            log.exception(f"Post-process of library track failed: {e}")

    log.warning("No music available — neither MusicGen nor library succeeded")
    if progress_cb:
        progress_cb("no music available — proceeding silent")
    return None


# ==============================================================================
# STANDALONE TEST
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    mood = sys.argv[1] if len(sys.argv) > 1 else "cheerful"
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0

    print(f"\nTesting music generation: mood={mood}, duration={duration}s\n")
    result = generate_music(
        mood=mood,
        duration_sec=duration,
        script_id="test",
        progress_cb=lambda m: print(f"  · {m}"),
    )

    if result:
        print(f"\n✅ Music ready: {result}")
        print(f"   Size: {result.stat().st_size/1024:.1f} KB")
    else:
        print("\n❌ No music generated.")
        print(f"   Drop MP3 files into {MUSIC_LIB_DIR / mood}/ for library fallback.")
