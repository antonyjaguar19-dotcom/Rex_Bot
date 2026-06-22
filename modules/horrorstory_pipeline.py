"""
Claw Bot — Horror Story Pipeline (orchestrator)

Renders an approved horror story:
  1. VoxCPM Voice Design — one deep ominous narrator voices every beat; the per-
     beat audio spans give exact scene timing (narration = source of truth).
  2. Photorealistic still per beat (image_backend, no character lock).
  3. horror_assembly — Ken Burns + narration + ambient drone → 16x9 + watermark.

Front-end agnostic (returns paths); claw_bot / dashboard post the result.
"""

import logging
import random
import sys
from pathlib import Path
from typing import Callable, Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import image_backend
from modules import gpu_utils
from modules import runtime_settings as rs
from modules import horror_assembly as hasm
from modules.script_generator import get_style_description

log = logging.getLogger("claw_bot.horrorstory_pipeline")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
STILLS_DIR = PROJECT_ROOT / "04_Outputs" / "storyboards"
NARRATION_DIR = PROJECT_ROOT / "04_Outputs" / "audio"

PHOTO_STYLE_ID = "photoreal"


def _scene_prompt(beat: dict, loc_map: dict, char_look: dict, use_lora: bool) -> str:
    """Beat prompt: scene + location description (+ character look tokens when no
    LoRA) + photoreal suffix. With LoRA, the character NAME in image_prompt makes
    flux_lora auto-stack the identity, so we skip the verbose look tokens."""
    suffix = (get_style_description(PHOTO_STYLE_ID) or {}).get("prompt_suffix", "")
    parts = [beat.get("image_prompt", "").strip()]
    loc = loc_map.get(beat.get("location", ""))
    if loc:
        parts.append(loc)
    if not use_lora:
        for nm in beat.get("characters", []):
            tok = char_look.get(nm)
            if tok:
                parts.append(f"{nm}: {tok}")
    if suffix:
        parts.append(suffix)
    return ", ".join(p for p in parts if p)


def _flux_lora_ready() -> bool:
    """True if the flux_lora backend can run: its checkpoint is installed AND at
    least one character LoRA is registered. Else we render with the active backend
    (degraded: no identity lock) so horror still works pre-install."""
    try:
        from modules import model_registry
        from modules.lora import store as lora_store
        if not lora_store.all_characters():
            return False
        cfg = model_registry.get_available("image_backend", "comfyui_flux_lora")
        if not cfg:
            return False
        ckpt = cfg.get("ckpt_name", "")
        comfy = PROJECT_ROOT / "01_ComfyUI"
        return bool(ckpt) and any(comfy.rglob(ckpt))
    except Exception:
        return False


def _make_image_backend(use_lora: bool):
    if use_lora:
        import importlib
        from modules import model_registry
        cfg = model_registry.get_available("image_backend", "comfyui_flux_lora")
        mod = importlib.import_module(cfg["module_path"])
        return mod.Backend(cfg)
    return image_backend.get_active_backend()


def _scene_durations(spans: list, total_dur: float) -> list[float]:
    """Tile the whole narration timeline across beats: beat i covers from its
    own start to the next beat's start (last beat runs to the end). spans items
    are (text, t_start, t_end)."""
    starts = [s[1] for s in spans]
    durs = []
    for i in range(len(starts)):
        nxt = starts[i + 1] if i + 1 < len(starts) else total_dur
        durs.append(max(0.5, round(nxt - starts[i], 3)))
    return durs


def _kokoro_narrate(narrations: list, out_path: Path, gap_sec: float = 0.45):
    """Voice each beat with Kokoro (a fixed preset voice → identical narrator
    every beat, zero drift), stitch with a silence gap. Returns (master_wav,
    spans) where spans = [(text, t_start, t_end)] — same shape as VoxCPM path."""
    import numpy as np
    import soundfile as sf
    from modules.tts_engine import TTSEngine

    voice = rs.get_horror_voice()
    speed = rs.get_horror_voice_speed()
    eng = TTSEngine(voice=voice, speed=speed)
    tmp = NARRATION_DIR / "_horror_beats"
    tmp.mkdir(parents=True, exist_ok=True)

    sr = None
    pieces, spans, cursor = [], [], 0
    for i, text in enumerate(narrations):
        wpath = eng.synthesize(text, output_path=tmp / f"b_{i:04d}.wav",
                               voice=voice, speed=speed)
        wav, this_sr = sf.read(str(wpath), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        sr = sr or this_sr
        if sr is None:
            sr = this_sr
        t0 = cursor / sr
        pieces.append(wav)
        cursor += len(wav)
        spans.append((text, round(t0, 3), round(cursor / sr, 3)))
        if i < len(narrations) - 1:
            gap = np.zeros(int(round(gap_sec * sr)), dtype="float32")
            pieces.append(gap)
            cursor += len(gap)

    master = np.concatenate(pieces) if pieces else np.zeros(1, dtype="float32")
    sf.write(str(out_path), master, sr or 24000)
    return out_path, spans


def _narrate_continuous(full_text: str, out_path: Path, voice_design: str = "") -> tuple:
    """Render the WHOLE narration in one pass → (wav, mode_label). Engine order:
    qwen (ComfyUI Qwen3-TTS, production — once installed) -> kokoro (preset,
    consistent, in-process) -> voxcpm. Continuous render = natural prosody; the
    visual cuts are placed later on detected silences."""
    engine = rs.get_horror_voice_engine()

    if engine == "qwen":
        # Production narrator: Qwen3-TTS via a dedicated ComfyUI adapter
        # (comfyui_qwen_tts) rendering the full text with a consistent Custom
        # Voice. Wired once the TTS-Audio-Suite node + its API workflow are
        # installed/exported; until then fall through to Kokoro.
        try:
            from modules.tts_qwen import synthesize_full  # built after install
            wav = synthesize_full(full_text, out_path)
            return Path(wav), "qwen3-tts"
        except Exception as e:
            log.warning(f"Qwen3-TTS not wired yet ({e}); falling back to Kokoro.")

    if engine == "voxcpm":
        try:
            from modules.tts_voxcpm import VoxCPMTTS
            tts = VoxCPMTTS(inference_timesteps=28)
            wav = tts.synthesize(full_text, output_path=out_path)
            return Path(wav), "voxcpm:whole"
        except Exception as e:
            log.warning(f"VoxCPM whole-text failed ({e}); falling back to Kokoro.")

    # Kokoro: fixed preset voice → fully consistent, in-process, handles long text.
    from modules.tts_engine import TTSEngine
    voice = rs.get_horror_voice()
    speed = rs.get_horror_voice_speed()
    wav = TTSEngine(voice=voice, speed=speed).synthesize(
        full_text, output_path=out_path, voice=voice, speed=speed)
    return Path(wav), f"kokoro:{voice}"


def render_horror(
    story: dict,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Full render: continuous narration -> silence-split -> photoreal stills
    -> 16x9 assembly (Ken Burns + ambient + watermark)."""
    def _p(msg: str):
        log.info(msg)
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    horror_id = story.get("horror_id") or story.get("_id")
    beats = story.get("beats", [])
    if not beats:
        raise ValueError("Horror story has no beats.")
    voice_design = story.get("voice_design", "")

    # ---- 1. Narration: ONE continuous render (natural prosody, consistent
    #         voice), then cut the visuals on the audio's real silences. ----
    gpu_utils.ensure_vram_free()
    NARRATION_DIR.mkdir(parents=True, exist_ok=True)
    narration_path = NARRATION_DIR / f"horror_{horror_id}.wav"
    full_text = " ".join(b.get("narration", "").strip() for b in beats).strip()
    _p(f"🎙️ narrating full story ({len(beats)} beats, continuous)...")
    narration_path, mode = _narrate_continuous(full_text, narration_path, voice_design)
    total_dur = hasm._probe_duration(narration_path)
    _p(f"🎙️ narration ready: {total_dur:.1f}s ({mode})")

    # Per-beat windows snapped to detected silences (images = silent splits).
    from modules import audio_segmenter
    weights = [max(1, len(b.get("narration", "").split())) for b in beats]
    try:
        durations = audio_segmenter.plan_windows_from_silence(narration_path, weights)
        _p(f"✂️ split on {len(durations)} silence-aligned windows")
    except Exception as e:
        log.warning(f"silence split failed ({e}); falling back to proportional.")
        wsum = sum(weights)
        durations = [round(w / wsum * total_dur, 3) for w in weights]

    # ---- 2a. Character LoRAs (identity lock) — train any missing, gated ----
    gpu_utils.free_comfyui_vram()
    if story.get("characters"):
        try:
            from modules import lora_autotrain
            lora_autotrain.ensure_character_loras(story, progress_cb=progress_cb)
        except Exception as e:
            log.warning(f"LoRA auto-train skipped: {e}")

    # ---- 2b. Photoreal stills (one per beat) ----
    use_lora = _flux_lora_ready()
    _p(f"🖼️ rendering {len(beats)} stills "
       f"({'flux_lora identity-locked' if use_lora else 'active backend (no LoRA)'})...")
    gpu_utils.free_comfyui_vram()
    img = _make_image_backend(use_lora)
    loc_map = {lc.get("name", ""): lc.get("description", "")
               for lc in story.get("locations", [])}
    char_look = {c.get("name", ""): c.get("locked_token", "")
                 for c in story.get("characters", [])}
    out_dir = STILLS_DIR / f"horror_{horror_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_images: list[Path] = []
    for i, b in enumerate(beats):
        _p(f"🖼️ still {i+1}/{len(beats)}...")
        path = img.generate(
            prompt=_scene_prompt(b, loc_map, char_look, use_lora),
            aspect_ratio="16:9",
            seed=random.randint(1, 2**31 - 1),
            output_filename=str(out_dir / f"beat_{i:04d}.png"),
        )
        scene_images.append(Path(path))

    # ---- 3. Assemble (16x9 + ambient + watermark) ----
    _p("🎬 assembling horror video...")
    gpu_utils.free_comfyui_vram()
    out = hasm.assemble_horror(
        story, narration_path, durations, scene_images,
        ambient=rs.get_horror_ambient_enabled(), progress_cb=progress_cb,
    )
    out["narration_audio"] = narration_path
    _p("✅ horror video complete")
    return out
