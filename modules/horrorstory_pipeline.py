"""
Claw Bot — Horror Story Pipeline (orchestrator)

Renders an approved horror story:
  1. VoxCPM Voice Design — one deep ominous narrator voices every beat; the per-
     beat audio spans give exact scene timing (narration = source of truth).
  2. Photorealistic still per beat (image_backend, no character lock).
  3. horror_assembly — Ken Burns + narration + ambient drone → 16x9 + watermark.

Front-end agnostic (returns paths); claw_bot / dashboard post the result.
"""

import json
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

# The LLM writes camera-framing cues into image_prompt, but USO/Flux under-weight
# them (esp. against a full-body character ref). Amplify: a forceful prompt prefix
# + matching negatives when a close-up is asked for; other beats keep their
# natural wide/medium framing untouched.
_CLOSEUP_NEG = ("full body, full figure, whole body, wide shot, long shot, "
                "distant framing, small in frame, environment dominates frame")


def _framing_hint(beat: dict) -> tuple[str, str]:
    """Return (prompt_prefix, extra_negatives) from the beat's framing cue.
    Empty when no strong cue is present."""
    ip = (beat.get("image_prompt") or "").lower()
    if "extreme close" in ip:
        return ("Extreme close-up on the character's face filling the frame, "
                "shallow depth of field. ", _CLOSEUP_NEG)
    if any(c in ip for c in ("close-up", "closeup", "close up", "close on")):
        return ("Tight close-up shot: the character's face and shoulders fill "
                "most of the frame, shallow depth of field. ", _CLOSEUP_NEG)
    return ("", "")


def _scene_prompt(beat: dict, loc_map: dict, char_look: dict) -> str:
    """Beat prompt: scene + location + character look tokens + photoreal suffix.
    Character consistency comes from these prompt tokens (locked_token) — no
    per-character LoRA. Same prompt-driven approach as the kids mode."""
    suffix = (get_style_description(PHOTO_STYLE_ID) or {}).get("prompt_suffix", "")
    parts = [beat.get("image_prompt", "").strip()]
    loc = loc_map.get(beat.get("location", ""))
    if loc:
        parts.append(loc)
    for nm in beat.get("characters", []):
        tok = char_look.get(nm)
        if tok:
            parts.append(f"{nm}: {tok}")
    if suffix:
        parts.append(suffix)
    return ", ".join(p for p in parts if p)


def _flux_ckpt_ready() -> bool:
    """True if the flux_lora backend's checkpoint is installed. We render horror
    stills on Flux + a fixed STYLE LoRA (Horrorstyle). No character LoRA needed.
    Falls back to the active backend (Z-Image) when the flux ckpt is absent."""
    try:
        from modules import model_registry
        cfg = model_registry.get_available("image_backend", "comfyui_flux_lora")
        if not cfg:
            return False
        ckpt = cfg.get("ckpt_name", "")
        comfy = PROJECT_ROOT / "01_ComfyUI"
        return bool(ckpt) and any(comfy.rglob(ckpt))
    except Exception:
        return False


def _make_image_backend(use_flux: bool):
    if use_flux:
        import importlib
        from modules import model_registry
        cfg = model_registry.get_available("image_backend", "comfyui_flux_lora")
        mod = importlib.import_module(cfg["module_path"])
        return mod.Backend(cfg)
    return image_backend.get_active_backend()


def _char_portrait(img_backend, char: dict, out_dir: Path, idx: int) -> Optional[Path]:
    """Render a neutral, photoreal head-and-shoulders reference of ONE character
    on a plain background (fixed seed = stable). USO copies IDENTITY from this
    ref while each BEAT's own prompt drives pose/action/camera. Anchoring to this
    (not a full scene still) is what stops every shot inheriting the anchor's
    pose — a scene anchor drags its composition into every beat."""
    name = (char.get("name") or "").strip()
    tok = (char.get("locked_token") or char.get("appearance") or "").strip()
    if not name:
        return None
    style_suffix = (get_style_description(PHOTO_STYLE_ID) or {}).get("prompt_suffix", "")
    prompt = (
        f"Photorealistic neutral character reference portrait of {name}: {tok}. "
        f"Single person alone, front-facing, calm neutral expression, head and "
        f"shoulders centered, plain solid dark-gray studio background, even soft "
        f"lighting, no props, no scene, no other people."
    )
    if style_suffix:
        prompt += f" {style_suffix}"
    try:
        p = img_backend.generate(
            prompt=prompt,
            negative_prompt=("full body, action pose, scene, landscape, multiple "
                             "people, crowd, text, watermark, blurry, deformed"),
            aspect_ratio="1:1",
            seed=90210 + idx,
            output_filename=str(out_dir / f"_ref_{name.lower().replace(' ', '_')}.png"),
        )
        return Path(p)
    except Exception as e:
        log.warning(f"Horror char portrait for {name} failed: {e}")
        return None


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


def _kokoro_narrate(narrations: list, out_path: Path, gap_sec: float = 0.45,
                    voice: Optional[str] = None, speed: Optional[float] = None):
    """Voice each beat with Kokoro (a fixed preset voice → identical narrator
    every beat, zero drift), stitch with a silence gap. Returns (master_wav,
    spans) where spans = [(text, t_start, t_end)] — same shape as VoxCPM path.

    voice/speed default to the horror settings; callers (e.g. facts mode) pass
    their own for a different delivery."""
    import numpy as np
    import soundfile as sf
    from modules.tts_engine import TTSEngine

    voice = voice or rs.get_horror_voice()
    speed = speed if speed is not None else rs.get_horror_voice_speed()
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


def _horror_reference_wav() -> Path:
    """Fixed deep reference clip Chatterbox clones for a consistent narrator.
    Uses 04_Outputs/audio/horror_ref.wav; generates one (Kokoro am_adam) if
    missing. Replace this file with a real deep-voice recording for best results."""
    ref = NARRATION_DIR / "horror_ref.wav"
    if not ref.exists():
        from modules.tts_engine import TTSEngine
        NARRATION_DIR.mkdir(parents=True, exist_ok=True)
        TTSEngine(voice="am_adam", speed=0.9).synthesize(
            "Listen closely, for this is a story you will not soon forget.",
            output_path=ref, voice="am_adam", speed=0.9)
    return ref


def _narrate_continuous(narrations: list, out_path: Path, voice_design: str = "") -> tuple:
    """Render the whole narration → (wav, mode_label). Engine order:
    qwen (ComfyUI Qwen3-TTS, production — once installed) -> chatterbox
    (expressive, clones one fixed ref) -> kokoro (preset, consistent) -> voxcpm.
    The visual cuts are placed later on detected silences."""
    engine = rs.get_horror_voice_engine()
    full_text = " ".join(t.strip() for t in narrations if t.strip())

    if engine == "qwen":
        # Production narrator: Qwen3-TTS Custom Voice (preset = deterministic =
        # consistent), driven in an isolated venv via the tts_qwen bridge. Each
        # beat voiced with the same preset speaker -> no drift.
        try:
            from modules import tts_qwen
            spk = rs.get_horror_qwen_speaker()
            wav, _ = tts_qwen.synthesize_segments(
                narrations, output_path=out_path, speaker=spk,
                instruct=rs.get_horror_qwen_instruct(),
                pitch=rs.get_horror_qwen_pitch())
            return Path(wav), f"qwen3-tts:{spk}"
        except Exception as e:
            log.warning(f"Qwen3-TTS failed ({e}); falling back to Chatterbox/Kokoro.")

    if engine == "chatterbox":
        # Expressive + consistent: every beat cloned from ONE fixed reference
        # (isolated venv → no ComfyUI risk). Per-beat groups keep each gen short.
        try:
            from modules import tts_chatterbox as cb
            ref = _horror_reference_wav()
            groups = [(t, str(ref), None) for t in narrations if t.strip()]
            wav, _ = cb.synthesize_segments(
                groups, output_path=out_path,
                exaggeration=rs.get_horror_chatterbox_exaggeration())
            return Path(wav), "chatterbox"
        except Exception as e:
            log.warning(f"Chatterbox failed ({e}); falling back to Kokoro.")

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


def _spans_path(wav: Path) -> Path:
    """Sidecar JSON holding real per-beat timing next to the narration wav."""
    return Path(wav).with_suffix(".spans.json")


def _render_narration(narrations: list, out_path: Path, voice_design: str = "") -> tuple:
    """Render narration and, when the engine exposes it, the REAL per-beat timing.

    Returns (wav, spans_or_None, mode). spans = [(text, t_start, t_end)] — one per
    beat, from the TTS itself (not a silence guess). These are the source of truth
    for scene windows + subtitle timing, so images/subs/video lock to the voice.

    kokoro  -> per-beat synth (each beat voiced separately) => exact spans.
    qwen    -> synthesize_segments already returns per-segment spans.
    others  -> fall back to one continuous render (spans=None => silence-split)."""
    engine = rs.get_horror_voice_engine()
    if engine == "kokoro":
        wav, spans = _kokoro_narrate(narrations, out_path)
        return Path(wav), spans, f"kokoro:{rs.get_horror_voice()}"
    if engine == "qwen":
        try:
            from modules import tts_qwen
            spk = rs.get_horror_qwen_speaker()
            wav, spans = tts_qwen.synthesize_segments(
                narrations, output_path=out_path, speaker=spk,
                instruct=rs.get_horror_qwen_instruct(),
                pitch=rs.get_horror_qwen_pitch())
            return Path(wav), spans, f"qwen3-tts:{spk}"
        except Exception as e:
            log.warning(f"Qwen3-TTS spans failed ({e}); continuous fallback.")
    wav, mode = _narrate_continuous(narrations, out_path, voice_design)
    return Path(wav), None, mode


def _load_spans(wav: Path) -> Optional[list]:
    """Load the per-beat spans sidecar for an already-rendered narration wav."""
    sp = _spans_path(wav)
    if not sp.exists():
        return None
    try:
        data = json.loads(sp.read_text(encoding="utf-8"))
        return [(t, float(a), float(b)) for t, a, b in data]
    except Exception as e:
        log.warning(f"spans sidecar unreadable ({e}); silence-split fallback.")
        return None


def narrate_story(story: dict, progress_cb: Optional[Callable[[str], None]] = None) -> Path:
    """Render JUST the full narration audio (no visuals). Lets callers post the
    story audio before the multi-hour image render. Returns the wav path.

    Also writes a per-beat spans sidecar (when the engine gives real timing) so
    a later render_horror(narration_path=...) reuses the exact windows instead of
    re-guessing them from silence."""
    horror_id = story.get("horror_id") or story.get("_id")
    beats = story.get("beats", [])
    if not beats:
        raise ValueError("Horror story has no beats.")
    gpu_utils.ensure_vram_free()
    NARRATION_DIR.mkdir(parents=True, exist_ok=True)
    out = NARRATION_DIR / f"horror_{horror_id}.wav"
    out, spans, mode = _render_narration(
        [b.get("narration", "") for b in beats], out, story.get("voice_design", ""))
    if spans:
        _spans_path(out).write_text(json.dumps(spans), encoding="utf-8")
    if progress_cb:
        progress_cb(f"narration ready ({mode})")
    return out


def render_horror(
    story: dict,
    progress_cb: Optional[Callable[[str], None]] = None,
    narration_path: Optional[Path] = None,
) -> dict:
    """Full render: continuous narration -> silence-split -> photoreal stills
    -> 16x9 assembly (Ken Burns + ambient + watermark). Pass `narration_path` to
    reuse already-rendered narration (skips re-voicing)."""
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
    spans = None
    if narration_path and Path(narration_path).exists():
        narration_path = Path(narration_path)
        spans = _load_spans(narration_path)
        _p("🎙️ using pre-rendered narration" + (" (+real per-beat spans)" if spans else ""))
    else:
        narration_path = NARRATION_DIR / f"horror_{horror_id}.wav"
        _p(f"🎙️ narrating full story ({len(beats)} beats)...")
        narration_path, spans, mode = _render_narration(
            [b.get("narration", "") for b in beats], narration_path, voice_design)
        if spans:
            _spans_path(narration_path).write_text(json.dumps(spans), encoding="utf-8")
    total_dur = hasm._probe_duration(narration_path)
    _p(f"🎙️ narration ready: {total_dur:.1f}s")

    # Per-beat windows. PREFERRED: the TTS's own per-beat spans (exact — every
    # image/subtitle/clip locks to the real voice, no drift, video len = narration).
    # FALLBACK (engine gave no spans): snap proportional boundaries to silences.
    if spans and len(spans) == len(beats):
        durations = _scene_durations(spans, total_dur)
        _p(f"🎯 {len(durations)} windows from real per-beat narration spans")
    else:
        from modules import audio_segmenter
        weights = [max(1, len(b.get("narration", "").split())) for b in beats]
        try:
            durations = audio_segmenter.plan_windows_from_silence(narration_path, weights)
            _p(f"✂️ split on {len(durations)} silence-aligned windows")
        except Exception as e:
            log.warning(f"silence split failed ({e}); falling back to proportional.")
            wsum = sum(weights)
            durations = [round(w / wsum * total_dur, 3) for w in weights]

    # ---- 2. Horror stills (one per beat) — Flux + a fixed STYLE LoRA
    #         (Horrorstyle). Character consistency = prompt tokens, NOT a trained
    #         per-character LoRA (dropped: costly + unstable training faces). The
    #         lora_autotrain setup stays in the repo for later reuse. ----
    use_flux = _flux_ckpt_ready()
    # USO mode (default): lock each recurring character with a neutral photoreal
    # PORTRAIT ref (built once), then reference the cast present in each beat so
    # identity stays consistent while the beat's OWN prompt drives pose/action/
    # camera. (Anchoring to a full scene still instead drags that pose into every
    # shot.) USO off / unhealthy -> Flux + Horrorstyle LoRA (or the active
    # Z-Image backend), prompt-token consistency only. Falls back cleanly.
    use_uso = rs.get_uso_mode_enabled()
    img = None
    if use_uso:
        try:
            img = image_backend.get_named_backend("comfyui_uso")
            ok, _msg = img.health_check()
            if not ok:
                _p(f"USO unhealthy ({_msg}); horror stills use Flux/active backend.")
                use_uso = False
        except Exception as e:
            _p(f"USO unavailable ({e}); horror stills use Flux/active backend.")
            use_uso = False

    style_kw = {}
    if not use_uso and use_flux:
        style = rs.get_horror_style_lora()
        if style:
            style_kw = {
                "use_char_lora": False,
                "extra_loras": [{"lora_file": style,
                                 "weight": rs.get_horror_style_lora_weight()}],
            }
    if use_uso:
        _mode_label = "USO Flux.1-dev + USO LoRA, anchor consistency"
    elif use_flux:
        _mode_label = "flux + Horrorstyle LoRA, prompt-token consistency"
    else:
        _mode_label = "active backend (Z-Image), prompt-token consistency"
    _p(f"🖼️ rendering {len(beats)} stills ({_mode_label})...")
    gpu_utils.free_comfyui_vram()
    if not use_uso:
        img = _make_image_backend(use_flux)
    loc_map = {lc.get("name", ""): lc.get("description", "")
               for lc in story.get("locations", [])}
    char_look = {c.get("name", ""): c.get("locked_token", "")
                 for c in story.get("characters", [])}
    out_dir = STILLS_DIR / f"horror_{horror_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build one neutral portrait ref per recurring character (USO identity lock).
    char_refs: dict[str, Path] = {}
    if use_uso:
        _p("🎭 building character reference portraits...")
        for idx, c in enumerate(story.get("characters", [])):
            ref = _char_portrait(img, c, out_dir, idx)
            if ref is not None:
                char_refs[(c.get("name") or "").strip()] = ref

    scene_images: list[Path] = []
    for i, b in enumerate(beats):
        _p(f"🖼️ still {i+1}/{len(beats)}...")
        _fpref, _fneg = _framing_hint(b)
        gen_kwargs = dict(
            prompt=_fpref + _scene_prompt(b, loc_map, char_look),
            negative_prompt=(_fneg or None),
            aspect_ratio="16:9",
            seed=random.randint(1, 2**31 - 1),
            output_filename=str(out_dir / f"beat_{i:04d}.png"),
        )
        if use_uso:
            # Reference only the cast present in THIS beat; fall back to the
            # protagonist portrait when the beat names no known character.
            refs = [char_refs[n] for n in b.get("characters", []) if n in char_refs]
            if not refs and char_refs:
                refs = list(char_refs.values())[:1]
            if len(refs) >= 2:
                path = img.generate(reference_image=refs[0],
                                    reference_images=refs, **gen_kwargs)
            elif len(refs) == 1:
                path = img.generate(reference_image=refs[0], **gen_kwargs)
            else:
                path = img.generate(**gen_kwargs)  # no cast -> pure text2img
        else:
            path = img.generate(**gen_kwargs, **style_kw)
        scene_images.append(Path(path))

    # ---- 3. Visuals: animated Wan clips per shot, or Ken Burns stills ----
    ambient = rs.get_horror_ambient_enabled()
    if rs.get_horror_video_mode() == "wan":
        _p("🎞️ animating shots (Wan I2V)...")
        from modules import horror_video
        clips = horror_video.render_shot_clips(
            story, scene_images, durations, aspect_ratio="16:9", progress_cb=progress_cb)
        _p("🎬 assembling horror video...")
        gpu_utils.free_comfyui_vram()
        out = hasm.assemble_horror_clips(
            story, narration_path, clips, ambient=ambient, progress_cb=progress_cb)
    else:
        _p("🎬 assembling horror video (Ken Burns)...")
        gpu_utils.free_comfyui_vram()
        out = hasm.assemble_horror(
            story, narration_path, durations, scene_images,
            ambient=ambient, progress_cb=progress_cb)
    out["narration_audio"] = narration_path

    # Upload kit (title + thumbnail) from a clean still — the render carries
    # burned-in subtitles.
    _p("🖼️ building title + thumbnail…")
    try:
        from modules import publish_kit
        video = out.get("16x9") or out.get("9x16")
        if video:
            context = "\n".join(b.get("narration", "")
                                for b in story.get("beats", []))[:1500]
            kit = publish_kit.attach(
                Path(video),
                fallback_title=story.get("title", "Horror Story"),
                context=context,
                description=(story.get("description") or "").strip(),
                mode="horror story (dark, tense, adult audience)",
                source_image=(Path(scene_images[0])
                              if scene_images else None),
            )
            out["publish"] = kit
            out["title"] = kit.get("title")
            out["thumbnail"] = kit.get("thumb_16x9") or kit.get("thumb_9x16")
    except Exception as e:
        log.warning(f"publish kit skipped: {e}")

    _p("✅ horror video complete")
    return out
