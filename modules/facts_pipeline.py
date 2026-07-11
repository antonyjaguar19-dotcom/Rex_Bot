"""
Claw Bot — Facts Shorts Pipeline (orchestrator)

Renders a facts reel:
  1. Narration — kokoro voices each beat separately; the real per-beat spans give
     exact scene timing + perfectly-synced on-screen text (reuses the horror
     pipeline's span machinery).
  2. Loose mood backdrop per beat (active image backend; NO character refs — there
     is no recurring subject). Falls back to a generated gradient if the backend
     is unavailable, so the reel renders even with ComfyUI down.
  3. facts_assembly — Ken Burns + BIG centered fact text + optional music → 9x16.

Front-end agnostic (returns paths).
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import image_backend
from modules import gpu_utils
from modules import runtime_settings as rs
from modules import facts_assembly as fasm
from modules.assembly import ASPECTS
# Reuse the horror pipeline's kokoro per-beat narration + span helpers. Facts
# voices with its OWN bright/fast voice (not horror's deep narrator).
from modules.horrorstory_pipeline import (
    _kokoro_narrate, _load_spans, _spans_path, _scene_durations,
)


def _voice_facts(narrations: list, out_path: Path):
    """Kokoro narration in the FACTS voice (bright + slightly fast = energetic).
    Returns (wav, spans)."""
    return _kokoro_narrate(narrations, out_path,
                           voice=rs.get_facts_voice(),
                           speed=rs.get_facts_voice_speed())

log = logging.getLogger("claw_bot.facts_pipeline")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
STILLS_DIR = PROJECT_ROOT / "04_Outputs" / "storyboards"
NARRATION_DIR = PROJECT_ROOT / "04_Outputs" / "audio"

# palette cycled for gradient fallbacks (dark, punchy, readable under white text)
_GRADIENTS = [
    ("0x1a1a2e", "0x16213e"), ("0x2b1055", "0x7597de"), ("0x0f2027", "0x2c5364"),
    ("0x232526", "0x414345"), ("0x141e30", "0x243b55"), ("0x3a1c71", "0xd76d77"),
]


# S2V renders at its native window size; assembly scales each clip onto the
# final 9x16 / 16x9 / 1x1 canvas. Rendering S2V at full 1080x1920 would blow
# VRAM and time for no gain.
_S2V_DIMS = {"9x16": (480, 832), "16x9": (832, 480), "1x1": (624, 624)}


def _probe_dur(path: Path) -> float:
    try:
        out = subprocess.run(
            [str(fasm.FFMPEG_EXE).replace("ffmpeg.exe", "ffprobe.exe"),
             "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
             str(path)], capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def _trim_video(src: Path, dur: float, dst: Path) -> Path:
    """Trim a clip to exactly `dur` seconds (drops S2V's frame-rounding tail so
    the concatenated reel stays in lock-step with the narration)."""
    if dur <= 0:
        return src
    r = subprocess.run(
        [str(fasm.FFMPEG_EXE), "-y", "-loglevel", "error", "-i", str(src),
         "-t", f"{dur:.3f}", "-c:v", "libx264", "-c:a", "aac", "-pix_fmt",
         "yuv420p", str(dst)], capture_output=True, text=True, timeout=300)
    return dst if dst.exists() and r.returncode == 0 else src


def _concat_wavs(wavs: list, dst: Path) -> Path:
    lst = dst.with_suffix(".txt")
    lst.write_text("".join(f"file '{w.as_posix()}'\n" for w in wavs), encoding="utf-8")
    subprocess.run(
        [str(fasm.FFMPEG_EXE), "-y", "-loglevel", "error", "-f", "concat",
         "-safe", "0", "-i", str(lst), "-c", "copy", str(dst)],
        capture_output=True, text=True, timeout=120)
    if not dst.exists():   # copy can fail on mismatched wavs — re-encode
        subprocess.run(
            [str(fasm.FFMPEG_EXE), "-y", "-loglevel", "error", "-f", "concat",
             "-safe", "0", "-i", str(lst), "-ar", "24000", str(dst)],
            capture_output=True, text=True, timeout=120)
    return dst


def _slice_wav(src: Path, start: float, end: float, dst: Path) -> Path:
    subprocess.run(
        [str(fasm.FFMPEG_EXE), "-y", "-loglevel", "error", "-i", str(src),
         "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-c", "copy", str(dst)],
        capture_output=True, text=True, timeout=60)
    if not dst.exists():   # stream-copy can't always cut cleanly — re-encode
        subprocess.run(
            [str(fasm.FFMPEG_EXE), "-y", "-loglevel", "error", "-i", str(src),
             "-ss", f"{start:.3f}", "-to", f"{end:.3f}", str(dst)],
            capture_output=True, text=True, timeout=60)
    return dst


def _voice_beats_mascot(narrations: list, out_dir: Path, _p) -> list:
    """One WAV per fact, in the mascot's voice.

    Qwen3-TTS is called ONCE for all the beats (the model loads in an isolated
    venv subprocess — per-beat calls would reload it every time) and the master
    is sliced back apart on the returned spans. Kokoro is the fallback so a dead
    bridge never blocks a render.
    """
    engine = rs.get_mascot_tts_engine()
    if engine == "qwen":
        try:
            from modules import tts_qwen
            ok, msg = tts_qwen.health_check()
            if not ok:
                raise RuntimeError(msg)
            speaker = rs.get_mascot_voice()
            _p(f"🎙️ mascot voice: qwen3-tts / {speaker}")
            master = out_dir / "mascot_master.wav"
            _, spans = tts_qwen.synthesize_segments(
                narrations, output_path=master, speaker=speaker,
                instruct=rs.get_mascot_voice_instruct())
            if len(spans) == len(narrations):
                return [_slice_wav(master, a, b, out_dir / f"beat_{i:02d}.wav")
                        for i, (_, a, b) in enumerate(spans)]
            _p(f"⚠️ qwen returned {len(spans)} spans for {len(narrations)} beats; "
               f"using kokoro.")
        except Exception as e:
            _p(f"⚠️ mascot voice (qwen) failed ({e}); using kokoro.")

    from modules.tts_engine import TTSEngine
    tts = TTSEngine()
    voice = rs.get_facts_voice()
    wavs = []
    for i, text in enumerate(narrations):
        wp = out_dir / f"beat_{i:02d}.wav"
        tts.synthesize(text, output_path=wp, voice=voice)
        wavs.append(wp)
    return wavs


def _render_facts_mascot(story, beats, aspect, music_path, _p, facts_id):
    """Every fact presented by the costumed mascot, lip-synced (Qwen still → S2V).

    VRAM discipline (one big model at a time on a 16 GB card):
      1. LLM writes all scenes (Ollama), then unloads.
      2. Qwen-Edit renders all mascot stills warm, then releases.
      3. S2V renders all talking clips warm, then releases.
      4. ffmpeg concat + music + captions (no GPU).
    """
    from modules import mascot, gpu_memory, video_backend as vb, model_registry as mr
    from modules.tts_engine import TTSEngine

    ok, why = mascot.is_available()
    if not ok:
        _p(f"mascot mode off: {why}")
        return None
    cfg = mr.get_available("video_backend", "comfyui_wan22_s2v")
    if not cfg:
        _p("mascot mode off: S2V backend not registered.")
        return None

    topic = story.get("topic", "") or story.get("title", "")
    out_dir = STILLS_DIR / f"facts_{facts_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sw, sh = _S2V_DIMS.get(aspect, _S2V_DIMS["9x16"])
    ar = aspect.replace("x", ":")

    # 1. Scenes (Ollama). A user-edited mascot_scene wins; otherwise the LLM
    #    writes one and we STORE it back so it can be edited next time. Written up
    #    front so the LLM unloads before Qwen loads.
    from modules import facts_writer as _fw
    _p(f"🎭 writing {len(beats)} mascot scenes…")
    scenes = [(b.get("mascot_scene") or "").strip() for b in beats]
    if not all(scenes):
        with gpu_memory.llm():
            for i, b in enumerate(beats):
                if not scenes[i]:
                    scenes[i] = mascot.explainer_scene(b.get("narration", ""), topic)
        for i, sc in enumerate(scenes):   # persist for editing / reuse
            try:
                _fw.set_beat_prompt(facts_id, i, "mascot_scene", sc)
            except Exception:
                pass

    # 2. Per-beat narration WAVs. The mascot is a young jaguar cub — Kokoro reads
    #    it flat and pitch-shifting it sounds artificial, so the mascot gets
    #    Qwen3-TTS with an emotion instruct (natural, expressive, deterministic =
    #    the same voice every clip). Falls back to Kokoro if the bridge is down.
    _p("🎙️ voicing each fact…")
    narrations = [b.get("narration", "") for b in beats]
    wavs = _voice_beats_mascot(narrations, out_dir, _p)

    # 3. Mascot stills via Qwen-Edit (identity held), warm across all beats.
    _p(f"🖼️ rendering {len(beats)} mascot stills (Qwen-Edit)…")
    stills = []
    gpu_memory.acquire(gpu_memory.QWEN_EDIT)
    try:
        for i, sc in enumerate(scenes):
            sp = out_dir / f"still_{i:02d}.png"
            got = mascot.render_scene(sc, sp, aspect=aspect, seed=1000 + i)
            stills.append(got or _gradient_bg(i, *ASPECTS.get(aspect, ASPECTS["9x16"]),
                                              out_dir / f"still_{i:02d}.png"))
    finally:
        gpu_memory.release(gpu_memory.QWEN_EDIT)

    # 4. S2V talking clips, warm across all beats.
    _p(f"🎬 animating {len(beats)} talking clips (Wan S2V)…")
    cfg = dict(cfg); cfg["_id"] = "comfyui_wan22_s2v"
    s2v = vb.build_backend(cfg)
    ok, msg = s2v.health_check()
    if not ok:
        _p(f"mascot mode off: S2V unhealthy ({msg}).")
        return None
    clips = []
    gpu_memory.acquire(gpu_memory.WAN_VIDEO)   # evicts Qwen/Ollama; S2V ≈ Wan size
    try:
        for i, b in enumerate(beats):
            raw = s2v.generate(
                prompt=scenes[i], input_image=Path(stills[i]),
                audio_path=wavs[i], aspect_ratio=ar, width=sw, height=sh,
                output_filename=f"s2v_{facts_id}_{i:02d}.mp4", seed=2000 + i)
            trimmed = _trim_video(Path(raw), _probe_dur(wavs[i]),
                                  out_dir / f"clip_{i:02d}.mp4")
            clips.append(trimmed)
            _p(f"  clip {i+1}/{len(beats)} done")
    finally:
        gpu_memory.release(gpu_memory.WAN_VIDEO)

    # 4b. Optional 4x upscale of each talking clip (Real-ESRGAN anime + polish),
    #     at the clip's native 480p where the detail gain is real. ComfyUI is free
    #     now (S2V released). Best-effort — a failed upscale keeps the original.
    if rs.get_upscale_enabled():
        from modules import upscaler
        _p(f"🔎 upscaling {len(clips)} clips (4x)…")
        for i, c in enumerate(clips):
            try:
                upscaler.upscale_clip(Path(c))
                _p(f"  upscaled {i+1}/{len(clips)}")
            except Exception as e:
                _p(f"  upscale {i+1} failed ({e}); keeping original")

    # 5. Narration = the exact per-beat WAVs concatenated, so the track the
    #    assembler overlays matches each clip's own audio frame-for-frame.
    narration = _concat_wavs(wavs, out_dir / "narration.wav")

    # 6. Concat clips + big captions (timed to real clip durations) + music.
    _p("🎬 assembling mascot reel…")
    return fasm.assemble_facts_clips(story, narration, clips, aspect=aspect,
                                     music_path=music_path, progress_cb=_p)


def _gradient_bg(idx: int, w: int, h: int, out_path: Path) -> Path:
    """Generate one gradient still (pure ffmpeg) — image-independent fallback."""
    c0, c1 = _GRADIENTS[idx % len(_GRADIENTS)]
    cmd = [
        str(fasm.FFMPEG_EXE), "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"gradients=s={w}x{h}:c0={c0}:c1={c1}:x0=0:y0=0:x1={w}:y1={h}",
        "-frames:v", "1", str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not out_path.exists():
        # last-ditch: flat colour
        subprocess.run([str(fasm.FFMPEG_EXE), "-y", "-loglevel", "error",
                        "-f", "lavfi", "-i", f"color=c={c0}:s={w}x{h}",
                        "-frames:v", "1", str(out_path)], timeout=120)
    return out_path


def narrate_facts(story: dict, progress_cb: Optional[Callable[[str], None]] = None) -> Path:
    """Voice the whole reel (kokoro per-beat) + write the spans sidecar. Returns wav."""
    facts_id = story.get("facts_id") or story.get("_id")
    beats = story.get("beats", [])
    if not beats:
        raise ValueError("Facts reel has no beats.")
    gpu_utils.ensure_vram_free()
    NARRATION_DIR.mkdir(parents=True, exist_ok=True)
    out = NARRATION_DIR / f"facts_{facts_id}.wav"
    out, spans = _voice_facts([b.get("narration", "") for b in beats], out)
    if spans:
        _spans_path(out).write_text(__import__("json").dumps(spans), encoding="utf-8")
    if progress_cb:
        progress_cb(f"narration ready (kokoro:{rs.get_facts_voice()})")
    return out


def render_facts(
    story: dict,
    progress_cb: Optional[Callable[[str], None]] = None,
    narration_path: Optional[Path] = None,
    aspect: str = "9x16",
    music_path: Optional[Path] = None,
    animate: Optional[bool] = None,
) -> dict:
    """Full render: kokoro narration -> per-beat spans -> mood backdrops -> 9x16
    Ken Burns + big centered fact text."""
    def _p(msg: str):
        log.info(msg)
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    facts_id = story.get("facts_id") or story.get("_id")
    beats = story.get("beats", [])
    if not beats:
        raise ValueError("Facts reel has no beats.")

    # Mascot mode: the mascot presents every fact in costume, lip-synced (S2V).
    # Falls back to the normal abstract-backdrop path if the mascot or S2V is
    # unavailable, so enabling it never breaks a render.
    if rs.get_facts_mascot_mode():
        try:
            out = _render_facts_mascot(story, beats, aspect, music_path, _p, facts_id)
            if out:
                return out
            _p("mascot mode unavailable; using abstract backdrops.")
        except Exception as e:
            log.exception("mascot facts render failed")
            _p(f"⚠️ mascot mode failed ({e}); falling back to abstract backdrops.")

    import json as _json
    gpu_utils.ensure_vram_free()
    NARRATION_DIR.mkdir(parents=True, exist_ok=True)
    spans = None
    if narration_path and Path(narration_path).exists():
        narration_path = Path(narration_path)
        spans = _load_spans(narration_path)
        _p("🎙️ using pre-rendered narration" + (" (+spans)" if spans else ""))
    else:
        narration_path = NARRATION_DIR / f"facts_{facts_id}.wav"
        _p(f"🎙️ voicing {len(beats)} beats (kokoro:{rs.get_facts_voice()})...")
        narration_path, spans = _voice_facts(
            [b.get("narration", "") for b in beats], narration_path)
        if spans:
            _spans_path(narration_path).write_text(_json.dumps(spans), encoding="utf-8")
    total_dur = fasm._probe_duration(narration_path)
    _p(f"🎙️ narration ready: {total_dur:.1f}s")

    # Per-beat windows from the real voice spans (exact) — text flips on the voice.
    if spans and len(spans) == len(beats):
        durations = _scene_durations(spans, total_dur)
        _p(f"🎯 {len(durations)} windows from real per-beat spans")
    else:
        wsum = sum(max(1, len(b.get("narration", "").split())) for b in beats) or len(beats)
        durations = [round(max(1, len(b.get("narration", "").split())) / wsum * total_dur, 3)
                     for b in beats]

    # ---- backdrops: active backend (loose, no refs) or gradient fallback ----
    w, h = ASPECTS.get(aspect, ASPECTS["9x16"])
    out_dir = STILLS_DIR / f"facts_{facts_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ar = aspect.replace("x", ":")

    # Facts backdrops follow the conceptual/comic prompts (a bee in a chef hat),
    # so prefer Qwen-Image-Edit — it obeys the concept where Z-Image renders a
    # generic pretty photo. Fall through: Qwen -> Z-Image Turbo -> active -> gradient.
    from modules import gpu_memory
    _GPU_LABEL = {"comfyui_qwen_edit": gpu_memory.QWEN_EDIT,
                  "comfyui_zimage_turbo": gpu_memory.ZIMAGE}
    backend = None
    backend_label = None
    for _bid in ("comfyui_qwen_edit", "comfyui_zimage_turbo", None):
        try:
            b = (image_backend.get_named_backend(_bid) if _bid
                 else image_backend.get_active_backend())
            ok, _msg = b.health_check()
            if ok:
                backend = b
                backend_label = _GPU_LABEL.get(_bid, gpu_memory.FLUX_USO)
                _p(f"backdrops via {_bid or 'active backend'}")
                break
            _p(f"{_bid or 'active'} unhealthy ({_msg}); trying next…")
        except Exception as e:
            _p(f"{_bid or 'active'} unavailable ({e}); trying next…")
    if backend is None:
        _p("no image backend healthy; using gradient backdrops.")

    # Clear the card BEFORE loading the image model. Story-gen leaves Ollama
    # resident (~12.6 GB); Qwen is ~13.5 GB and the two collide on a 16 GB card,
    # which crashed the CUDA context in testing. acquire() evicts Ollama + frees
    # whatever ComfyUI held (Wan from a prior reel).
    if backend is not None:
        gpu_memory.acquire(backend_label)

    _p(f"🖼️ building {len(beats)} backdrops ({'backend' if backend else 'gradient'})...")
    backgrounds: list[Path] = []
    _reused = 0
    for i, b in enumerate(beats):
        dst = out_dir / f"bg_{i:04d}.png"
        # Reuse an already-rendered backdrop for this reel (same facts_id) — skips
        # re-paying image gen and sidesteps occasional USO hangs when iterating on
        # the video stage. Delete the bg_*.png files to force a fresh render.
        if dst.exists() and dst.stat().st_size > 0:
            backgrounds.append(dst); _reused += 1; continue
        if backend is not None:
            try:
                p = backend.generate(
                    prompt=b.get("image_prompt", "abstract cinematic background, no text"),
                    negative_prompt="text, words, letters, captions, subtitles, watermark, logo, "
                                    "blurry, low quality, distorted",
                    aspect_ratio=ar,
                    output_filename=f"facts_{facts_id}/bg_{i:04d}.png",
                )
                backgrounds.append(Path(p)); continue
            except Exception as e:
                # A backend hang/timeout usually means ComfyUI is stuck — don't keep
                # retrying it for every remaining beat (7×300s); drop to gradients.
                _p(f"backdrop {i+1} backend failed ({e}); using gradients from here.")
                backend = None
        backgrounds.append(_gradient_bg(i, w, h, dst))

    if _reused:
        _p(f"♻️ reused {_reused} existing backdrop(s)")

    # Backdrops done — hand the card back so Wan (also ~13.6 GB) starts clean.
    if backend is not None:
        gpu_memory.release(backend_label)

    # ---- assemble: animate each cut (Wan I2V) OR Ken Burns stills ----
    want_wan = (rs.get_facts_video_mode() == "wan") if animate is None else animate
    out = None
    if want_wan:
        _p("🎞️ animating each cut (Wan I2V)…")
        try:
            from modules import horror_video
            # facts beats carry no motion prompt — give each a gentle, upbeat move.
            for b in beats:
                b["motion_prompt"] = (b.get("motion_prompt")
                                      or "subtle cinematic motion, gentle slow push-in, "
                                         "the scene softly comes alive, no text")
            clips = horror_video.render_shot_clips(
                story, backgrounds, durations, aspect_ratio=ar, progress_cb=progress_cb,
                fill_mode="retime")
            gpu_utils.free_comfyui_vram()
            out = fasm.assemble_facts_clips(story, narration_path, clips, aspect=aspect,
                                            music_path=music_path, progress_cb=progress_cb)
        except Exception as e:
            _p(f"⚠️ Wan animation failed ({e}); falling back to Ken Burns stills.")
            out = None

    if out is None:
        _p("🎬 assembling facts reel (Ken Burns)…")
        gpu_utils.free_comfyui_vram()
        out = fasm.assemble_facts(story, narration_path, durations, backgrounds,
                                  aspect=aspect, music_path=music_path,
                                  progress_cb=progress_cb)
    out["narration_audio"] = narration_path
    # Save an upload-ready description (title + facts + hashtags) next to the reel.
    desc = (story.get("description") or "").strip()
    if desc:
        try:
            from modules.assembly import FINAL_DIR
            dfile = FINAL_DIR / f"facts_{facts_id}_description.txt"
            dfile.write_text(desc, encoding="utf-8")
            out["description"] = desc
            out["description_file"] = str(dfile)
            _p(f"📝 description saved: {dfile.name}")
        except Exception as e:
            log.warning(f"description save failed: {e}")

    # Upload kit: a pasteable title + a thumbnail you can set. Built from the
    # second backdrop (the first is the hook card) because the RENDERED reel has
    # read-along captions burned in. Never allowed to fail the render.
    _p("🖼️ building title + thumbnail…")
    out.update(_attach_publish_kit(story, out, backgrounds))
    _p("✅ facts reel complete")
    return out


def _attach_publish_kit(story: dict, out: dict, backgrounds: list) -> dict:
    """Title + thumbnails beside the finished reel. Best-effort, never raises."""
    try:
        from modules import publish_kit
        video = out.get("9x16") or out.get("16x9")
        if not video:
            return {}
        still = None
        if backgrounds:
            still = Path(backgrounds[1] if len(backgrounds) > 1 else backgrounds[0])
        context = "\n".join(
            b.get("narration", "") for b in story.get("beats", [])
        )[:1500]
        from modules import runtime_settings as rs
        kit = publish_kit.attach(
            Path(video),
            fallback_title=story.get("title", "Facts"),
            context=context,
            description=(story.get("description") or "").strip(),
            mode="facts short (true facts, fast cuts)",
            source_image=still,
            thumbnail=rs.get_facts_thumbnail_enabled(),
        )
        return {"publish": kit, "title": kit.get("title"),
                "thumbnail": kit.get("thumb_9x16") or kit.get("thumb_16x9")}
    except Exception as e:
        log.warning(f"publish kit skipped: {e}")
        return {}
