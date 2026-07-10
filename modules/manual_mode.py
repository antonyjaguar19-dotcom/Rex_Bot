"""
Claw Bot — Manual Mode

Direct-drive creative bench: you talk straight to ComfyUI through the bot.
No LLM story chain — YOU write the prompt, pick the model, set the params,
and build the storyboard shot by shot (generated images or your own uploads),
animate shots with the active video backend, add narration/music, assemble.

Think of it as the Nuke node graph instead of the automated dailies machine:
same render farm (ComfyUI + backends + assembly), you drive every knob.

State = one JSON manifest per project:
    04_Outputs/manual/{project_id}/project.json
    04_Outputs/manual/{project_id}/images/   stills (generated or uploaded)
    04_Outputs/manual/{project_id}/clips/    animated shots
    04_Outputs/manual/{project_id}/audio/    narration wavs + music
    04_Outputs/manual/{project_id}/final/    assembled MP4s

Shared between Discord + dashboard the same way every other pipeline is:
both read/write the manifest on disk (atomic writes).

Duration rules honor Core Rule 5: narration is the source of truth — a shot's
assembled length is max(shot.duration, narration_duration + pad). Audio is
never trimmed; video freeze-pads.
"""

import json
import logging
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules.file_utils import atomic_write_json
from modules import model_registry
from modules import gpu_utils

log = logging.getLogger("claw_bot.manual_mode")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
MANUAL_DIR = PROJECT_ROOT / "04_Outputs" / "manual"
CURRENT_MARKER = MANUAL_DIR / "_current.txt"

SEG_FPS = 30                 # uniform segment fps for assembly (kenburns-smooth)
NARR_PAD_SEC = 0.25          # breath after narration before the cut
DEFAULT_SHOT_SECONDS = 5.0
CROSSFADE_NONE = "cut"       # manual mode assembles hard cuts (simple + predictable)

MUSIC_VOLUME = 0.30          # music bed level under narration


# ==============================================================================
# PROJECT STORE
# ==============================================================================

def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def project_dir(pid: str) -> Path:
    return MANUAL_DIR / pid


def _manifest_path(pid: str) -> Path:
    return project_dir(pid) / "project.json"


def create_project(name: str = "", aspect_ratio: str = "16:9") -> dict:
    pid = _now_id()
    pdir = project_dir(pid)
    for sub in ("images", "clips", "audio", "final"):
        (pdir / sub).mkdir(parents=True, exist_ok=True)
    proj = {
        "_id": pid,
        "name": name or f"Manual {pid}",
        "created": datetime.now().isoformat(timespec="seconds"),
        "aspect_ratio": aspect_ratio,
        "shots": [],          # ordered; index order IS board order
        "music": None,        # {"tags":..., "lyrics":..., "path": "audio/..."}
        "final": {},          # {"16:9": "final/....mp4", ...}
    }
    save_project(proj)
    set_current(pid)
    log.info(f"Manual project created: {pid} ({proj['name']})")
    return proj


def load_project(pid: str) -> Optional[dict]:
    p = _manifest_path(pid)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_project(proj: dict) -> Path:
    p = _manifest_path(proj["_id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(p, proj)
    return p


def list_projects() -> list[tuple[str, str]]:
    """[(pid, name)] newest first."""
    if not MANUAL_DIR.exists():
        return []
    out = []
    for d in sorted(MANUAL_DIR.iterdir(), reverse=True):
        m = d / "project.json"
        if not m.exists():
            continue
        try:
            j = json.loads(m.read_text(encoding="utf-8"))
            out.append((j.get("_id", d.name), j.get("name", d.name)))
        except Exception:
            continue
    return out


def project_stats(pid: str) -> dict:
    """What a delete would destroy: file count, MB, shot count. For confirm UIs."""
    pdir = project_dir(pid)
    files = [f for f in pdir.rglob("*") if f.is_file()] if pdir.exists() else []
    proj = load_project(pid)
    return {
        "files": len(files),
        "mb": round(sum(f.stat().st_size for f in files) / (1024 * 1024), 1),
        "shots": len(proj["shots"]) if proj else 0,
        "name": proj.get("name", pid) if proj else pid,
        "finals": len(proj.get("final") or {}) if proj else 0,
    }


def delete_project(pid: str) -> dict:
    """Permanently delete a manual project and every artifact under it
    (stills, clips, audio, finals). Irreversible — callers MUST confirm first.

    Guards: the id must name a real project (project.json present) and the
    resolved directory must live inside MANUAL_DIR — so a crafted id like
    '../../04_Outputs' can never escape the sandbox.

    Returns the pre-delete stats. Clears the current marker if it pointed here.
    """
    if not pid or not str(pid).strip():
        raise ValueError("Project id required.")
    pdir = project_dir(pid)

    # Containment: resolve and verify the target is a child of MANUAL_DIR.
    root = MANUAL_DIR.resolve()
    try:
        target = pdir.resolve()
    except OSError:
        raise ValueError(f"Bad project id: {pid!r}")
    if target == root or root not in target.parents:
        raise ValueError(f"Refusing to delete outside {root}: {target}")
    if not _manifest_path(pid).exists():
        raise FileNotFoundError(f"No manual project '{pid}' (project.json missing).")

    stats = project_stats(pid)
    shutil.rmtree(target)
    log.warning(f"Manual project DELETED: {pid} "
                f"({stats['files']} files, {stats['mb']} MB, {stats['shots']} shots)")

    # Repoint the marker: the deleted project must not stay 'current'.
    if CURRENT_MARKER.exists() and CURRENT_MARKER.read_text(encoding="utf-8").strip() == pid:
        CURRENT_MARKER.unlink(missing_ok=True)
        remaining = list_projects()
        if remaining:
            set_current(remaining[0][0])
    return stats


def set_current(pid: str) -> None:
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    CURRENT_MARKER.write_text(pid, encoding="utf-8")


def current_project_id() -> Optional[str]:
    """Marker first; falls back to newest project on disk."""
    if CURRENT_MARKER.exists():
        pid = CURRENT_MARKER.read_text(encoding="utf-8").strip()
        if pid and _manifest_path(pid).exists():
            return pid
    projs = list_projects()
    return projs[0][0] if projs else None


def abs_path(proj: dict, rel: Optional[str]) -> Optional[Path]:
    if not rel:
        return None
    return project_dir(proj["_id"]) / rel


# ==============================================================================
# SHOT BOARD OPS
# ==============================================================================

def _renumber(proj: dict) -> None:
    for i, s in enumerate(proj["shots"], start=1):
        s["n"] = i


def add_shot(proj: dict, image_path: Path, prompt: str = "",
             seed: Optional[int] = None,
             duration: float = DEFAULT_SHOT_SECONDS,
             position: Optional[int] = None) -> dict:
    """Copy an image (generated or uploaded) into the project and append a shot.
    position: 1-based insert point; None = append."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    pdir = project_dir(proj["_id"])
    dest_name = f"shot_{int(time.time() * 1000)}{image_path.suffix.lower() or '.png'}"
    dest = pdir / "images" / dest_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if image_path.resolve() != dest.resolve():
        shutil.copy2(image_path, dest)
    shot = {
        "n": 0,
        "image": f"images/{dest_name}",
        "prompt": prompt or "",
        "seed": seed,
        "motion_prompt": "",
        "clip": None,
        "duration": float(duration),
        "narration": "",
        "narration_audio": None,
    }
    if position is not None and 1 <= position <= len(proj["shots"]):
        proj["shots"].insert(position - 1, shot)
    else:
        proj["shots"].append(shot)
    _renumber(proj)
    save_project(proj)
    return shot


def get_shot(proj: dict, n: int) -> dict:
    for s in proj["shots"]:
        if s["n"] == n:
            return s
    raise IndexError(f"No shot {n} (board has {len(proj['shots'])})")


def remove_shot(proj: dict, n: int) -> None:
    shot = get_shot(proj, n)
    proj["shots"].remove(shot)
    _renumber(proj)
    save_project(proj)


def move_shot(proj: dict, n: int, direction: str) -> None:
    """direction: 'up' (earlier) or 'down' (later)."""
    idx = n - 1
    if idx < 0 or idx >= len(proj["shots"]):
        raise IndexError(f"No shot {n}")
    j = idx - 1 if direction == "up" else idx + 1
    if j < 0 or j >= len(proj["shots"]):
        return
    shots = proj["shots"]
    shots[idx], shots[j] = shots[j], shots[idx]
    _renumber(proj)
    save_project(proj)


def set_shot_fields(proj: dict, n: int, **fields) -> dict:
    """Update editable fields: motion_prompt, duration, narration, prompt."""
    shot = get_shot(proj, n)
    allowed = {"motion_prompt", "duration", "narration", "prompt"}
    for k, v in fields.items():
        if k not in allowed:
            raise KeyError(f"Field '{k}' not editable")
        shot[k] = float(v) if k == "duration" else v
    save_project(proj)
    return shot


def replace_shot_image(proj: dict, n: int, image_path: Path,
                       prompt: str = None, seed: Optional[int] = None) -> dict:
    """Swap a shot's still for a new one; clears its stale clip."""
    shot = get_shot(proj, n)
    image_path = Path(image_path)
    pdir = project_dir(proj["_id"])
    dest_name = f"shot_{int(time.time() * 1000)}{image_path.suffix.lower() or '.png'}"
    dest = pdir / "images" / dest_name
    shutil.copy2(image_path, dest)
    shot["image"] = f"images/{dest_name}"
    if prompt is not None:
        shot["prompt"] = prompt
    if seed is not None:
        shot["seed"] = seed
    shot["clip"] = None      # old animation no longer matches the new still
    save_project(proj)
    return shot


# ==============================================================================
# GENERATION — image / video / narration / music
# ==============================================================================

def check_prompt_safety(prompt: str) -> tuple[bool, list]:
    """Adult-profile hard-block check on a freeform prompt (owner tool, so only
    the never-allowed terms block; horror-style prompts pass)."""
    try:
        from modules import safety_filter
        fake_script = {"title": "", "shots": [{"narration": prompt,
                                               "first_frame_prompt": prompt}]}
        ok, blocked, _warn = safety_filter.check_safety(fake_script, profile="adult")
        return ok, blocked
    except Exception as e:
        log.warning(f"Safety check unavailable ({e}) — allowing prompt")
        return True, []


def generate_image(prompt: str,
                   backend_id: Optional[str] = None,
                   negative_prompt: Optional[str] = None,
                   aspect_ratio: Optional[str] = None,
                   seed: Optional[int] = None,
                   reference_image: Optional[Path] = None,
                   proj: Optional[dict] = None) -> dict:
    """Freeform still generation. Steps/cfg come from runtime_settings overrides
    (same knobs the Settings card writes) — adapters already read those.
    Returns {"path": Path, "seed": int, "backend": id}. Copies the result into
    the project's images/ (as a loose gen, NOT yet a shot) when proj given."""
    ok, blocked = check_prompt_safety(prompt)
    if not ok:
        raise ValueError(f"Prompt blocked by safety filter: {', '.join(blocked[:5])}")

    from modules import image_backend as ib
    backend = ib.get_named_backend(backend_id) if backend_id else ib.get_active_backend()

    gpu_utils.ensure_vram_free(min_gb=6.0)

    if seed is None or seed < 0:
        seed = random.randint(1, 2**31 - 1)
    aspect = aspect_ratio or "16:9"

    kwargs = dict(negative_prompt=negative_prompt, aspect_ratio=aspect, seed=seed)
    if reference_image:
        kwargs["reference_image"] = Path(reference_image)
    try:
        result_path = backend.generate(prompt, **kwargs)
    except TypeError:
        # Adapter doesn't take reference_image — retry without it.
        kwargs.pop("reference_image", None)
        result_path = backend.generate(prompt, **kwargs)

    result_path = Path(result_path)
    used_seed = int(getattr(backend, "_last_seed", seed) or seed)

    if proj is not None:
        pdir = project_dir(proj["_id"])
        dest = pdir / "images" / f"gen_{int(time.time() * 1000)}{result_path.suffix or '.png'}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result_path, dest)
        result_path = dest

    return {"path": result_path, "seed": used_seed, "backend": backend.backend_id}


def animate_shot(proj: dict, n: int, motion_prompt: Optional[str] = None,
                 seed: Optional[int] = None) -> Path:
    """Animate one shot's still via the ACTIVE video backend (I2V). Duration
    capped at the model's max_clip_seconds (no chaining in manual mode —
    keep it one honest take; raise the shot count instead)."""
    shot = get_shot(proj, n)
    img = abs_path(proj, shot["image"])
    if not img or not img.exists():
        raise FileNotFoundError(f"Shot {n} still missing: {shot['image']}")
    motion = (motion_prompt if motion_prompt is not None else shot.get("motion_prompt")) or ""
    if not motion.strip():
        raise ValueError(f"Shot {n} has no motion prompt — set one first.")

    ok, blocked = check_prompt_safety(motion)
    if not ok:
        raise ValueError(f"Motion prompt blocked: {', '.join(blocked[:5])}")

    from modules import video_backend as vb
    backend = vb.get_active_backend()
    cfg = backend.config
    fps = int(cfg.get("default_fps", 16))
    max_sec = float(cfg.get("max_clip_seconds", 5.0))
    want_sec = min(float(shot.get("duration", DEFAULT_SHOT_SECONDS)), max_sec)
    frame_count = max(1, int(round(want_sec * fps)))

    gpu_utils.ensure_vram_free(min_gb=8.0)
    if seed is None:
        seed = random.randint(1, 2**31 - 1)

    result = backend.generate(
        prompt=motion,
        input_image=img,
        frame_count=frame_count,
        fps=fps,
        aspect_ratio=proj.get("aspect_ratio", "16:9"),
        seed=seed,
    )
    result = Path(result)

    pdir = project_dir(proj["_id"])
    dest = pdir / "clips" / f"shot{n}_{int(time.time())}.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result, dest)

    shot["motion_prompt"] = motion
    shot["clip"] = f"clips/{dest.name}"
    save_project(proj)
    return dest


_TTS_ENGINE = None


def narrate_shot(proj: dict, n: int, text: Optional[str] = None,
                 voice: Optional[str] = None) -> Path:
    """Kokoro TTS narration for one shot. Narration length wins over shot
    duration at assembly (Core Rule 5)."""
    global _TTS_ENGINE
    shot = get_shot(proj, n)
    line = (text if text is not None else shot.get("narration")) or ""
    if not line.strip():
        raise ValueError(f"Shot {n} has no narration text.")

    from modules.tts_engine import TTSEngine
    if _TTS_ENGINE is None:
        _TTS_ENGINE = TTSEngine()

    pdir = project_dir(proj["_id"])
    out = pdir / "audio" / f"narr_shot{n}_{int(time.time())}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    _TTS_ENGINE.synthesize(line, output_path=out, voice=voice)

    shot["narration"] = line
    shot["narration_audio"] = f"audio/{out.name}"
    save_project(proj)
    return out


def generate_music(proj: dict, tags: str, lyrics: str = "",
                   duration_sec: Optional[float] = None) -> Path:
    """Music bed via the active audio backend (ACE-Step). Default duration =
    total board duration + 2s."""
    from modules import audio_backend as ab
    backend = ab.get_active_backend()

    if duration_sec is None:
        duration_sec = max(10.0, total_duration(proj) + 2.0)

    gpu_utils.ensure_vram_free(min_gb=6.0)
    result = backend.generate(tags=tags, lyrics=lyrics or "[inst]",
                              duration_sec=float(duration_sec))
    result = Path(result)

    pdir = project_dir(proj["_id"])
    dest = pdir / "audio" / f"music_{int(time.time())}{result.suffix or '.mp3'}"
    shutil.copy2(result, dest)
    proj["music"] = {"tags": tags, "lyrics": lyrics, "path": f"audio/{dest.name}"}
    save_project(proj)
    return dest


# ==============================================================================
# ASSEMBLY — fixed/narration-driven durations, hard cuts
# ==============================================================================

def _ffmpeg_paths():
    from modules.assembly import FFMPEG_EXE, FFPROBE_EXE
    return FFMPEG_EXE, FFPROBE_EXE


def _probe_dur(path: Path) -> float:
    from modules.assembly import _probe_duration
    return _probe_duration(Path(path))


def shot_target_duration(shot: dict, narr_dur: Optional[float] = None) -> float:
    """Assembled length of a shot: narration wins when longer (Rule 5)."""
    base = float(shot.get("duration", DEFAULT_SHOT_SECONDS) or DEFAULT_SHOT_SECONDS)
    if narr_dur and narr_dur > 0:
        return max(base, narr_dur + NARR_PAD_SEC)
    return base


def total_duration(proj: dict) -> float:
    total = 0.0
    for s in proj["shots"]:
        nd = None
        na = abs_path(proj, s.get("narration_audio"))
        if na and na.exists():
            try:
                nd = _probe_dur(na)
            except Exception:
                nd = None
        total += shot_target_duration(s, nd)
    return total


def _segment_from_clip(clip: Path, narr: Optional[Path], seg_dur: float,
                       w: int, h: int, out: Path) -> Path:
    """Normalize one animated clip to w×h @SEG_FPS, freeze-pad to seg_dur,
    attach narration (silence-padded) or a silent track. Never trims audio:
    seg_dur already >= narration duration."""
    ffmpeg, _ = _ffmpeg_paths()
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps={SEG_FPS},setsar=1,"
          f"tpad=stop_mode=clone:stop_duration={seg_dur + 1.0:.3f}")
    cmd = [str(ffmpeg), "-y", "-loglevel", "error", "-i", str(clip)]
    if narr and narr.exists():
        cmd += ["-i", str(narr)]
        af = "[1:a]aresample=48000,pan=stereo|c0=c0|c1=c0,apad[a]"
    else:
        cmd += ["-f", "lavfi", "-t", f"{seg_dur:.3f}",
                "-i", "anullsrc=r=48000:cl=stereo"]
        af = "[1:a]anull[a]"
    cmd += [
        "-filter_complex", f"[0:v]{vf}[v];{af}",
        "-map", "[v]", "-map", "[a]",
        "-t", f"{seg_dur:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", str(SEG_FPS),
        "-c:a", "aac", "-ar", "48000", "-ac", "2",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"Segment encode failed ({clip.name}):\n{r.stderr.strip()}")
    return out


def _segment_from_still(img: Path, narr: Optional[Path], seg_dur: float,
                        w: int, h: int, out: Path, zoom_in: bool) -> Path:
    """Ken Burns a still to seg_dur, then attach audio like the clip path."""
    from modules.musicvideo_assembly import _ken_burns_segment
    tmp = out.with_suffix(".kb.mp4")
    _ken_burns_segment(img, seg_dur, w, h, tmp, zoom_in)
    try:
        return _segment_from_clip(tmp, narr, seg_dur, w, h, out)
    finally:
        tmp.unlink(missing_ok=True)


def assemble(proj: dict, aspects: Optional[list] = None,
             progress_cb=None) -> dict:
    """Build final MP4(s). Per shot: animated clip if present, else Ken Burns
    the still. Hard cuts. Music (if any) mixed under at MUSIC_VOLUME.
    Returns {aspect: Path}."""
    if not proj["shots"]:
        raise ValueError("Board is empty — add at least one shot.")
    aspects = aspects or [proj.get("aspect_ratio", "16:9")]
    pdir = project_dir(proj["_id"])
    temp = pdir / "_temp"
    temp.mkdir(exist_ok=True)
    ffmpeg, _ = _ffmpeg_paths()
    finals = {}

    for aspect in aspects:
        w, h = model_registry.get_resolution(aspect)
        segments = []
        for i, shot in enumerate(proj["shots"]):
            if progress_cb:
                progress_cb(f"[{aspect}] shot {shot['n']}/{len(proj['shots'])}")
            narr = abs_path(proj, shot.get("narration_audio"))
            narr = narr if (narr and narr.exists()) else None
            nd = _probe_dur(narr) if narr else None
            seg_dur = shot_target_duration(shot, nd)
            seg = temp / f"seg_{aspect.replace(':', 'x')}_{shot['n']}.mp4"
            clip = abs_path(proj, shot.get("clip"))
            if clip and clip.exists():
                _segment_from_clip(clip, narr, seg_dur, w, h, seg)
            else:
                img = abs_path(proj, shot.get("image"))
                if not img or not img.exists():
                    raise FileNotFoundError(f"Shot {shot['n']} has no image on disk.")
                _segment_from_still(img, narr, seg_dur, w, h, seg,
                                    zoom_in=(i % 2 == 0))
            segments.append(seg)

        # Concat (identical codec params → stream copy).
        list_file = temp / f"concat_{aspect.replace(':', 'x')}.txt"
        list_file.write_text(
            "".join(f"file '{p.as_posix()}'\n" for p in segments), encoding="utf-8")
        concat_out = temp / f"assembled_{aspect.replace(':', 'x')}.mp4"
        r = subprocess.run(
            [str(ffmpeg), "-y", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", str(list_file),
             "-c", "copy", str(concat_out)],
            capture_output=True, text=True, timeout=600)
        if r.returncode != 0 or not concat_out.exists():
            raise RuntimeError(f"Concat failed ({aspect}):\n{r.stderr.strip()}")

        # Music bed (optional).
        final_name = f"manual_{proj['_id']}_{aspect.replace(':', 'x')}.mp4"
        final_path = pdir / "final" / final_name
        final_path.parent.mkdir(exist_ok=True)
        music = abs_path(proj, (proj.get("music") or {}).get("path"))
        if music and music.exists():
            vid_dur = _probe_dur(concat_out)
            r = subprocess.run(
                [str(ffmpeg), "-y", "-loglevel", "error",
                 "-i", str(concat_out), "-i", str(music),
                 "-filter_complex",
                 f"[1:a]volume={MUSIC_VOLUME},aresample=48000[m];"
                 f"[0:a][m]amix=inputs=2:duration=first:dropout_transition=3[a]",
                 "-map", "0:v", "-map", "[a]",
                 "-t", f"{vid_dur:.3f}",
                 "-c:v", "copy", "-c:a", "aac", "-ar", "48000", "-ac", "2",
                 str(final_path)],
                capture_output=True, text=True, timeout=900)
            if r.returncode != 0 or not final_path.exists():
                raise RuntimeError(f"Music mux failed ({aspect}):\n{r.stderr.strip()}")
        else:
            shutil.copy2(concat_out, final_path)

        proj.setdefault("final", {})[aspect] = f"final/{final_name}"
        finals[aspect] = final_path
        log.info(f"Manual final [{aspect}]: {final_path}")

    save_project(proj)
    return finals


# ==============================================================================
# SUMMARY (Discord board view)
# ==============================================================================

def board_summary(proj: dict) -> str:
    lines = [f"🎬 **{proj['name']}** (`{proj['_id']}`) — "
             f"{proj.get('aspect_ratio', '16:9')}, {len(proj['shots'])} shots, "
             f"~{total_duration(proj):.1f}s"]
    for s in proj["shots"]:
        clip = "🎞️" if s.get("clip") else "🖼️"
        narr = "🎙️" if s.get("narration_audio") else "·"
        motion = (s.get("motion_prompt") or "")[:40]
        lines.append(f"`{s['n']:>2}` {clip}{narr} {s.get('duration', 5):.1f}s — "
                     f"{(s.get('prompt') or 'uploaded')[:50]}"
                     + (f" | 🎥 {motion}" if motion else ""))
    m = proj.get("music")
    if m:
        lines.append(f"🎵 music: {m.get('tags', '')[:60]}")
    return "\n".join(lines)
