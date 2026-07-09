"""
Claw Bot — Horror Video (animated Wan I2V per shot)

Turns each horror beat's still into a moving Wan 2.2 I2V clip (like the kids
video stage), then freeze-pads it to the beat's narration window so the visuals
track the audio. One Wan clip per beat (Wan caps ~5s); the remainder of a longer
beat holds the last frame. Returns per-beat window-length silent clips for the
assembler to concat + mux the narration over.

Full-duration animation of every beat would be 3-5x the Wan calls; this keeps it
to one clip per shot (motion at each cut, frozen dread tail).
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import video_backend, gpu_utils
from modules.assembly import FFMPEG_EXE, _probe_duration
from modules.musicvideo_assembly import TEMP_DIR

log = logging.getLogger("claw_bot.horror_video")

DEFAULT_FPS = 16
MAX_WAN_SEC = 5.0   # Wan 2.2 single-clip ceiling


def _freeze_pad(clip: Path, target_dur: float, out: Path) -> Path:
    """Clone the last frame so the clip lasts `target_dur` (silent)."""
    have = _probe_duration(clip)
    pad = max(0.0, target_dur - have)
    if pad < 0.05:
        # already long enough — just trim
        cmd = [str(FFMPEG_EXE), "-y", "-loglevel", "error", "-i", str(clip),
               "-t", f"{target_dur:.3f}", "-c:v", "libx264", "-preset", "veryfast",
               "-pix_fmt", "yuv420p", "-an", str(out)]
    else:
        cmd = [str(FFMPEG_EXE), "-y", "-loglevel", "error", "-i", str(clip),
               "-vf", f"tpad=stop_mode=clone:stop_duration={pad:.3f},format=yuv420p",
               "-t", f"{target_dur:.3f}", "-c:v", "libx264", "-preset", "veryfast",
               "-pix_fmt", "yuv420p", "-an", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"freeze-pad failed:\n{r.stderr.strip()}")
    return out


def _retime_fill(clip: Path, target_dur: float, out: Path, fps: int = DEFAULT_FPS) -> Path:
    """Stretch/squeeze the clip to EXACTLY target_dur by retiming (setpts) — the
    motion plays for the whole window instead of freezing on the last frame. A
    short Wan clip covering a longer window becomes gentle slow-motion (no freeze)."""
    have = max(0.1, _probe_duration(clip))
    factor = max(0.1, target_dur) / have          # >1 slows down, <1 speeds up
    cmd = [str(FFMPEG_EXE), "-y", "-loglevel", "error", "-i", str(clip),
           "-vf", f"setpts={factor:.5f}*PTS,fps={fps},format=yuv420p",
           "-t", f"{target_dur:.3f}", "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"retime-fill failed:\n{r.stderr.strip()}")
    return out


def render_shot_clips(
    story: dict,
    scene_images: list,
    durations: list,
    aspect_ratio: str = "16:9",
    progress_cb: Optional[Callable[[str], None]] = None,
    fill_mode: str = "freeze",
) -> list[Path]:
    """One Wan I2V clip per beat, fit to its narration window. Returns per-beat
    clip paths (same order/length as beats). `fill_mode`: 'freeze' (hold last
    frame for the window remainder — horror) or 'retime' (stretch the motion to
    fill the window, no freeze — facts)."""
    beats = story.get("beats", [])
    n = len(scene_images)
    if not (n == len(durations) == len(beats)):
        raise ValueError(f"len mismatch: imgs={n} durs={len(durations)} beats={len(beats)}")

    horror_id = story.get("horror_id") or story.get("_id")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    be = video_backend.get_active_backend()
    fps = int(getattr(be, "default_fps", DEFAULT_FPS) or DEFAULT_FPS)

    clips = []
    for i, (img, dur, beat) in enumerate(zip(scene_images, durations, beats)):
        if progress_cb:
            progress_cb(f"wan clip {i+1}/{n}...")
        gpu_utils.ensure_vram_free()
        wan_sec = min(float(dur), MAX_WAN_SEC)
        raw = TEMP_DIR / f"hwan_{horror_id}_{i:04d}.mp4"
        be.generate(
            prompt=(beat.get("motion_prompt") or "slow ominous push-in"),
            input_image=Path(img),
            frame_count=int(round(wan_sec * fps)),
            fps=fps,
            aspect_ratio=aspect_ratio,
            output_filename=raw.name,
        )
        # backend writes into its OUTPUT_DIR; locate the produced file
        produced = _locate_backend_output(raw.name)
        seg = TEMP_DIR / f"hseg_{horror_id}_{i:04d}.mp4"
        if fill_mode == "retime":
            _retime_fill(produced, float(dur), seg, fps)
        else:
            _freeze_pad(produced, float(dur), seg)
        clips.append(seg)
    return clips


def _locate_backend_output(filename: str) -> Path:
    """Wan backend saves into 04_Outputs/videos/<filename>."""
    p = _HERE.parent / "04_Outputs" / "videos" / filename
    if p.exists():
        return p
    # fallback: search
    hits = list((_HERE.parent / "04_Outputs" / "videos").glob(filename))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Wan output not found: {filename}")
