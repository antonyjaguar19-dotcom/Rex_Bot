"""
Claw Bot — Music Video Assembly (Ken Burns)

Takes a song's scene stills + the rendered song audio and builds the final
music videos (9x16 / 16x9 / 1x1). Each still gets a slow ffmpeg `zoompan`
pan/zoom (Ken Burns), segments are concatenated, then the SONG is muxed over
the visuals as the authoritative audio track.

Core Rule 5 honored: the song is the source of truth for length. Visuals are
padded (last frame cloned) if short and trimmed to the song length — the audio
itself is never cropped (no `-shortest`).

Reuses assembly.py conventions (ffmpeg paths, ASPECTS, duration probe).
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules.assembly import FFMPEG_EXE, FFPROBE_EXE, ASPECTS, FINAL_DIR, _probe_duration

log = logging.getLogger("claw_bot.musicvideo_assembly")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SONGS_DIR = PROJECT_ROOT / "04_Outputs" / "songs"
TEMP_DIR = PROJECT_ROOT / "04_Outputs" / "clips" / "_mv_temp"

FPS = 30                 # smooth Ken Burns motion
TAIL_PAD_SEC = 3.0       # clone last frame this long before trimming to song len


def _ken_burns_segment(image_path: Path, seconds: float, w: int, h: int,
                       out_path: Path, zoom_in: bool) -> Path:
    """Render ONE still into a panning/zooming video segment at w x h.

    zoompan reads a SINGLE input frame and emits exactly `frames` output frames
    (d=frames). Do NOT also -loop/-t the input — that multiplies the work by the
    looped input-frame count and makes the render crawl.
    """
    frames = max(1, round(seconds * FPS))

    # Mild zoom (≤1.12) so the inward crop stays sharp without supersampling.
    if zoom_in:
        zexpr = f"min(zoom+{0.12/frames:.6f},1.12)"
    else:
        zexpr = f"if(eq(on,0),1.12,max(zoom-{0.12/frames:.6f},1.0))"
    xexpr = "iw/2-(iw/zoom/2)"
    yexpr = "ih/2-(ih/zoom/2)"

    # Pre-scale the still to COVER the target, then zoompan zooms in within it.
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},"
        f"zoompan=z='{zexpr}':x='{xexpr}':y='{yexpr}':d={frames}:s={w}x{h}:fps={FPS},"
        f"setsar=1,format=yuv420p"
    )
    cmd = [
        str(FFMPEG_EXE), "-y", "-loglevel", "error",
        "-i", str(image_path),
        "-vf", vf, "-frames:v", str(frames), "-r", str(FPS),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-an",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"Ken Burns segment failed for {image_path.name}:\n{result.stderr.strip()}")
    return out_path


def _concat_segments(segments: list[Path], out_path: Path, tag: str) -> Path:
    """Concat same-codec segments via the concat demuxer (stream copy)."""
    list_file = TEMP_DIR / f"_concat_{tag}.txt"
    list_file.write_text(
        "".join(f"file '{p.as_posix()}'\n" for p in segments), encoding="utf-8"
    )
    cmd = [
        str(FFMPEG_EXE), "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c", "copy", str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"Concat failed ({tag}):\n{result.stderr.strip()}")
    return out_path


def _mux_song(video_path: Path, song_path: Path, song_dur: float,
              out_path: Path) -> Path:
    """Mux the song over the visuals. Clone the last video frame as a tail pad,
    then trim the whole thing to the song length. Song audio mapped in full."""
    vf = f"tpad=stop_mode=clone:stop_duration={TAIL_PAD_SEC},format=yuv420p"
    cmd = [
        str(FFMPEG_EXE), "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-i", str(song_path),
        "-filter_complex", f"[0:v]{vf}[v]",
        "-map", "[v]", "-map", "1:a",
        "-t", f"{song_dur:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"Song mux failed:\n{result.stderr.strip()}")
    return out_path


def assemble_musicvideo(
    song: dict,
    song_audio: Path,
    scene_images: list[Path],
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Build 9x16/16x9/1x1 music videos.

    song          — the song dict (scenes carry `seconds`).
    song_audio    — path to the rendered ACE-Step song.
    scene_images  — one rendered still per scene, same order as song["scenes"].
    """
    if not FFMPEG_EXE.exists():
        raise FileNotFoundError(f"ffmpeg not found at {FFMPEG_EXE}")
    scenes = song.get("scenes", [])
    if not scene_images or len(scene_images) != len(scenes):
        raise ValueError(
            f"scene_images ({len(scene_images)}) must match scenes ({len(scenes)})."
        )

    song_id = song.get("song_id") or song.get("_id")
    song_dur = _probe_duration(song_audio)
    log.info(f"Assembling music video {song_id}: {len(scenes)} scenes, song={song_dur:.1f}s")

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    outputs: dict = {"song_id": song_id, "scene_count": len(scenes), "song_dur": song_dur}

    for aspect_key in ("9x16", "16x9", "1x1"):
        w, h = ASPECTS[aspect_key]
        if progress_cb:
            progress_cb(f"rendering {aspect_key}...")
        log.info(f"Building {aspect_key} ({w}x{h})")

        segments = []
        for i, (sc, img) in enumerate(zip(scenes, scene_images)):
            seg = TEMP_DIR / f"seg_{song_id}_{aspect_key}_{i:03d}.mp4"
            _ken_burns_segment(
                img, float(sc.get("seconds", 8.0)), w, h, seg, zoom_in=(i % 2 == 0)
            )
            segments.append(seg)

        concat_path = TEMP_DIR / f"concat_{song_id}_{aspect_key}.mp4"
        _concat_segments(segments, concat_path, f"{song_id}_{aspect_key}")

        out_path = FINAL_DIR / f"song_{song_id}_{aspect_key}.mp4"
        _mux_song(concat_path, song_audio, song_dur, out_path)
        outputs[aspect_key] = out_path
        log.info(f"✅ {aspect_key} ready: {out_path.name}")

    return outputs
