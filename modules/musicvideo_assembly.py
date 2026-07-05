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
from modules.subtitles import (
    lyric_chunks, lyric_lines, windows_to_events, write_captions_ass,
    distribute_lines_over_windows, align_lines_to_words,
)
from modules import lyric_aligner

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


WATERMARK_TEXT = "Rexjaw"

# Each lyric line needs at least this long on screen to be readable. If the
# detected singing windows can't give every line this much (e.g. whisper barely
# heard the vocals on a heavily-produced electronic track and returned a 0.3s
# window for a 17-line song), the detection is unreliable — fall back to the
# proportional spread rather than cramming every line into a fraction of a
# second.
MIN_SEC_PER_LINE = 0.7


def _lyric_caption_events(song_dur: float, lyrics: str, song_audio: Optional[Path]) -> list:
    """Lyric caption timing, best method first (ONE ASR pass, reused):

      1. WORD ALIGNMENT — time each lyric line to the actual sung word
         timestamps (subtitles.align_lines_to_words). True per-line sync.
      2. VOCAL WINDOWS — if word coverage is poor, place lines inside detected
         singing windows (instrumental gaps stay empty).
      3. PROPORTIONAL — if singing is too sparse/undetected (e.g. heavily
         produced electronic vocals ASR can't parse), spread lines across the
         whole song so captions stay readable.

    ACE-Step has instrumental intros/outros/breaks and doesn't sing the lyrics
    1:1, which is why (1) beats a blind spread. Call ONCE per song (not per
    aspect) — transcription is real CPU work."""
    lines = lyric_lines(lyrics)
    if not lines:
        return []
    words = lyric_aligner.get_word_timings(song_audio) if song_audio else None
    if words:
        # 1) real per-line sync from the sung word times
        aligned = align_lines_to_words(lines, words)
        if aligned:
            log.info(f"Lyric captions: word-aligned {len(aligned)}/{len(lines)} lines.")
            return aligned
        # 2) place lines inside detected singing windows (if enough singing)
        windows = lyric_aligner.group_windows(words)
        total_sing = sum(b - a for a, b in windows)
        if windows and total_sing >= MIN_SEC_PER_LINE * len(lines):
            log.info(f"Lyric captions: distributed over {len(windows)} vocal windows.")
            return distribute_lines_over_windows(lines, windows)
        log.warning(f"Detected singing ({total_sing:.1f}s) too sparse for "
                    f"{len(lines)} lines; using proportional caption spread.")
    return windows_to_events([(0.0, song_dur, lyrics)], chunk_fn=lyric_chunks)


def _write_overlay_ass(song_dur: float, w: int, h: int, path: Path,
                       events: Optional[list] = None) -> Path:
    """Write an .ass file with the 'Rexjaw' watermark AND the given (already
    computed — see _lyric_caption_events) lyric caption events, if any."""
    return write_captions_ass(song_dur, w, h, path, events=events, watermark_text=WATERMARK_TEXT)


def _mux_song(video_path: Path, song_path: Path, song_dur: float,
              out_path: Path, subs_path: Path) -> Path:
    """Mux the song over the visuals + burn the overlay .ass (lyric captions +
    'Rexjaw' watermark). Clone the last video frame as a tail pad, then trim to
    the song length. Song audio mapped in full (no -shortest).

    ffmpeg runs with cwd=TEMP_DIR so the subtitles filter can reference the .ass
    by relative name — dodges Windows drive-letter ':' escaping in the filter."""
    chain = [
        f"tpad=stop_mode=clone:stop_duration={TAIL_PAD_SEC}",
        "format=yuv420p",
        f"subtitles={subs_path.name}",
    ]
    vf = ",".join(chain)
    # Inputs/outputs as ABSOLUTE paths — cwd is TEMP_DIR (for the .ass ref), so
    # relative file paths would resolve against the wrong directory.
    cmd = [
        str(FFMPEG_EXE), "-y", "-loglevel", "error",
        "-i", str(Path(video_path).resolve()),
        "-i", str(Path(song_path).resolve()),
        "-filter_complex", f"[0:v]{vf}[v]",
        "-map", "[v]", "-map", "1:a",
        "-t", f"{song_dur:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        str(Path(out_path).resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                            cwd=str(TEMP_DIR))
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

    # Stretch the scene timeline to the ACTUAL song length so visuals cover the
    # whole song (ACE-Step may not hit the requested duration exactly). Without
    # this the visuals run to the requested duration and the video gets trimmed
    # short of — or padded past — the real song.
    base = [max(0.1, float(sc.get("seconds", 8.0))) for sc in scenes]
    total = sum(base) or 1.0
    scene_secs = [b / total * song_dur for b in base]
    log.info(f"Scene timeline scaled to song: sum={sum(scene_secs):.1f}s "
             f"(was {total:.1f}s)")

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    outputs: dict = {"song_id": song_id, "scene_count": len(scenes), "song_dur": song_dur}

    if progress_cb:
        progress_cb("aligning lyrics to song audio...")
    caption_events = _lyric_caption_events(song_dur, song.get("lyrics", ""), song_audio)

    for aspect_key in ("9x16", "16x9", "1x1"):
        w, h = ASPECTS[aspect_key]
        if progress_cb:
            progress_cb(f"rendering {aspect_key}...")
        log.info(f"Building {aspect_key} ({w}x{h})")

        segments = []
        for i, img in enumerate(scene_images):
            seg = TEMP_DIR / f"seg_{song_id}_{aspect_key}_{i:03d}.mp4"
            _ken_burns_segment(
                img, scene_secs[i], w, h, seg, zoom_in=(i % 2 == 0)
            )
            segments.append(seg)

        concat_path = TEMP_DIR / f"concat_{song_id}_{aspect_key}.mp4"
        _concat_segments(segments, concat_path, f"{song_id}_{aspect_key}")

        # Per-aspect overlay .ass (watermark + lyric captions; PlayRes matches dims).
        subs_path = TEMP_DIR / f"overlay_{song_id}_{aspect_key}.ass"
        _write_overlay_ass(song_dur, w, h, subs_path, events=caption_events)

        out_path = FINAL_DIR / f"song_{song_id}_{aspect_key}.mp4"
        _mux_song(concat_path, song_audio, song_dur, out_path, subs_path)
        outputs[aspect_key] = out_path
        log.info(f"✅ {aspect_key} ready: {out_path.name}")

    return outputs
