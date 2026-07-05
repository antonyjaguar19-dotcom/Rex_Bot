"""
Claw Bot — Horror Story Assembly (16x9 only)

Tiles photoreal stills over the narration with Ken Burns motion, mixes a low
ambient drone under the deep-voice narration, and burns the Rexjaw watermark.
Single aspect: 16x9.

Reuses the Ken Burns / watermark helpers from musicvideo_assembly. Narration is
the source of truth for length (no -shortest); the drone is synthesized in pure
ffmpeg (no model) and mixed quietly under the voice.
"""

import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules.assembly import FFMPEG_EXE, FINAL_DIR, _probe_duration
from modules.musicvideo_assembly import (
    ASPECTS, TEMP_DIR, WATERMARK_TEXT, _ass_time, _ken_burns_segment,
    _concat_segments, _write_overlay_ass,
)

log = logging.getLogger("claw_bot.horror_assembly")

ASPECT_KEY = "16x9"
DRONE_VOLUME = 0.18   # ambient bed level under narration (narration stays front)
CAP_MAX_CHARS = 84    # max chars per on-screen subtitle chunk (sentence-level)


def _sentence_chunks(text: str) -> list[str]:
    """Split a beat's narration into short on-screen cues: sentence boundaries,
    with tiny sentences merged up to ~CAP_MAX_CHARS so captions stay readable."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    chunks, cur = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if cur and len(cur) + 1 + len(p) > CAP_MAX_CHARS:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return chunks or [(text or "").strip()]


def _ass_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\n", " ").replace("{", "(").replace("}", ")")


def _caption_events(beats: list, spans: list) -> list:
    """Turn per-beat (start,end) windows into sentence-level (start,end,text)
    caption cues, distributing each beat's window by chunk length."""
    events = []
    for b, (t0, t1) in zip(beats, spans):
        text = (b.get("narration") or "").strip()
        if not text or t1 <= t0:
            continue
        chunks = _sentence_chunks(text)
        total = sum(len(c) for c in chunks) or 1
        cur = t0
        for c in chunks:
            dur = (t1 - t0) * (len(c) / total)
            events.append((cur, min(cur + dur, t1), c))
            cur += dur
    return events


def _write_captions_ass(total_dur: float, w: int, h: int, path: Path,
                        beats: Optional[list] = None,
                        spans: Optional[list] = None) -> Path:
    """Write an .ass with the Rexjaw watermark AND burned-in narration subtitles
    (bottom-center, white with a black outline + drop shadow). Falls back to a
    watermark-only file when no beats/spans are supplied."""
    wm_size = max(14, round(h * 0.024))
    wm_margin = max(12, round(w * 0.012))
    cap_size = max(22, round(h * 0.045))
    cap_marginv = max(28, round(h * 0.06))
    cap_side = max(40, round(w * 0.08))
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {w}\nPlayResY: {h}\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, "
        "Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV\n"
        f"Style: Mark,Arial,{wm_size},&H80FFFFFF,&H80000000,&H00000000,"
        f"0,0,1,1,1,6,40,{wm_margin},40\n"
        # Captions: opaque white, black outline (2px) + shadow (1px), bottom-center.
        f"Style: Cap,Arial,{cap_size},&H00FFFFFF,&H00000000,&H90000000,"
        f"0,0,1,2,1,2,{cap_side},{cap_side},{cap_marginv}\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        f"Dialogue: 0,{_ass_time(0)},{_ass_time(total_dur)},Mark,,0,0,0,,{WATERMARK_TEXT}\n"
    )
    lines = [header]
    if beats and spans:
        for t0, t1, txt in _caption_events(beats, spans):
            lines.append(
                f"Dialogue: 0,{_ass_time(t0)},{_ass_time(t1)},Cap,,0,0,0,,{_ass_escape(txt)}\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _spans_from_durations(durations: list) -> list:
    """Cumulative (start,end) per beat from per-beat durations."""
    spans, cur = [], 0.0
    for d in durations:
        spans.append((cur, cur + float(d)))
        cur += float(d)
    return spans


def _build_drone(duration: float, out_path: Path) -> Path:
    """Synthesize an eerie ambient drone of `duration` seconds (pure ffmpeg).
    Two detuned low sines + slow tremolo + long echo + lowpass = ominous bed."""
    d = max(1.0, duration)
    cmd = [
        str(FFMPEG_EXE), "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"sine=frequency=55:duration={d:.2f}",
        "-f", "lavfi", "-i", f"sine=frequency=58.5:duration={d:.2f}",
        "-filter_complex",
        "[0][1]amix=inputs=2,"
        "tremolo=f=0.18:d=0.7,"
        "aecho=0.8:0.7:1100:0.4,"
        "lowpass=f=320,"
        f"volume={DRONE_VOLUME}[a]",
        "-map", "[a]", "-c:a", "pcm_s16le",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"Drone synth failed:\n{result.stderr.strip()}")
    return out_path


def _mux_horror(video_path: Path, narration_path: Path, drone_path: Optional[Path],
                total_dur: float, subs_path: Path, out_path: Path) -> Path:
    """Mux narration (+ optional drone) over the visuals, burn watermark .ass.
    cwd=TEMP_DIR so the subtitles filter can use a relative .ass name."""
    inputs = [
        "-i", str(Path(video_path).resolve()),
        "-i", str(Path(narration_path).resolve()),
    ]
    if drone_path is not None:
        inputs += ["-i", str(Path(drone_path).resolve())]
        # narration = input 1, drone = input 2; mix, end with narration.
        audio_fc = "[1:a][2:a]amix=inputs=2:duration=first:dropout_transition=0[a]"
    else:
        audio_fc = "[1:a]anull[a]"

    fc = (
        f"[0:v]tpad=stop_mode=clone:stop_duration=3,format=yuv420p,"
        f"subtitles={subs_path.name}[v];{audio_fc}"
    )
    cmd = [
        str(FFMPEG_EXE), "-y", "-loglevel", "error",
        *inputs,
        "-filter_complex", fc,
        "-map", "[v]", "-map", "[a]",
        "-t", f"{total_dur:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        str(Path(out_path).resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                            cwd=str(TEMP_DIR))
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"Horror mux failed:\n{result.stderr.strip()}")
    return out_path


def assemble_horror_clips(
    story: dict,
    narration_audio: Path,
    clip_paths: list[Path],
    ambient: bool = True,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Assemble from PRE-MADE per-shot clips (animated Wan, already window-sized):
    concat → mux narration (+ drone) + watermark. 16x9."""
    if not FFMPEG_EXE.exists():
        raise FileNotFoundError(f"ffmpeg not found at {FFMPEG_EXE}")
    horror_id = story.get("horror_id") or story.get("_id")
    total_dur = _probe_duration(narration_audio)
    w, h = ASPECTS[ASPECT_KEY]
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    def _p(m):
        if progress_cb:
            progress_cb(m)

    _p("concatenating clips...")
    concat_path = TEMP_DIR / f"hvconcat_{horror_id}.mp4"
    _concat_segments(list(clip_paths), concat_path, f"horrorvid_{horror_id}")

    drone_path = None
    if ambient:
        _p("building ambient drone...")
        drone_path = TEMP_DIR / f"hdrone_{horror_id}.wav"
        _build_drone(total_dur, drone_path)

    subs_path = TEMP_DIR / f"overlay_horror_{horror_id}.ass"
    # Per-clip durations give caption timing (one clip per beat, in order).
    clip_durs = [max(0.1, _probe_duration(c)) for c in clip_paths]
    _write_captions_ass(total_dur, w, h, subs_path,
                        beats=story.get("beats", []),
                        spans=_spans_from_durations(clip_durs))

    _p("muxing...")
    out_path = FINAL_DIR / f"horror_{horror_id}_{ASPECT_KEY}.mp4"
    _mux_horror(concat_path, narration_audio, drone_path, total_dur, subs_path, out_path)
    log.info(f"✅ horror (animated) ready: {out_path.name}")
    return {"horror_id": horror_id, "clip_count": len(clip_paths),
            "duration": total_dur, ASPECT_KEY: out_path}


def assemble_horror(
    story: dict,
    narration_audio: Path,
    scene_durations: list[float],
    scene_images: list[Path],
    ambient: bool = True,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Build the 16x9 horror video.

    scene_durations / scene_images are per-beat and same length as story['beats'].
    """
    if not FFMPEG_EXE.exists():
        raise FileNotFoundError(f"ffmpeg not found at {FFMPEG_EXE}")
    n = len(scene_images)
    if not (n == len(scene_durations) == len(story.get("beats", []))):
        raise ValueError(
            f"length mismatch: images={n} durations={len(scene_durations)} "
            f"beats={len(story.get('beats', []))}"
        )

    horror_id = story.get("horror_id") or story.get("_id")
    total_dur = _probe_duration(narration_audio)
    w, h = ASPECTS[ASPECT_KEY]
    log.info(f"Assembling horror {horror_id}: {n} beats, narration={total_dur:.1f}s")

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    def _p(m):
        if progress_cb:
            progress_cb(m)

    segments = []
    for i, img in enumerate(scene_images):
        if i % 10 == 0:
            _p(f"ken burns {i+1}/{n}...")
        seg = TEMP_DIR / f"hseg_{horror_id}_{i:04d}.mp4"
        _ken_burns_segment(img, max(0.5, float(scene_durations[i])), w, h, seg,
                           zoom_in=(i % 2 == 0))
        segments.append(seg)

    _p("concatenating...")
    concat_path = TEMP_DIR / f"hconcat_{horror_id}.mp4"
    _concat_segments(segments, concat_path, f"horror_{horror_id}")

    drone_path = None
    if ambient:
        _p("building ambient drone...")
        drone_path = TEMP_DIR / f"hdrone_{horror_id}.wav"
        _build_drone(total_dur, drone_path)

    subs_path = TEMP_DIR / f"overlay_horror_{horror_id}.ass"
    _write_captions_ass(total_dur, w, h, subs_path,
                        beats=story.get("beats", []),
                        spans=_spans_from_durations(scene_durations))

    _p("muxing...")
    out_path = FINAL_DIR / f"horror_{horror_id}_{ASPECT_KEY}.mp4"
    _mux_horror(concat_path, narration_audio, drone_path, total_dur, subs_path, out_path)

    log.info(f"✅ horror ready: {out_path.name}")
    return {"horror_id": horror_id, "beat_count": n, "duration": total_dur,
            ASPECT_KEY: out_path}
