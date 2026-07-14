"""
Claw Bot — lesson mode: putting the lesson together

Per-beat clips + the narration → one 16x9 lesson video with read-along captions.

The whole file exists because of one problem: **a lesson's clips do not match.** You
tick a few beats for real animation and leave the rest as stills, so the segment list
is half Wan and half Ken Burns — and those two do not agree on anything:

    Ken Burns (musicvideo_assembly)   30 fps, canvas-sized
    Wan (horror_video._retime_fill)   16 fps, the model's native size

`_concat_segments` joins with the concat demuxer and `-c copy`, which needs identical
codec parameters. Stream-copying a 16fps clip into a 30fps timeline does not error —
it RE-CONTAINERS the frames at the wrong rate, and the segment plays for a different
length than it was cut to. That is the exact bug that once cropped the narration off
the end of every video in this project ("fps relabel ≠ resample", CLAUDE.md).

So every segment — animated or still — goes through `normalize_segment()` first:
resampled to one fps, padded to one size, cut to exactly its beat's length. After that
the concat is a stream copy of things that genuinely match, and the video's duration
equals the narration's by construction.

Facts' own assembler is not reused directly: `assemble_facts_clips()` hardcodes a
`facts_<id>` filename and a caption layout with a `#N` fact badge, and a lesson written
to a facts filename is a file that lies about its mode. The proven mux underneath it
IS reused (`facts_assembly._mux_facts`) — it is mode-neutral, and it is where the
"narration wins, never -shortest" rule actually lives.
"""

import logging
import subprocess
from pathlib import Path
from typing import Callable, Optional

from modules import facts_assembly as fasm
from modules.assembly import ASPECTS
from modules.musicvideo_assembly import (FPS, TEMP_DIR, _concat_segments,
                                         _ken_burns_segment)
from modules.subtitles import ass_escape, ass_time, sentence_chunks, windows_to_events

log = logging.getLogger("claw_bot.lesson_assembly")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FFMPEG_EXE = PROJECT_ROOT / "00_Tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
LESSONS_DIR = PROJECT_ROOT / "04_Outputs" / "lessons"

ASPECT = "16x9"          # a lesson is watched on a screen, not a phone held upright


def normalize_segment(src: Path, dur: float, w: int, h: int, out: Path) -> Path:
    """One beat's clip, made identical to every other beat's clip.

    RESAMPLED to `FPS`, not relabelled: re-containering a 16fps Wan clip as 30fps
    keeps its frames and changes its DURATION, which is how a video ends up shorter
    than the narration it was cut for.

    `tpad` clones the last frame if the clip is short of its window (Wan tops out at
    5 seconds), so the picture never runs out before the teacher stops talking. The
    audio is dropped here — the narration is muxed once, at the end, over the whole
    lesson.
    """
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
          f"fps={FPS},setsar=1,"
          f"tpad=stop_mode=clone:stop_duration=3,format=yuv420p")
    cmd = [str(FFMPEG_EXE), "-y", "-loglevel", "error", "-i", str(src),
           "-vf", vf, "-t", f"{max(0.1, float(dur)):.3f}", "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"could not normalise {src.name}: {r.stderr.strip()[:200]}")
    return out


def still_segment(image: Path, dur: float, w: int, h: int, out: Path,
                  zoom_in: bool = True) -> Path:
    """A beat nobody ticked: the still, with a slow pan. Free — no GPU."""
    return _ken_burns_segment(image, max(0.5, float(dur)), w, h, out, zoom_in)


# ==============================================================================
# CAPTIONS
# ==============================================================================
# A lesson is read along with, not skimmed. So: the spoken words at the bottom, synced
# to the beat that is actually being said, and the beat's KEY TERM held at the top —
# a six-year-old needs to see the word, not just hear it. No `#N` badge: that is a
# Shorts device for counting facts, and a lesson is not a countdown.

_ASS_HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Term,Arial Black,{term},&H0000F0FF,&H00202020,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,8,60,60,54,1
Style: Sub,Arial,{sub},&H00FFFFFF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,3,3,1,2,90,90,64,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def write_lesson_ass(beats: list, durations: list, w: int, h: int,
                     path: Path) -> Path:
    """Read-along subtitles + the key term for each line."""
    lines = [_ASS_HEAD.format(w=w, h=h, term=int(h * 0.052), sub=int(h * 0.040))]

    t = 0.0
    for beat, dur in zip(beats, durations):
        dur = max(0.3, float(dur))
        start, end = t, t + dur
        t = end

        term = (beat.get("on_screen") or "").strip()
        if term:
            lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Term,,0,0,0,,"
                         f"{ass_escape(term)}")

        # The spoken line, chunked so a child can actually read it, spread across the
        # beat's REAL window — the words appear as they are said.
        for c0, c1, text in windows_to_events(
                [(start, end, beat.get("narration", ""))], sentence_chunks):
            lines.append(f"Dialogue: 0,{ass_time(c0)},{ass_time(c1)},Sub,,0,0,0,,"
                         f"{ass_escape(text)}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ==============================================================================
# ASSEMBLE
# ==============================================================================

def assemble_lesson(lesson: dict, narration: Path, clips: list, durations: list,
                    progress_cb: Optional[Callable[[str], None]] = None) -> dict:
    """Per-beat clips (mixed Wan + Ken Burns) + narration → the lesson video."""
    def _p(msg: str):
        log.info(msg)
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    lesson_id = lesson["lesson_id"]
    beats = lesson.get("beats", [])
    if not (len(beats) == len(clips) == len(durations)):
        raise ValueError(f"length mismatch: {len(beats)} beats, {len(clips)} clips, "
                         f"{len(durations)} durations")

    w, h = ASPECTS[ASPECT]
    out_dir = LESSONS_DIR / lesson_id
    (out_dir / "final").mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Make every segment identical — see normalize_segment(). This is what lets the
    #    concat be a stream copy, and what keeps the video the length of the words.
    _p(f"🎞️ normalising {len(clips)} segments to {FPS}fps {w}x{h}…")
    segs = []
    for i, (clip, dur) in enumerate(zip(clips, durations)):
        seg = TEMP_DIR / f"lseg_{lesson_id}_{i:03d}.mp4"
        segs.append(normalize_segment(Path(clip), dur, w, h, seg))

    body = TEMP_DIR / f"lbody_{lesson_id}.mp4"
    _concat_segments(segs, body, f"lesson_{lesson_id}")

    total = fasm._probe_duration(narration)
    _p(f"🗣️ narration {total:.1f}s — the video is cut to the words, never the reverse")

    subs = TEMP_DIR / f"lesson_{lesson_id}.ass"
    write_lesson_ass(beats, durations, w, h, subs)

    out = out_dir / "final" / f"lesson_{lesson_id}_{ASPECT}.mp4"
    # The proven mux: burns the subs, caps at the NARRATION's length, and never uses
    # -shortest (which crops the last word off).
    fasm._mux_facts(body, narration, None, total, subs, out, w=w, h=h,
                    progress_cb=progress_cb)

    _p(f"✅ lesson assembled: {out.name}")
    return {"lesson_id": lesson_id, ASPECT: out, "duration": total,
            "beat_count": len(beats)}
