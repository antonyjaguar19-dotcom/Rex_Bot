"""
Claw Bot — Facts Shorts Assembly (vertical 9x16 by default)

Tiles a loose mood backdrop per fact with slow Ken Burns motion, burns the fact
as LARGE centered on-screen text (the star of the frame — the image is just
wallpaper), and mixes an optional music bed under the narration. Captions are
timed to the REAL per-beat voice spans (durations), so text flips exactly when
the narrator moves to the next fact.

Reuses the Ken Burns / concat helpers from musicvideo_assembly.
"""

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules.assembly import (FFMPEG_EXE, FFPROBE_EXE, FINAL_DIR, ASPECTS,
                              _probe_duration)
from modules.musicvideo_assembly import (
    TEMP_DIR, WATERMARK_TEXT, WATERMARK_PNG, logo_overlay_filter,
    _ken_burns_segment, _concat_segments,
)
from modules.subtitles import ass_time, ass_escape, sentence_chunks, windows_to_events

log = logging.getLogger("claw_bot.facts_assembly")

# The bed is NORMALISED to a target loudness, not scaled by a fixed factor.
# A fixed volume=0.16 assumes MusicGen hands us a track at a known level, and it
# does not — one take came out at -29 LUFS, so x0.16 dropped the bed to -45 LUFS
# against a -18.8 LUFS voice. 26 dB under is not "subtle", it is inaudible: the
# reel sounded exactly like one with no music at all.
# -32 LUFS puts it ~13 dB under the narration: clearly there, never competing.
MUSIC_LUFS = -32.0
MUSIC_TP = -2.0       # true-peak ceiling, so a loud bar cannot poke through the voice


def _wrap(text: str, max_chars: int = 22) -> str:
    """Greedy word-wrap into short lines for big centered text (\\N = ASS break)."""
    words, lines, cur = (text or "").split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\\N".join(lines)


def _spans_from_durations(durations: list) -> list:
    spans, cur = [], 0.0
    for d in durations:
        spans.append((cur, cur + float(d)))
        cur += float(d)
    return spans


def _write_facts_ass(total_dur: float, w: int, h: int, path: Path,
                     beats: list, spans: list) -> Path:
    """Two-layer captions:
      TOP    — the punchy fact label (on_screen) + a # number badge above it.
      BOTTOM — real subtitles of the SPOKEN narration, sentence-chunked + synced
               within each beat's real voice span, so viewers read along.
    Plus the Rexjaw watermark."""
    title_size = max(38, round(h * 0.050))
    num_size = max(26, round(h * 0.030))
    sub_size = max(30, round(h * 0.036))
    wm_size = max(16, round(h * 0.020))
    side = max(60, round(w * 0.08))
    top_num_mv = max(40, round(h * 0.055))    # # badge near top edge
    top_title_mv = max(120, round(h * 0.115))  # label just below the badge
    sub_mv = max(80, round(h * 0.10))          # subtitle above bottom edge
    # an=8 top-center, an=2 bottom-center.
    styles = [
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, "
        "Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV\n",
        f"Style: Title,Arial Black,{title_size},&H00FFFFFF,&H00000000,&H90000000,"
        f"1,0,1,4,2,8,{side},{side},{top_title_mv}\n",
        f"Style: Num,Arial Black,{num_size},&H0055CCFF,&H00000000,&H00000000,"
        f"1,0,1,2,1,8,40,40,{top_num_mv}\n",
        f"Style: Sub,Arial,{sub_size},&H00FFFFFF,&H00000000,&H90000000,"
        f"1,0,1,2,1,2,{side},{side},{sub_mv}\n",
    ]
    # Watermark is now the brand logo PNG (overlaid in the mux), not burned text.
    events = []
    for b, (t0, t1) in zip(beats, spans):
        if t1 <= t0:
            continue
        kind = b.get("kind")
        # Animation override tags (ASS): the label BOUNCE-POPS in (45% -> 108%
        # overshoot -> 100%) + fades; the badge pops, subtitles fade. Tags are
        # literal ASS braces (not escaped). The overshoot makes it clearly visible
        # on every title (a plain 170ms scale was too quick to notice on some).
        POP = (r"{\fad(90,50)\fscx45\fscy45"
               r"\t(0,170,\fscx108\fscy108)\t(170,250,\fscx100\fscy100)}")
        FADE_N = (r"{\fad(120,70)\fscx55\fscy55"
                  r"\t(0,180,\fscx100\fscy100)}")
        FADE_S = r"{\fad(90,70)}"
        # TOP label (escape FIRST, then insert \N breaks, then prepend the tag).
        raw = (b.get("on_screen") or b.get("narration") or "").strip()
        title = _wrap(ass_escape(raw))
        if title:
            events.append(
                f"Dialogue: 0,{ass_time(t0)},{ass_time(t1)},Title,,0,0,0,,{POP}{title}\n")
        if kind == "fact" and b.get("index"):
            events.append(
                f"Dialogue: 0,{ass_time(t0)},{ass_time(t1)},Num,,0,0,0,,{FADE_N}#{b['index']}\n")
        # BOTTOM subtitle = spoken narration, sentence-chunked + synced across the
        # beat's real span. Skip the outro (its label already IS the spoken line).
        narr = (b.get("narration") or "").strip()
        if narr and kind != "outro":
            for c0, c1, ctext in windows_to_events([(t0, t1, narr)], sentence_chunks):
                events.append(
                    f"Dialogue: 0,{ass_time(c0)},{ass_time(c1)},Sub,,0,0,0,,{FADE_S}{ass_escape(ctext)}\n")

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {w}\nPlayResY: {h}\n"
        "ScaledBorderAndShadow: yes\n\n"
        + "".join(styles) + "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    path.write_text(header + "".join(events), encoding="utf-8")
    return path


# Stamped into the reel's metadata when the thumbnail is held on its front, so a
# second pass can see it is already there.
_THUMB_TAG = "rexjaw_thumb_hold"

# The reel exactly as assembled, before any poster frame was held on its front.
# A SUBFOLDER of final/ — every finals glob in the project is non-recursive, so
# these masters never show up as a second copy of the reel.
POSTERLESS_DIR = FINAL_DIR / "_posterless"


def _thumb_hold_of(video: Path) -> Optional[float]:
    """Seconds of thumbnail already held on this reel, or None if it has none."""
    r = subprocess.run(
        [str(FFPROBE_EXE), "-v", "error", "-show_entries", "format_tags=comment",
         "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True, timeout=60)
    tag = (r.stdout or "").strip()
    if tag.startswith(f"{_THUMB_TAG}="):
        try:
            return float(tag.split("=", 1)[1])
        except ValueError:
            return None
    return None


def _probe_dims(video: Path) -> tuple[int, int]:
    """The reel's real frame size — never assume 1080x1920; 16x9 and 1x1 exist too."""
    r = subprocess.run(
        [str(FFPROBE_EXE), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(video)],
        capture_output=True, text=True, timeout=60)
    try:
        w, h = (int(x) for x in (r.stdout or "").strip().split("x")[:2])
        return w, h
    except Exception:
        return 1080, 1920


def thumb_hold_of(video: Path) -> Optional[float]:
    """Public read of the poster-hold tag — callers outside this module need to
    know whether a reel already opens on its thumbnail."""
    return _thumb_hold_of(video)


def _posterless_copy(video: Path, existing_hold: Optional[float]) -> Optional[Path]:
    """The reel as it was BEFORE a poster frame was held on its front.

    Kept so a re-rolled thumbnail can replace the old one instead of stacking on
    top of it (or forcing a re-encode of a re-encode). Reels made before this
    existed are rebuilt by cutting the recorded hold back off.
    """
    src = POSTERLESS_DIR / video.name
    if src.exists():
        return src
    POSTERLESS_DIR.mkdir(parents=True, exist_ok=True)
    if existing_hold is None:
        shutil.copy2(video, src)
        return src
    r = subprocess.run(
        [str(FFMPEG_EXE), "-y", "-loglevel", "error",
         "-ss", f"{existing_hold:.3f}", "-i", str(video.resolve()),
         "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         str(src)],
        capture_output=True, text=True, timeout=900)
    if r.returncode != 0 or not src.exists():
        log.warning(f"could not strip the old poster frame ({r.stderr[-200:]})")
        src.unlink(missing_ok=True)
        return None
    return src


def prepend_still(video: Path, image: Path, hold_sec: float = 0.5,
                  replace: bool = False) -> Path:
    """Hold the thumbnail image on the front of the reel for a fraction of a second.

    Shorts custom thumbnails are not offered in every region, and where they are not,
    the platform grabs the FIRST FRAME. So the thumbnail has to BE the first frame —
    generating a beautiful one that nobody can upload is just a nice file on disk.

    Held, not animated: it is a poster frame, not a shot.

    Rewrites the reel in place (same filename), so every path already handed out —
    publish kit, Discord, the dashboard — keeps working. The narration moves with the
    picture, and the captions are burned into the pixels, so nothing desyncs.

    `replace=True` swaps a poster the reel already carries (a re-rolled thumbnail).
    It is built from a pristine posterless copy, so re-rolling ten times costs one
    generation of re-encode, not ten — and never stacks holds.
    """
    if not image or not Path(image).exists():
        return video

    existing = _thumb_hold_of(video)
    source = video
    if existing is not None:
        # The reel remembers that it already carries a poster frame. Without this
        # a publish-kit re-run would bolt ANOTHER half-second of still onto the
        # front; with it, a re-rolled thumbnail never reached the video at all.
        if not replace:
            log.info(f"{video.name} already opens on its thumbnail; not stacking another")
            return video
        source = _posterless_copy(video, existing)
        if source is None:
            return video
    else:
        _posterless_copy(video, None)      # keep the clean master for later re-rolls

    hold = max(0.1, min(float(hold_sec), 3.0))
    w, h = _probe_dims(video)
    tmp = video.with_name(f"{video.stem}_thumbed.mp4")

    r = subprocess.run(
        [str(FFMPEG_EXE), "-y", "-loglevel", "error",
         "-loop", "1", "-t", f"{hold:.3f}", "-i", str(Path(image).resolve()),
         "-i", str(Path(source).resolve()),
         "-f", "lavfi", "-t", f"{hold:.3f}",
         "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
         "-filter_complex",
         # The still is fitted to the reel's own frame — a thumbnail rendered at a
         # different size would otherwise force a resolution change mid-file.
         f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
         f"crop={w}:{h},setsar=1,fps=16,format=yuv420p[t];"
         f"[1:v]scale={w}:{h},setsar=1,fps=16,format=yuv420p[v];"
         f"[t][v]concat=n=2:v=1:a=0[outv];"
         f"[2:a][1:a]concat=n=2:v=0:a=1[outa]",
         "-map", "[outv]", "-map", "[outa]",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         "-metadata", f"comment={_THUMB_TAG}={hold:.3f}",   # output option, not input
         str(tmp)],
        capture_output=True, text=True, timeout=900)
    if r.returncode != 0 or not tmp.exists():
        log.warning(f"thumbnail hold failed ({r.stderr[-200:]}); shipping without it")
        return video
    tmp.replace(video)
    log.info(f"🖼️ thumbnail held on the front of {video.name} for {hold:.2f}s")
    return video


def trim_trailing_silence(music: Path) -> Path:
    """Cut the dead air off the end of a generated track.

    ACE-Step composes a piece SHORTER than the length asked for and pads the rest
    with silence — a 34.4s request came back as 27s of music and 7.4s of nothing.
    Left alone, that silence is what lands on the last fact. The music has to be
    found before it can be aligned, so the padding goes first.
    """
    out = music.with_name(f"{music.stem}_trimmed.wav")
    r = subprocess.run(
        [str(FFMPEG_EXE), "-y", "-loglevel", "error", "-i", str(music),
         "-af", "areverse,silenceremove=start_periods=1:start_threshold=-50dB:"
                "start_silence=0.1,areverse",
         str(out)],
        capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not out.exists():
        log.warning(f"silence trim failed ({r.stderr[:120]}); using the raw bed")
        return music
    before, after = _probe_duration(music), _probe_duration(out)
    if after < 1.0:                     # the whole thing read as silence — trust the source
        return music
    log.info(f"🎵 trimmed dead tail: {before:.1f}s -> {after:.1f}s of music")
    return out


def _align_music_tail(music: Path, total_dur: float) -> Path:
    """Land the music's LAST note on the reel's last frame.

    ACE-Step is asked for the reel's length but does not hit it to the sample, and
    the mux cuts the bed dead at the narration's end. Off by a second and the outro
    it composed gets chopped — which is the sudden cut we are trying to remove.

    So the END is anchored, not the start: a long track is trimmed from the FRONT
    (keeping its ending), a short one is pushed back with leading silence. Either
    way the final chord lands with the final word.
    """
    d = _probe_duration(music)
    if abs(d - total_dur) < 0.08:
        return music

    out = music.with_name(f"{music.stem}_aligned.wav")
    if d > total_dur:
        args = ["-ss", f"{d - total_dur:.3f}", "-i", str(music)]
        af = []
    else:
        args = ["-i", str(music)]
        af = ["-af", f"adelay={int(round((total_dur - d) * 1000))}:all=1"]
    r = subprocess.run(
        [str(FFMPEG_EXE), "-y", "-loglevel", "error", *args, *af,
         "-t", f"{total_dur:.3f}", str(out)],
        capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not out.exists():
        log.warning(f"music tail-align failed ({r.stderr[:120]}); using the raw bed")
        return music
    log.info(f"🎵 music tail aligned: {d:.2f}s -> {total_dur:.2f}s "
             f"({'trimmed from the front' if d > total_dur else 'delayed'})")
    return out


def _mux_facts(video_path: Path, narration_path: Path, music_path: Optional[Path],
               total_dur: float, subs_path: Path, out_path: Path,
               w: int = 1080, h: int = 1920) -> Path:
    inputs = ["-i", str(Path(video_path).resolve()),
              "-i", str(Path(narration_path).resolve())]
    if music_path is not None and Path(music_path).exists():
        music_path = _align_music_tail(Path(music_path), total_dur)
        inputs += ["-i", str(Path(music_path).resolve())]
        # Music (input 2) normalised to a fixed loudness, then mixed under the
        # narration (input 1); the mix ends on the narration.
        # amix divides by the input count, which would halve the VOICE too — so the
        # voice is handed through at unity and the bed is set by loudnorm alone.
        audio_fc = (
            f"[2:a]loudnorm=I={MUSIC_LUFS}:TP={MUSIC_TP}:LRA=11[m];"
            f"[1:a][m]amix=inputs=2:duration=first:dropout_transition=0:"
            f"weights='1 1':normalize=0[a]"
        )
    else:
        audio_fc = "[1:a]anull[a]"

    # Fit the source to the w×h vertical canvas: a blurred COVER fill behind the
    # sharp source scaled to FIT (no stretch, no black bars). A full-frame source
    # (Ken Burns path, already w×h) just covers the fill = unchanged. A landscape
    # Wan clip (832x480) sits centered over its own blurred enlargement.
    fit = (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},gblur=sigma=24[bgb];"
        f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2:shortest=1[fit];"
    )
    # Brand logo overlay (bottom-right) — added as the last input if present.
    logo_idx = len(inputs) // 2
    wm = logo_overlay_filter("vsub", "v", logo_idx, w) if WATERMARK_PNG.exists() else ""
    vtail = "[vsub]" if wm else "[v]"
    fc = (f"{fit}[fit]tpad=stop_mode=clone:stop_duration=3,format=yuv420p,"
          f"subtitles={subs_path.name}{vtail};{audio_fc}")
    if wm:
        inputs += ["-i", str(WATERMARK_PNG.resolve())]
        fc = f"{fc};{wm}"
    cmd = [
        str(FFMPEG_EXE), "-y", "-loglevel", "error",
        *inputs,
        "-filter_complex", fc,
        "-map", "[v]", "-map", "[a]",
        "-t", f"{total_dur:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        str(Path(out_path).resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                            cwd=str(TEMP_DIR))
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"Facts mux failed:\n{result.stderr.strip()}")
    return out_path


def assemble_facts_clips(
    story: dict,
    narration_audio: Path,
    clip_paths: list[Path],
    aspect: str = "9x16",
    music_path: Optional[Path] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Assemble from PRE-MADE per-beat clips (animated Wan, window-sized): concat
    → big captions (timed to real clip durations) → mux narration (+music). 9x16."""
    if not FFMPEG_EXE.exists():
        raise FileNotFoundError(f"ffmpeg not found at {FFMPEG_EXE}")
    beats = story.get("beats", [])
    facts_id = story.get("facts_id") or story.get("_id")
    total_dur = _probe_duration(narration_audio)
    w, h = ASPECTS.get(aspect, ASPECTS["9x16"])
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    def _p(m):
        if progress_cb:
            progress_cb(m)

    _p("concatenating clips...")
    concat_path = TEMP_DIR / f"fvconcat_{facts_id}.mp4"
    _concat_segments(list(clip_paths), concat_path, f"factsvid_{facts_id}")

    clip_durs = [max(0.1, _probe_duration(c)) for c in clip_paths]
    subs_path = TEMP_DIR / f"overlay_facts_{facts_id}.ass"
    _write_facts_ass(total_dur, w, h, subs_path, beats, _spans_from_durations(clip_durs))

    _p("muxing...")
    out_path = FINAL_DIR / f"facts_{facts_id}_{aspect}.mp4"
    _mux_facts(concat_path, narration_audio, music_path, total_dur, subs_path, out_path, w, h)
    log.info(f"✅ facts reel (animated) ready: {out_path.name}")
    return {"facts_id": facts_id, "beat_count": len(clip_paths),
            "duration": total_dur, aspect: out_path}


def assemble_facts(
    story: dict,
    narration_audio: Path,
    durations: list[float],
    backgrounds: list[Path],
    aspect: str = "9x16",
    music_path: Optional[Path] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Build the vertical facts reel. durations/backgrounds are per-beat, same
    length as story['beats']."""
    if not FFMPEG_EXE.exists():
        raise FileNotFoundError(f"ffmpeg not found at {FFMPEG_EXE}")
    beats = story.get("beats", [])
    n = len(backgrounds)
    if not (n == len(durations) == len(beats)):
        raise ValueError(f"length mismatch: bg={n} durations={len(durations)} beats={len(beats)}")

    facts_id = story.get("facts_id") or story.get("_id")
    total_dur = _probe_duration(narration_audio)
    w, h = ASPECTS.get(aspect, ASPECTS["9x16"])
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    def _p(m):
        if progress_cb:
            progress_cb(m)

    segments = []
    for i, img in enumerate(backgrounds):
        if i % 5 == 0:
            _p(f"ken burns {i+1}/{n}...")
        seg = TEMP_DIR / f"fseg_{facts_id}_{i:04d}.mp4"
        _ken_burns_segment(img, max(0.5, float(durations[i])), w, h, seg,
                           zoom_in=(i % 2 == 0))
        segments.append(seg)

    _p("concatenating...")
    concat_path = TEMP_DIR / f"fconcat_{facts_id}.mp4"
    _concat_segments(segments, concat_path, f"facts_{facts_id}")

    subs_path = TEMP_DIR / f"overlay_facts_{facts_id}.ass"
    _write_facts_ass(total_dur, w, h, subs_path, beats,
                     _spans_from_durations(durations))

    _p("muxing...")
    out_path = FINAL_DIR / f"facts_{facts_id}_{aspect}.mp4"
    _mux_facts(concat_path, narration_audio, music_path, total_dur, subs_path, out_path, w, h)
    log.info(f"✅ facts reel ready: {out_path.name}")
    return {"facts_id": facts_id, "beat_count": n, "duration": total_dur,
            aspect: out_path}
