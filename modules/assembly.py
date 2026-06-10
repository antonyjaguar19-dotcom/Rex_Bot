"""
Claw Bot — Final Assembly (Phase 6A)

Stitch all upscaled per-shot clips of a script_id into one final MP4,
with 0.3s crossfade transitions between shots.

Exports TWO versions:
  - 9:16 vertical (YouTube Shorts) — pad/crop to 1080x1920
  - 16:9 horizontal — pad/crop to 1920x1080

Pattern: pure ffmpeg orchestration, mirrors clip_generator.py conventions.
No Discord, no LLM. The bot calls this once videos are approved.
"""

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Callable
# Music is best-effort: import lazily so assembly still works if music_generator
# breaks. We import inside the function, not at module load.

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import runtime_settings as rs

log = logging.getLogger("claw_bot.assembly")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
FFMPEG_EXE = PROJECT_ROOT / "00_Tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE_EXE = PROJECT_ROOT / "00_Tools" / "ffmpeg" / "bin" / "ffprobe.exe"
CLIPS_DIR = PROJECT_ROOT / "04_Outputs" / "clips"
FINAL_DIR = PROJECT_ROOT / "04_Outputs" / "final"

CROSSFADE_SEC = 0.3
# Hard-cut breath: trailing pad added to each shot (except the last) so the
# narration's last word finishes before the instant cut to the next shot.
# Video freezes its last frame for this long (NOT black), audio gets silence —
# both padded equally so A/V stays synced. The cut itself is still instant.
CUT_GAP_SEC = 0.4
FPS = 24

# Target canvases
ASPECTS = {
    "9x16": (1080, 1920),   # YouTube Shorts / TikTok / Reels
    "16x9": (1920, 1080),   # Standard horizontal / YouTube
    "1x1":  (1080, 1080),   # Instagram square / cross-platform
}


# ==============================================================================
# HELPERS
# ==============================================================================
def _mix_music_under_narration(
    narration_video: Path,
    music_track: Path,
    output_path: Path,
    music_db: float = -12.0,
) -> Path:
    """Mix a music track UNDER the existing narration audio of a video.

    Music is ducked by `music_db` dB so narration stays in front.
    Video stream is copied (no re-encode). Audio is remixed.
    """
    cmd = [
        str(FFMPEG_EXE), "-y", "-loglevel", "error",
        "-i", str(narration_video),
        "-i", str(music_track),
        "-filter_complex",
        # amix halves volume by inputs=2; pre-boost narration 2x to compensate,
        # then duck music to music_db. Final mix: narration at unity, music at music_db.
        f"[0:a]volume=2.0[nar];"
        f"[1:a]volume={music_db}dB,volume=2.0[mlow];"
        f"[nar][mlow]amix=inputs=2:duration=first:dropout_transition=2[mixed]",
        "-map", "0:v",
        "-map", "[mixed]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg music-mix failed:\n{result.stderr.strip()}"
        )
    return output_path

def _probe_duration(path: Path) -> float:
    """Return media duration in seconds via ffprobe."""
    cmd = [
        str(FFPROBE_EXE),
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
    return float(r.stdout.strip())


def _has_audio(path: Path) -> bool:
    """Return True if the file contains at least one audio stream."""
    cmd = [
        str(FFPROBE_EXE),
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return bool(r.stdout.strip())


def _find_clips_for_script(script_id: str) -> list[Path]:
    """
    Find all clips for a script_id, picking the latest revision per shot.
    Returns clips sorted by shot number ascending.

    Naming convention (from clip_generator/upscaler):
      clip_<script_id>_shot<N>[_v<rev>].mp4
    """
    pattern = f"clip_{script_id}_shot*.mp4"
    candidates = list(CLIPS_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No clips found in {CLIPS_DIR} matching {pattern}"
        )

    # Group by shot number, keep latest revision
    by_shot: dict[int, Path] = {}
    for p in candidates:
        # Parse: clip_<script_id>_shot<N>[_v<rev>].mp4
        stem = p.stem  # filename without .mp4
        try:
            # Find the "shot<N>" segment
            after_shot = stem.split("_shot", 1)[1]
            # Strip _v<rev> if present
            shot_part = after_shot.split("_")[0]
            shot_num = int(shot_part)
        except (IndexError, ValueError):
            log.warning(f"Skipping unparseable clip name: {p.name}")
            continue

        # Determine revision (default 0)
        rev = 0
        if "_v" in after_shot:
            try:
                rev = int(after_shot.split("_v")[1])
            except (IndexError, ValueError):
                pass

        existing = by_shot.get(shot_num)
        if existing is None:
            by_shot[shot_num] = p
        else:
            # Compare revisions on existing
            existing_after = existing.stem.split("_shot", 1)[1]
            existing_rev = 0
            if "_v" in existing_after:
                try:
                    existing_rev = int(existing_after.split("_v")[1])
                except (IndexError, ValueError):
                    pass
            if rev > existing_rev:
                by_shot[shot_num] = p

    ordered = [by_shot[k] for k in sorted(by_shot.keys())]
    log.info(f"Found {len(ordered)} shots for {script_id}: "
             f"{[p.name for p in ordered]}")
    return ordered


def _build_xfade_filter(
    clip_count: int,
    durations: list[float],
    target_w: int,
    target_h: int,
    audio_input_idx: list[int],
) -> str:
    """
    Build the filter_complex string for chained crossfades + canvas fit.

    Each clip is first scaled+padded to (target_w, target_h), then xfaded.
    Audio is acrossfaded in parallel.

    audio_input_idx[i] = the ffmpeg -i index where clip i's audio lives.
    (Same as video index if clip has audio; separate index if silent
    audio was synthesized via anullsrc.)

    NOTE: ffmpeg -i indices for VIDEO are sequential per clip (0,1,2...)
    only when no silent injections happen. When a silent track is
    synthesized for a clip, it consumes the NEXT -i slot. So this
    builder needs the video input index for each clip too — but since
    we always add the video clip FIRST and audio_input_idx tells us
    the audio location, we derive video indices by walking the list.

    Returns the filter graph string. Final outputs labeled [vout] and [aout].
    """
    # Derive video input index per clip: it's audio_input_idx[i] if that
    # clip's audio came from the same file, otherwise audio_input_idx[i] - 1.
    # Simpler: walk forward, video always comes first.
    video_input_idx: list[int] = []
    cursor = 0
    for i in range(clip_count):
        video_input_idx.append(cursor)
        # Next clip's video starts after this clip's video AND
        # any synthesized silent audio input for this clip.
        cursor += 1
        if audio_input_idx[i] != video_input_idx[i]:
            cursor += 1  # silent audio took a slot

    parts = []

    # Step 1: normalize each input to target canvas (scale + pad, no crop).
    # Every clip except the last gets a CROSSFADE_SEC silent/frozen TAIL so the
    # crossfade overlap region lands on that pad — NOT on the shot's narration.
    # Without the pad, xfade pulls the next clip CROSSFADE_SEC earlier and the
    # two shots' narration mix together (the bug this fixes).
    for i in range(clip_count):
        v_idx = video_input_idx[i]
        chain = (
            f"[{v_idx}:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1,fps={FPS},format=yuv420p"
        )
        if i < clip_count - 1:
            chain += f",tpad=stop_duration={CROSSFADE_SEC}:stop_mode=clone"
        parts.append(chain + f"[v{i}]")

    # Pad each non-last clip's audio with CROSSFADE_SEC of trailing silence so
    # the acrossfade overlaps silence-into-next, never narration-into-narration.
    for i in range(clip_count):
        a_idx = audio_input_idx[i]
        if i < clip_count - 1:
            parts.append(f"[{a_idx}:a]apad=pad_dur={CROSSFADE_SEC}[a{i}]")
        else:
            parts.append(f"[{a_idx}:a]anull[a{i}]")

    if clip_count == 1:
        # No crossfade needed — use null/anull (copy/acopy are codec specs, not filters)
        parts.append(f"[v0]null[vout]")
        parts.append(f"[a0]anull[aout]")
        return ";".join(parts)

    # Step 2: chain xfades. With the silent tail pad, the transition starts
    # exactly when the previous shot's narration ends (raw cumulative duration).
    cum = 0.0
    prev_v_label = "v0"
    prev_a_label = "a0"

    for i in range(1, clip_count):
        cum += durations[i - 1]
        offset = max(0.0, cum)  # prev clip's padded tail begins here

        is_last = (i == clip_count - 1)
        v_out = "vout" if is_last else f"vx{i}"
        a_out = "aout" if is_last else f"ax{i}"

        parts.append(
            f"[{prev_v_label}][v{i}]xfade=transition=fade:"
            f"duration={CROSSFADE_SEC}:offset={offset:.3f}[{v_out}]"
        )
        parts.append(
            f"[{prev_a_label}][a{i}]acrossfade="
            f"d={CROSSFADE_SEC}[{a_out}]"
        )

        prev_v_label = v_out
        prev_a_label = a_out

    return ";".join(parts)


def _build_cut_filter(
    clip_count: int,
    target_w: int,
    target_h: int,
    audio_input_idx: list[int],
) -> str:
    """
    Build filter_complex for HARD-CUT assembly — no crossfade, no overlap.
    Each clip plays its full length then cuts immediately to the next.
    Narration is never altered or mixed. Final outputs labeled [vout] [aout].
    """
    # Derive video input index per clip (same logic as the xfade builder).
    video_input_idx: list[int] = []
    cursor = 0
    for i in range(clip_count):
        video_input_idx.append(cursor)
        cursor += 1
        if audio_input_idx[i] != video_input_idx[i]:
            cursor += 1

    parts = []
    # Each shot except the last gets a CUT_GAP_SEC breath: video freezes its
    # last frame (tpad), audio gets matching trailing silence (apad). Equal pad
    # keeps the segment's v/a the same length, so concat stays in sync and the
    # cut to the next shot is still instant.
    for i in range(clip_count):
        v_idx = video_input_idx[i]
        chain = (
            f"[{v_idx}:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1,fps={FPS},format=yuv420p"
        )
        if i < clip_count - 1:
            chain += f",tpad=stop_duration={CUT_GAP_SEC}:stop_mode=clone"
        parts.append(chain + f"[v{i}]")
    for i in range(clip_count):
        a_idx = audio_input_idx[i]
        chain = f"[{a_idx}:a]aresample=48000"
        if i < clip_count - 1:
            chain += f",apad=pad_dur={CUT_GAP_SEC}"
        parts.append(chain + f"[a{i}]")

    if clip_count == 1:
        parts.append("[v0]null[vout]")
        parts.append("[a0]anull[aout]")
        return ";".join(parts)

    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(clip_count))
    parts.append(f"{concat_inputs}concat=n={clip_count}:v=1:a=1[vout][aout]")
    return ";".join(parts)


def _assemble_one_aspect(
    clips: list[Path],
    durations: list[float],
    aspect_key: str,
    output_path: Path,
    audio_flags: list[bool],
    progress_cb: Optional[Callable] = None,
    transition: str = "crossfade",
) -> Path:
    """Run ffmpeg to produce ONE aspect-ratio version.

    audio_flags[i] = True if clip i has an audio stream, False otherwise.
    Audio-less clips get a synthesized silent track of matching duration.
    transition = "crossfade" (0.3s dissolve, silent-padded) or "cut" (hard cut).
    """
    target_w, target_h = ASPECTS[aspect_key]
    log.info(f"Assembling {aspect_key} → {output_path.name} ({target_w}x{target_h})")
    if progress_cb:
        progress_cb(f"rendering {aspect_key}...")

    cmd = [str(FFMPEG_EXE), "-y", "-loglevel", "error"]

    # Add each clip as input. For audio-less clips, add a silent audio source
    # as a SEPARATE input right after — we'll alias it in the filter graph.
    # We track the input index for each clip's audio stream.
    audio_input_idx: list[int] = []   # which ffmpeg -i index holds clip i's audio
    next_input = 0

    for i, c in enumerate(clips):
        cmd += ["-i", str(c)]
        clip_v_idx = next_input
        next_input += 1

        if audio_flags[i]:
            audio_input_idx.append(clip_v_idx)  # audio is in same file
        else:
            # Synthesize silent audio of matching duration
            cmd += [
                "-f", "lavfi",
                "-t", f"{durations[i]:.3f}",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            ]
            audio_input_idx.append(next_input)
            next_input += 1

    if transition == "cut":
        filter_complex = _build_cut_filter(
            len(clips), target_w, target_h, audio_input_idx
        )
    else:
        filter_complex = _build_xfade_filter(
            len(clips), durations, target_w, target_h, audio_input_idx
        )

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg assembly failed ({aspect_key}):\n{result.stderr.strip()}"
        )
    return output_path


# ==============================================================================
# PUBLIC API
# ==============================================================================

def assemble_final(
    script_id: str,
    progress_cb: Optional[Callable] = None,
) -> dict:
    """
    Stitch all approved/upscaled clips of a script into final MP4s.

    Produces three files in 04_Outputs/final/:
      - final_<script_id>_9x16.mp4  (YouTube Shorts / TikTok / Reels)
      - final_<script_id>_16x9.mp4  (YouTube standard)
      - final_<script_id>_1x1.mp4   (Instagram square)

    Returns:
        {
          "script_id": str,
          "shot_count": int,
          "9x16": Path,
          "16x9": Path,
          "1x1":  Path,
          "total_duration_sec": float,
        }
    """
    _asm_start = time.time()
    if not FFMPEG_EXE.exists():
        raise FileNotFoundError(f"ffmpeg not found at {FFMPEG_EXE}")
    if not FFPROBE_EXE.exists():
        raise FileNotFoundError(f"ffprobe not found at {FFPROBE_EXE}")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    clips = _find_clips_for_script(script_id)
    if len(clips) == 0:
        raise RuntimeError(f"No clips to assemble for {script_id}")

    transition = rs.get_effective_transition_mode()
    durations = [_probe_duration(c) for c in clips]
    audio_flags = [_has_audio(c) for c in clips]
    silent_count = sum(1 for f in audio_flags if not f)
    # Both modes preserve every shot's full narration:
    #  - cut: clips play full length back-to-back + a CUT_GAP_SEC breath between
    #  - crossfade: silent tail absorbs the overlap, narration untouched
    if transition == "cut":
        total_dur = sum(durations) + max(0, len(clips) - 1) * CUT_GAP_SEC
    else:
        total_dur = sum(durations)
    log.info(f"Total clips: {len(clips)}, transition={transition}, "
             f"sum={sum(durations):.2f}s, total={total_dur:.2f}s")
    if silent_count:
        log.info(f"Note: {silent_count}/{len(clips)} clips have no audio — "
                 f"silent track will be injected.")

    if progress_cb:
        progress_cb(f"assembling {len(clips)} shots, ~{total_dur:.1f}s total")

    outputs = {}
    for aspect_key in ("9x16", "16x9", "1x1"):
        out_path = FINAL_DIR / f"final_{script_id}_{aspect_key}.mp4"
        _assemble_one_aspect(
            clips, durations, aspect_key, out_path, audio_flags, progress_cb,
            transition=transition,
        )
        outputs[aspect_key] = out_path
        log.info(f"✅ {aspect_key} ready (no music yet): {out_path.name}")

    # ── Phase 6B: generate background music (once) and mix under narration ──
    music_track = None
    try:
        from modules import music_generator as mg
        mood = mg.get_effective_mood()  # uses override > script > default
        if progress_cb:
            progress_cb(f"generating background music ({mood})...")
        music_track = mg.generate_music(
            mood=mood,
            duration_sec=total_dur,
            script_id=script_id,
            progress_cb=progress_cb,
        )
    except Exception as e:
        log.exception(f"Music generation failed (continuing without music): {e}")

    if music_track is not None and music_track.exists():
        log.info(f"🎵 Mixing music under narration at -12 dB: {music_track.name}")
        for aspect_key in ("9x16", "16x9", "1x1"):
            silent_final = outputs[aspect_key]
            tmp_mixed = silent_final.with_suffix(".mixed.mp4")
            try:
                if progress_cb:
                    progress_cb(f"mixing music into {aspect_key}...")
                _mix_music_under_narration(
                    silent_final, music_track, tmp_mixed, music_db=-12.0
                )
                # Replace original with mixed version
                silent_final.unlink(missing_ok=True)
                tmp_mixed.rename(silent_final)
                log.info(f"✅ {aspect_key} final with music: {silent_final.name} "
                         f"({silent_final.stat().st_size/(1024*1024):.2f} MB)")
            except Exception as e:
                log.exception(f"Music mix failed for {aspect_key}: {e}")
                tmp_mixed.unlink(missing_ok=True)
    else:
        log.warning("No music track available — finals will have narration only.")
        if progress_cb:
            progress_cb("finals ready (no music)")

    return {
        "script_id": script_id,
        "shot_count": len(clips),
        "9x16": outputs["9x16"],
        "16x9": outputs["16x9"],
        "1x1":  outputs["1x1"],
        "total_duration_sec": total_dur,
        "wall_sec": round(time.time() - _asm_start, 1),
    }


# ==============================================================================
# STANDALONE TEST
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(sys.argv) < 2:
        print("Usage: python assembly.py <script_id>")
        print("Example: python assembly.py script_20260513_080919")
        sys.exit(1)

    script_id = sys.argv[1]
    result = assemble_final(
        script_id,
        progress_cb=lambda msg: print(f"  · {msg}"),
    )

    print("\n✅ Assembly complete:")
    print(f"   Shots: {result['shot_count']}")
    print(f"   Duration: {result['total_duration_sec']:.2f}s")
    print(f"   9x16: {result['9x16']}")
    print(f"   16x9: {result['16x9']}")
    print(f"   1x1:  {result['1x1']}")
