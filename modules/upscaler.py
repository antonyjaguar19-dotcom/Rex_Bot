"""
Claw Bot — Video Upscaler (Phase 6.5)

Per-shot upscale via ComfyUI + Real-ESRGAN (realesr-animevideov3).
Pipeline trigger: after the entire storyboard's videos are user-approved.
Per-shot, sequential, replaces 480p originals with 4x upscaled versions.
Originals are NOT archived (per user choice — saves disk).

Audio: ComfyUI's video pipeline strips audio. We re-mux the original
audio back onto the upscaled output via ffmpeg.

Pattern mirrors comfyui_zimage_base.py:
  - POST workflow JSON to /prompt
  - Poll /history/{prompt_id} until done
  - Download result from ComfyUI's output folder
  - Move to final destination, delete original
"""

import json
import logging
import shutil
import subprocess
import sys
import time as _t
from pathlib import Path
from typing import Optional, Callable

import requests

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import gpu_utils  # noqa

log = logging.getLogger("claw_bot.upscaler")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
VIDEOS_DIR = PROJECT_ROOT / "04_Outputs" / "clips"
WORKFLOWS_DIR = PROJECT_ROOT / "05_Config" / "workflows"
COMFY_ROOT = PROJECT_ROOT / "01_ComfyUI" / "ComfyUI_windows_portable" / "ComfyUI"
COMFY_INPUT = COMFY_ROOT / "input"
COMFY_OUTPUT = COMFY_ROOT / "output"

COMFY_URL = "http://127.0.0.1:8188"
WORKFLOW_FILE = WORKFLOWS_DIR / "upscale_video.json"

FFMPEG_EXE = PROJECT_ROOT / "00_Tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE_EXE = PROJECT_ROOT / "00_Tools" / "ffmpeg" / "bin" / "ffprobe.exe"

POLL_INTERVAL_SEC = 2.0
JOB_TIMEOUT_SEC = 600  # 10 min per clip — generous

# --- Polish pass ----------------------------------------------------------
# The Real-ESRGAN anime model 4x's to ~1920px, but that raw output looks TOO
# sharp and the fine detail (fur, spots) "morphs"/crawls frame-to-frame — the
# source Wan clip already wobbles a little and 4x sharpening magnifies it.
# Two cheap ffmpeg steps fix it:
#   1. Supersample DOWN to a 1080 short side (lanczos) — averages the amplified
#      high-frequency shimmer and lands on the reel's actual delivery size.
#   2. hqdn3d with a strong TEMPORAL term — averages the residual crawl across
#      frames while leaving motion intact. Spatial term kept low to keep detail.
# Verified on a mascot clip: raw 4x = clay/morphing; this = clean.
UPSCALE_TARGET_SHORT = 1080
UPSCALE_DENOISE = "2:1.5:10:10"   # luma_spatial:chroma_spatial:luma_tmp:chroma_tmp


def _polish_clip(src: Path, dst: Path) -> Path:
    """Downscale to a 1080 short side + temporal denoise. Preserves audio.

    Short side (not a fixed WxH) so it works for 9x16, 16x9 and 1x1 alike.
    """
    # Scale the SHORTER dimension to UPSCALE_TARGET_SHORT, keep aspect, even dims.
    t = UPSCALE_TARGET_SHORT
    scale = (f"scale='if(lt(iw,ih),{t},-2)':'if(lt(iw,ih),-2,{t})'"
             f":flags=lanczos")
    vf = f"{scale},hqdn3d={UPSCALE_DENOISE}"
    cmd = [str(FFMPEG_EXE), "-y", "-loglevel", "error", "-i", str(src),
           "-vf", vf, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "copy", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if not dst.exists() or r.returncode != 0:
        log.warning(f"polish failed ({r.stderr[:200]}); keeping raw upscale")
        shutil.copy2(src, dst)
    return dst


# ==============================================================================
# COMFY API HELPERS
# ==============================================================================

def _comfy_alive() -> bool:
    try:
        r = requests.get(f"{COMFY_URL}/system_stats", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _post_workflow(workflow: dict) -> str:
    """POST workflow, return prompt_id."""
    r = requests.post(f"{COMFY_URL}/prompt", json={"prompt": workflow}, timeout=30)
    r.raise_for_status()
    data = r.json()
    pid = data.get("prompt_id")
    if not pid:
        raise RuntimeError(f"ComfyUI did not return a prompt_id: {data}")
    return pid


def _poll_until_done(prompt_id: str, timeout: float, progress_cb: Optional[Callable] = None) -> dict:
    """Poll /history/{prompt_id} until result lands. Return the history entry."""
    start = _t.time()
    last_progress = 0.0
    while _t.time() - start < timeout:
        try:
            r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10)
            if r.status_code == 200:
                data = r.json()
                if prompt_id in data:
                    return data[prompt_id]
        except Exception as e:
            log.debug(f"Poll error (will retry): {e}")

        if progress_cb and (_t.time() - last_progress) > 5.0:
            elapsed = int(_t.time() - start)
            progress_cb(f"upscaling... {elapsed}s elapsed")
            last_progress = _t.time()

        _t.sleep(POLL_INTERVAL_SEC)
    raise TimeoutError(f"Upscale job {prompt_id} did not finish in {timeout}s")


def _extract_output_video(history_entry: dict) -> Path:
    """Find the output video file path from history."""
    outputs = history_entry.get("outputs", {})
    for node_id, node_out in outputs.items():
        # VHS_VideoCombine puts results under 'gifs' (yes, even MP4s — that's the node)
        for key in ("gifs", "videos", "images"):
            if key in node_out:
                for item in node_out[key]:
                    fn = item.get("filename")
                    sub = item.get("subfolder", "")
                    folder_type = item.get("type", "output")
                    if fn:
                        base = COMFY_OUTPUT if folder_type == "output" else COMFY_INPUT
                        candidate = base / sub / fn
                        if candidate.exists():
                            return candidate
    raise RuntimeError(f"No output file found in history: {history_entry}")


# ==============================================================================
# AUDIO HELPERS (NEW)
# ==============================================================================

def _has_audio_stream(path: Path) -> bool:
    """Return True if the file has at least one audio stream."""
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


def _probe_fps(path: Path, fallback: float = 16.0) -> float:
    """Return the source video's frame rate (e.g. 16.0). Falls back if unknown.
    The upscale output MUST be combined at this same rate, otherwise the same
    frames replayed at a different fps change the clip's DURATION (e.g. 16fps
    frames replayed at 24fps run 0.667x short → narration gets cropped)."""
    cmd = [
        str(FFPROBE_EXE),
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0",
        str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    raw = r.stdout.strip()
    try:
        if "/" in raw:
            num, den = raw.split("/")
            return float(num) / float(den) if float(den) else fallback
        return float(raw) if raw else fallback
    except (ValueError, ZeroDivisionError):
        return fallback


def _reattach_audio(silent_video: Path, audio_source: Path, output_path: Path) -> None:
    """Copy audio from audio_source onto silent_video, write to output_path.

    Narration is the source of truth — never crop it. We do NOT use `-shortest`
    here: if the upscaled video came out a hair shorter, -shortest would clip the
    last word. Mapping the full audio keeps narration intact (the matching fps
    fix means the streams already line up; this is just a safety net)."""
    cmd = [
        str(FFMPEG_EXE),
        "-y",
        "-loglevel", "error",
        "-i", str(silent_video),
        "-i", str(audio_source),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio re-mux failed:\n{result.stderr.strip()}"
        )


# ==============================================================================
# WORKFLOW BUILDER
# ==============================================================================

def _build_workflow(input_video_relative_path: str, source_fps: float = 16.0) -> dict:
    """Load template, inject the input video path + match the output frame rate
    to the SOURCE so upscaling never changes the clip's duration."""
    if not WORKFLOW_FILE.exists():
        raise FileNotFoundError(f"Workflow template missing: {WORKFLOW_FILE}")
    template = json.loads(WORKFLOW_FILE.read_text(encoding="utf-8"))
    # VHS_LoadVideo expects a path RELATIVE to ComfyUI's input/ folder
    template["1"]["inputs"]["video"] = input_video_relative_path
    # VHS_VideoCombine must output at the SAME fps the frames came in at.
    # Hardcoded 24 here against 16fps sources shrank duration 0.667x and the
    # audio re-mux cropped narration to match. Round to int (VHS wants int/float).
    for node in template.values():
        if node.get("class_type") == "VHS_VideoCombine":
            node["inputs"]["frame_rate"] = round(source_fps)
    return template


# ==============================================================================
# SINGLE-SHOT UPSCALE
# ==============================================================================

def upscale_clip(video_path: Path, progress_cb: Optional[Callable] = None) -> Path:
    """Upscale one video file, replace original. Preserves audio."""
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if not _comfy_alive():
        raise RuntimeError(f"ComfyUI not reachable at {COMFY_URL} — is it running?")
    if not FFMPEG_EXE.exists():
        raise FileNotFoundError(f"ffmpeg not found at {FFMPEG_EXE}")

    # Check audio presence + source fps BEFORE we touch the file
    had_audio = _has_audio_stream(video_path)
    source_fps = _probe_fps(video_path)
    log.info(
        f"Source audio: {'present' if had_audio else 'none'}, "
        f"fps={source_fps:.3f} — {video_path.name}"
    )

    # Stage input into ComfyUI's input folder
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    staged_name = f"upscale_in_{int(_t.time() * 1000)}_{video_path.name}"
    staged_path = COMFY_INPUT / staged_name
    shutil.copy2(video_path, staged_path)

    try:
        if progress_cb:
            progress_cb(f"queued {video_path.name}")

        workflow = _build_workflow(staged_name, source_fps=source_fps)
        pid = _post_workflow(workflow)
        log.info(f"Upscale queued: prompt_id={pid} for {video_path.name}")

        history = _poll_until_done(pid, JOB_TIMEOUT_SEC, progress_cb)
        output_video = _extract_output_video(history)
        log.info(f"Upscale complete (silent): {output_video}")

        # If source had audio, re-attach it to the upscaled (silent) output
        if had_audio:
            if progress_cb:
                progress_cb(f"re-attaching audio for {video_path.name}")
            remuxed = output_video.with_suffix(".withaudio.mp4")
            _reattach_audio(
                silent_video=output_video,
                audio_source=video_path,
                output_path=remuxed,
            )
            output_video.unlink(missing_ok=True)
            output_video = remuxed
            log.info(f"Audio re-attached: {output_video}")

        # Polish: downscale to the 1080 delivery size + temporal denoise, which
        # removes the over-sharp "clay"/morphing look of the raw 4x output.
        if progress_cb:
            progress_cb(f"polishing {video_path.name}")
        polished = output_video.with_suffix(".polished.mp4")
        _polish_clip(output_video, polished)
        output_video.unlink(missing_ok=True)
        output_video = polished
        log.info(f"Polished (1080 + denoise): {output_video}")

        # Replace original (per user's "save disk" choice)
        tmp_replacement = video_path.with_suffix(video_path.suffix + ".upscaled")
        shutil.move(str(output_video), str(tmp_replacement))
        video_path.unlink()  # delete original
        tmp_replacement.rename(video_path)

        return video_path
    finally:
        # Always clean up staged input
        try:
            staged_path.unlink(missing_ok=True)
        except Exception as e:
            log.warning(f"Could not clean staged input {staged_path}: {e}")


# ==============================================================================
# BATCH UPSCALE — all shots of a script_id
# ==============================================================================

def upscale_storyboard_videos(
    script_id: str,
    progress_cb: Optional[Callable] = None,
) -> dict:
    """Upscale every shot's video for a given script_id, sequentially.

    Clips live flat in 04_Outputs/clips/ as: clip_<script_id>_shot<N>[_v<rev>].mp4
    For shots with multiple revisions, only the latest version is upscaled.

    Returns: {script_id, total_shots, succeeded: [shot_nums], failed: [(shot, err)], duration_sec}
    """
    if not VIDEOS_DIR.exists():
        raise FileNotFoundError(f"Clips folder not found: {VIDEOS_DIR}")

    all_clips = sorted(VIDEOS_DIR.glob(f"clip_{script_id}_shot*.mp4"))
    if not all_clips:
        raise FileNotFoundError(f"No clips found for script_id={script_id} in {VIDEOS_DIR}")

    # Pick the LATEST version per shot (shot1.mp4 vs shot1_v2.mp4 -> v2 wins)
    import re
    by_shot: dict[int, tuple[int, Path]] = {}
    for p in all_clips:
        m = re.match(r"clip_.+_shot(\d+)(?:_v(\d+))?$", p.stem)
        if not m:
            log.warning(f"Skipping unparseable clip: {p.name}")
            continue
        shot_n = int(m.group(1))
        ver = int(m.group(2)) if m.group(2) else 1
        if shot_n not in by_shot or ver > by_shot[shot_n][0]:
            by_shot[shot_n] = (ver, p)

    clips = [by_shot[k][1] for k in sorted(by_shot.keys())]
    if not clips:
        raise FileNotFoundError(f"No parseable clips for script_id={script_id}")

    log.info(f"Upscale batch: {len(clips)} clips for script_id={script_id}")
    start = _t.time()
    succeeded: list[int] = []
    failed: list[tuple[int, str]] = []

    for idx, clip in enumerate(clips, start=1):
        m = re.match(r"clip_.+_shot(\d+)", clip.stem)
        shot_num = int(m.group(1)) if m else idx

        if progress_cb:
            progress_cb(f"[{idx}/{len(clips)}] upscaling shot {shot_num} ({clip.name})...")

        try:
            gpu_utils.cleanup_after_job(free_comfy=False, free_ollama=True)
        except Exception:
            pass

        try:
            upscale_clip(clip, progress_cb=None)
            succeeded.append(shot_num)
            log.info(f"Shot {shot_num} upscaled successfully")
        except Exception as e:
            log.exception(f"Shot {shot_num} upscale failed")
            failed.append((shot_num, str(e)))
            if progress_cb:
                progress_cb(f"⚠️ shot {shot_num} failed: {e}")

    duration = round(_t.time() - start, 1)
    log.info(f"Upscale batch done: {len(succeeded)} ok, {len(failed)} failed, {duration}s")

    return {
        "script_id": script_id,
        "total_shots": len(clips),
        "succeeded": succeeded,
        "failed": failed,
        "duration_sec": duration,
    }


# ==============================================================================
# DISCORD-FRIENDLY SUMMARY
# ==============================================================================

def format_upscale_summary(result: dict) -> str:
    total = result.get("total_shots", 0)
    ok = len(result.get("succeeded", []))
    failed = result.get("failed", [])
    dur = result.get("duration_sec", 0)
    mins = dur / 60

    lines = [f"📈 **Upscale complete:** {ok}/{total} shots in {mins:.1f} min"]
    if failed:
        lines.append("")
        lines.append("⚠️ Failed:")
        for sn, err in failed:
            lines.append(f"  • Shot {sn}: `{err[:120]}`")
    return "\n".join(lines)


# ==============================================================================
# STANDALONE TEST
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python upscaler.py <script_id>")
        sys.exit(1)
    sid = sys.argv[1]

    def cb(msg):
        print(f"  [progress] {msg}")

    result = upscale_storyboard_videos(sid, progress_cb=cb)
    print("\n" + format_upscale_summary(result))
