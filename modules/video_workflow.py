"""
Claw Bot — Video Discord Workflow

Bridges clip_generator with Discord's approval pattern.
Reuses ✅/❌/⏹️/💾 from scripts and storyboards. Frees VRAM after each job.

Triggered automatically when a storyboard is approved. For each shot in the
storyboard manifest, runs TTS + video gen + ffmpeg mux, posts the resulting
narrated MP4 to the #videos channel, then posts a final approval prompt.
"""

import asyncio
import json
import logging
import sys
import time as _t
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import discord

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules.clip_generator import ClipGenerator
from modules import video_backend as vb
from modules import model_registry
from modules import gpu_utils
from modules import runtime_settings as rs
from modules import voice_casting as vc
from modules import generation_meta as gm
from modules import health_monitor as hm
from modules import prompt_approval as pap
from modules.file_utils import atomic_write_json

log = logging.getLogger("claw_bot.video_workflow")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
STORYBOARDS_DIR = PROJECT_ROOT / "04_Outputs" / "storyboards"
SCRIPTS_DIR = PROJECT_ROOT / "04_Outputs" / "scripts"
CLIPS_DIR = PROJECT_ROOT / "04_Outputs" / "clips"
APPROVED_VIDEOS_DIR = PROJECT_ROOT / "04_Outputs" / "approved_videos"


# ==============================================================================
# Helpers — same patterns as storyboard_workflow
# ==============================================================================

def load_script(script_id: str) -> dict:
    path = SCRIPTS_DIR / f"script_{script_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_storyboard_manifest(script_id: str) -> dict:
    path = STORYBOARDS_DIR / script_id / "storyboard.json"
    if not path.exists():
        raise FileNotFoundError(f"Storyboard manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_dur(sec: float) -> str:
    """Short mm ss duration for stage/shot timing display."""
    sec = int(round(max(0.0, sec)))
    m, s = divmod(sec, 60)
    return f"{m}m {s}s" if m else f"{s}s"


async def _free_vram_async():
    try:
        cleanup = await asyncio.to_thread(gpu_utils.cleanup_after_job, True, False)
        after = cleanup.get("after", {})
        free_gb = after.get("free_gb")
        if free_gb is not None:
            log.info(f"VRAM cleanup: free now {free_gb} GB")
    except Exception as e:
        log.warning(f"VRAM cleanup failed (non-fatal): {e}")


# ==============================================================================
# Main workflow
# ==============================================================================

async def generate_and_post_video(
    channel: discord.TextChannel,
    script_id: str,
    owner_id: int,
    requested_by_mention: str = "system",
    is_auto: bool = False,
) -> Optional[dict]:
    """
    For each shot in the approved storyboard, generate a narrated video clip
    and post it. Then post a final approval prompt.
    """
    # 1. Load script + manifest
    try:
        script = load_script(script_id)
        manifest = load_storyboard_manifest(script_id)
    except FileNotFoundError as e:
        await channel.send(f"❌ {e}")
        return None

    title = script.get("title", "Untitled")
    theme = script.get("theme", "")
    shots = script.get("shots", [])
    frames = manifest.get("frames", [])

    # Map shot_number -> image path (from manifest, which has correct paths)
    image_by_shot = {}
    for frame in frames:
        shot_num = frame.get("shot_number")
        img_path = frame.get("image_path")
        if shot_num and img_path:
            image_by_shot[shot_num] = PROJECT_ROOT / img_path

    if not image_by_shot:
        await channel.send(
            f"❌ Storyboard manifest for `{script_id}` has no frames. Cannot generate videos."
        )
        return None

    _job_start = _t.time()
    _vram_peak_mb = gpu_utils.get_vram_stats().get("used_mb", 0)

    # Get active backends
    video_cfg = model_registry.get_active("video_backend")
    video_backend_name = video_cfg.get("_id", "unknown")

    prefix = (
        "🎥 **Auto-generating video clips** (storyboard approved)"
        if is_auto
        else f"🎥 Video generation requested by {requested_by_mention}"
    )

    estimated_min = len(image_by_shot) * 2  # ~2 min per clip on LTX-2
    status_msg = await channel.send(
        f"{prefix}\n"
        f"Script: **{title}** (`{script_id}`)\n"
        f"Theme: _{theme}_\n"
        f"Video backend: `{video_backend_name}` · Voice: `{rs.get_effective_voice()}`\n"
        f"Shots: `{len(image_by_shot)}` · "
        f"Sync: `{rs.get_effective_sync_mode()}`\n"
        f"⏳ Starting... (~{estimated_min} minutes total)"
    )

    # Capture event loop on bot thread
    bot_loop = asyncio.get_running_loop()
    from modules.progress_bar import ProgressTracker
    tracker = ProgressTracker(
        total=len(image_by_shot),
        label="Generating clips",
        emoji="🎥",
    )
    progress_state = {"last_update": 0.0}

    def progress(text: str, current: int, total: int):
        now = datetime.now(timezone.utc).timestamp()
        if now - progress_state["last_update"] < 2.0 and current < total:
            return
        progress_state["last_update"] = now
        tracker.total = total  # in case it changes
        tracker.update(current)

        new_content = (
            f"{prefix}\n"
            f"Script: **{title}** (`{script_id}`)\n"
            f"Theme: _{theme}_\n"
            f"Video backend: `{video_backend_name}` · Voice: `{rs.get_effective_voice()}`\n"
            f"Sync: `{rs.get_effective_sync_mode()}`\n"
            f"{tracker.render(extra=text)}"
        )
        asyncio.run_coroutine_threadsafe(
            status_msg.edit(content=new_content), bot_loop
        )

    # 2. Build clip generator with current runtime settings
    try:
        clip_gen = ClipGenerator(
            sync_mode=rs.get_effective_sync_mode(),
        )
        # Override TTS voice if set
        clip_gen.tts.voice = rs.get_effective_voice()
    except Exception as e:
        log.exception("ClipGenerator init failed")
        await status_msg.edit(content=f"❌ ClipGenerator setup failed: `{e}`")
        return None

    # 3. Generate one clip per shot
    completed_clips = []
    failed_shots = []

    # One-time VRAM clear BEFORE the shot loop: evict BOTH the storyboard/image
    # model (ComfyUI /free) AND the LLM (Ollama keep_alive=0). The LLM is ~12GB;
    # if it stays resident the 12GB video model can't fit 16GB VRAM and ComfyUI
    # offloads layers to RAM — the 4min→15min slowdown. After this the video
    # model stays resident across shots (clip_generator no longer /free's per
    # shot), so shot 1 pays the cold load (~7-10min) and shots 2+ run cached.
    await _free_vram_async()
    try:
        await asyncio.to_thread(gpu_utils.free_ollama_vram)
    except Exception as e:
        log.warning(f"Pre-video Ollama unload failed (non-fatal): {e}")
    _vram = gpu_utils.get_vram_stats()
    log.info(f"Pre-video VRAM: {_vram.get('free_gb', '?')} GB free of {_vram.get('total_gb', '?')} GB")

    total_shots = len(image_by_shot)
    for idx, shot in enumerate(shots, start=1):
        shot_num = shot.get("shot_number")
        if shot_num not in image_by_shot:
            log.warning(f"Skipping shot {shot_num} — no storyboard image")
            continue

        narration = shot.get("narration", "").strip()
        # PREFERRED: user-approved motion prompt + seed from prompt_approval
        approved = pap.get_shot_prompts(script_id, shot_num) or {}
        approved_motion = (approved.get("motion_prompt") or "").strip()
        approved_motion_seed = approved.get("motion_seed", -1)
        if approved_motion:
            action = approved_motion
            log.info(f"Shot {shot_num}: using user-approved motion prompt")
        else:
            action = (shot.get("motion_prompt", "").strip()
                      or shot.get("action", "").strip()
                      or shot.get("visual_description", "").strip()
                      or shot.get("description", "").strip()
                      or narration)
        # Defensive: append explicit no-speech instruction in case LLM lapsed
        action = (
            action
            + " No speech, no dialog, no mouth movement, no lip sync. "
            + "Characters do not speak. Narration is voiceover added separately."
        )
        image_path = image_by_shot[shot_num]

        if not narration:
            # Fall back to visual_description for breathing shots (LLM sometimes leaves narration empty)
            narration = shot.get("visual_description", "").strip()
            if narration:
                log.info(f"Shot {shot_num}: narration empty, using visual_description as fallback")
            else:
                log.warning(f"Shot {shot_num} has no narration AND no visual_description; skipping")
                failed_shots.append((shot_num, "no narration or visual_description"))
                continue

        tracker.mark_step_start()
        progress(f"Shot {shot_num}: TTS + video gen...", idx - 1, total_shots)

        # Per-shot retry budget (3 attempts with backoff)
        clip_path = None
        last_err = None
        beat = (shot.get("beat") or "").strip().lower()
        seed_arg = approved_motion_seed if isinstance(approved_motion_seed, int) and approved_motion_seed > 0 else None
        shot_t0 = _t.perf_counter()
        # Audio-first: no per-shot TTS. The whole narration is one master VO laid
        # over the timeline at assembly, so each shot is a SILENT clip sized to
        # exactly its segment window (shot["win_dur"]). Classic path does per-shot
        # TTS inside generate_clip.
        _audio_first = bool(script.get("_audio_first"))
        _win_dur = float(shot.get("win_dur") or 0.0)
        for attempt in range(1, 4):
            try:
                if _audio_first and _win_dur > 0:
                    clip_path = await asyncio.to_thread(
                        clip_gen.generate_silent_clip,
                        shot_id=f"{script_id}_shot{shot_num}",
                        motion_prompt=action,
                        storyboard_image=image_path,
                        target_dur=_win_dur,
                        output_filename=f"clip_{script_id}_shot{shot_num}.mp4",
                        seed=seed_arg,
                        beat=beat,
                    )
                else:
                    clip_path = await asyncio.to_thread(
                        clip_gen.generate_clip,
                        shot_id=f"{script_id}_shot{shot_num}",
                        narration=narration,
                        action_prompt=action,
                        storyboard_image=image_path,
                        output_filename=f"clip_{script_id}_shot{shot_num}.mp4",
                        seed=seed_arg,
                        beat=beat,
                        voice=vc.resolve_voice(script, shot.get("speaker")),
                        lipsync=vc.is_character_speaker(shot.get("speaker")),
                    )
                break  # success
            except Exception as e:
                last_err = e
                log.warning(
                    f"Shot {shot_num} clip attempt {attempt}/3 failed: "
                    f"{type(e).__name__}: {e}"
                )
                if attempt < 3:
                    # Do NOT /free ComfyUI between retries — that evicts the video
                    # model and forces a cold reload every attempt (the slow loop).
                    # Unload Ollama only; keep the video model warm so the retry is
                    # fast. (If the failure was OOM, clip_generator's pre-gen Ollama
                    # unload already reclaims the room.)
                    try:
                        await asyncio.to_thread(gpu_utils.free_ollama_vram)
                    except Exception:
                        pass
                    await asyncio.sleep(8 * attempt)  # 8s, 16s backoff
        if clip_path is None:
            log.exception(f"Clip gen failed for shot {shot_num} after 3 attempts")
            failed_shots.append((shot_num, f"after 3 retries: {str(last_err)[:180]}"))
            await channel.send(
                f"⚠️ Shot {shot_num} failed after 3 retries: "
                f"`{type(last_err).__name__}: {str(last_err)[:180]}`"
            )
            continue

        # Persist the seed used for this clip alongside the manifest entry
        try:
            seed_used = int(getattr(clip_gen, "last_seed", -1) or -1)
            seeds_path = CLIPS_DIR / f"clip_{script_id}_seeds.json"
            seeds_path.parent.mkdir(parents=True, exist_ok=True)
            existing = {}
            if seeds_path.exists():
                existing = json.loads(seeds_path.read_text(encoding="utf-8"))
            existing[str(shot_num)] = seed_used
            atomic_write_json(seeds_path, existing)
        except Exception as e:
            log.warning(f"Could not persist seed for shot {shot_num}: {e}")

        shot_secs = _t.perf_counter() - shot_t0
        tracker.mark_step_done()
        progress(f"Shot {shot_num} done in {_fmt_dur(shot_secs)}", idx, total_shots)
        log.info(f"Shot {shot_num} render time: {_fmt_dur(shot_secs)} ({shot_secs:.1f}s)")
        completed_clips.append((shot_num, narration, clip_path, shot_secs))

        # Track VRAM peak after each clip
        try:
            cur = gpu_utils.get_vram_stats().get("used_mb", 0)
            if cur > _vram_peak_mb:
                _vram_peak_mb = cur
        except Exception:
            pass

        # Post the clip
        if clip_path.stat().st_size > 24 * 1024 * 1024:
            # Discord free-tier limit ~25 MB; warn instead of crashing
            await channel.send(
                f"⚠️ Shot {shot_num} clip is too large to upload "
                f"({clip_path.stat().st_size/(1024*1024):.1f} MB). "
                f"Saved locally: `{clip_path}`"
            )
        else:
            # Truncate motion prompt for Discord (1024 char field cap)
            motion_text = (
                shot.get("motion_prompt", "").strip()
                or shot.get("action", "").strip()
                or shot.get("visual_description", "").strip()
                or "(no prompt)"
            )[:900]
            if len(motion_text) >= 900:
                motion_text += "..."

            from modules.embed_styles import themed_embed
            from modules.approval_buttons import ClipControlView
            embed = themed_embed("video", f"Shot {shot_num}", f"🎙️ *{narration[:200]}*")
            embed.add_field(name="🎥 Motion Prompt", value=f"```{motion_text}```", inline=False)
            embed.set_footer(
                text=f"{clip_path.stat().st_size/(1024*1024):.2f} MB · {video_backend_name} · ⏱ {_fmt_dur(shot_secs)}"
            )
            clip_view = ClipControlView(
                script_id=script_id, shot_number=shot_num, owner_id=owner_id,
            )
            try:
                await channel.send(
                    embed=embed,
                    file=discord.File(str(clip_path), filename=clip_path.name),
                    view=clip_view,
                )
            except discord.HTTPException as e:
                await channel.send(
                    f"⚠️ Could not upload Shot {shot_num} clip: `{e}`. "
                    f"Saved at `{clip_path}`."
                )

    # 4. Final summary + approval prompt
    if not completed_clips:
        await status_msg.edit(
            content=f"❌ No clips generated successfully for `{script_id}`. "
                    f"Failed: {len(failed_shots)}/{total_shots}"
        )
        await _free_vram_async()
        return None

    success_line = (
        f"✅ Generated {len(completed_clips)}/{total_shots} clips"
        if not failed_shots
        else f"⚠️ Generated {len(completed_clips)}/{total_shots} clips "
             f"({len(failed_shots)} failed)"
    )
    await status_msg.edit(
        content=f"{success_line} for **{title}** (`{script_id}`).\n"
                f"Posting summary below..."
    )

    stage_secs = _t.time() - _job_start
    summary_lines = [f"━━━━━━━━━━━━━━━━━━━━━",
                     f"**🎥 {title} — Video Clips Complete**",
                     f"⏱ **Total video stage:** {_fmt_dur(stage_secs)}"]
    for shot_num, narration, clip_path, shot_secs in completed_clips:
        size_mb = clip_path.stat().st_size / (1024 * 1024)
        summary_lines.append(f"• Shot {shot_num} — {size_mb:.2f} MB · ⏱ {_fmt_dur(shot_secs)}")
    if failed_shots:
        summary_lines.append("")
        summary_lines.append("**Failed shots:**")
        for shot_num, err in failed_shots:
            summary_lines.append(f"• Shot {shot_num} — `{err}`")
    await channel.send("\n".join(summary_lines))

    from modules import approval_buttons
    view = approval_buttons.VideoApprovalView(script_id=script_id, owner_id=owner_id)
    approval_msg = await channel.send(
        f"**Video clips for `{script_id}`** — awaiting your decision.",
        view=view,
    )

    # Log to generation_meta + refresh dashboard
    total_size_mb = sum(p.stat().st_size for _, _, p, _ in completed_clips) / (1024 * 1024)
    gm.record({
        "kind": "video",
        "script_id": script_id,
        "duration_sec": round(_t.time() - _job_start, 1),
        "vram_peak_mb": _vram_peak_mb,
        "settings": {
            "backend_id": video_backend_name,
            "voice": rs.get_effective_voice(),
            "sync_mode": rs.get_effective_sync_mode(),
            "shots": len(completed_clips),
            "failed_shots": len(failed_shots),
            "size_mb": round(total_size_mb, 2),
        },
        "success": len(failed_shots) == 0,
    })
    try:
        await hm.trigger_refresh()
    except Exception:
        pass

    await _free_vram_async()

    return {
        "approval_msg_id": approval_msg.id,
        "script_id": script_id,
        "owner_id": owner_id,
        "channel_id": channel.id,
        "completed_clips": [(n, str(p)) for n, _, p, _ in completed_clips],
        "failed_shots": failed_shots,
    }


# ==============================================================================
# Re-render specific shots (called when user rejects a video pass)
# ==============================================================================

async def regenerate_video_shots(
    channel: discord.TextChannel,
    script_id: str,
    shot_numbers: list[int],
    owner_id: int,
) -> Optional[dict]:
    """Re-generate specific shot clips after rejection."""
    try:
        script = load_script(script_id)
        manifest = load_storyboard_manifest(script_id)
    except FileNotFoundError as e:
        await channel.send(f"❌ {e}")
        return None

    title = script.get("title", "Untitled")
    image_by_shot = {
        f.get("shot_number"): PROJECT_ROOT / f.get("image_path")
        for f in manifest.get("frames", [])
        if f.get("shot_number") and f.get("image_path")
    }

    status_msg = await channel.send(
        f"🔄 Re-generating video clips for shots `{', '.join(map(str, shot_numbers))}` "
        f"of **{title}**...\n"
        f"⏳ (~{2 * len(shot_numbers)} minutes)"
    )

    clip_gen = ClipGenerator(sync_mode=rs.get_effective_sync_mode())
    clip_gen.tts.voice = rs.get_effective_voice()

    # One-time VRAM clear before re-gen: evict the image model (ComfyUI /free)
    # AND the LLM (Ollama) so the video model fits 16GB without RAM-offload.
    # clip_generator also unloads Ollama per shot, but free the image model here.
    await _free_vram_async()
    try:
        await asyncio.to_thread(gpu_utils.free_ollama_vram)
    except Exception as e:
        log.warning(f"Pre-regen Ollama unload failed (non-fatal): {e}")
    _vram = gpu_utils.get_vram_stats()
    log.info(f"Pre-regen VRAM: {_vram.get('free_gb', '?')} GB free of {_vram.get('total_gb', '?')} GB")

    completed = []
    for shot_num in shot_numbers:
        if shot_num not in image_by_shot:
            await channel.send(f"⚠️ Shot {shot_num} has no storyboard image; skipping.")
            continue
        shot = next(
            (s for s in script["shots"] if s.get("shot_number") == shot_num),
            None,
        )
        if not shot:
            await channel.send(f"⚠️ Shot {shot_num} not in script; skipping.")
            continue

        narration = shot.get("narration", "").strip()
        approved = pap.get_shot_prompts(script_id, shot_num) or {}
        approved_motion = (approved.get("motion_prompt") or "").strip()
        approved_motion_seed = approved.get("motion_seed", -1)
        if approved_motion:
            action = approved_motion
        else:
            action = (shot.get("motion_prompt", "").strip()
                      or shot.get("action", "").strip()
                      or shot.get("visual_description", "").strip()
                      or narration)
        action = (
            action
            + " No speech, no dialog, no mouth movement, no lip sync. "
            + "Characters do not speak. Narration is voiceover added separately."
        )
        clip_path = None
        last_err = None
        beat = (shot.get("beat") or "").strip().lower()
        seed_arg = approved_motion_seed if isinstance(approved_motion_seed, int) and approved_motion_seed > 0 else None
        for attempt in range(1, 4):
            try:
                clip_path = await asyncio.to_thread(
                    clip_gen.generate_clip,
                    shot_id=f"{script_id}_shot{shot_num}_v2",
                    narration=narration,
                    action_prompt=action,
                    storyboard_image=image_by_shot[shot_num],
                    output_filename=f"clip_{script_id}_shot{shot_num}_v2.mp4",
                    seed=seed_arg,
                    beat=beat,
                    voice=vc.resolve_voice(script, shot.get("speaker")),
                    lipsync=vc.is_character_speaker(shot.get("speaker")),
                )
                break
            except Exception as e:
                last_err = e
                log.warning(f"Shot {shot_num} re-gen attempt {attempt}/3 failed: {e}")
                if attempt < 3:
                    await _free_vram_async()
                    await asyncio.sleep(8 * attempt)
        if clip_path is None:
            log.exception(f"Re-gen failed for shot {shot_num} after 3 attempts")
            await channel.send(f"⚠️ Shot {shot_num} re-gen failed after 3 retries: `{last_err}`")
            continue
        completed.append((shot_num, narration, clip_path))

    if not completed:
        await status_msg.edit(content="❌ No shots re-generated successfully.")
        await _free_vram_async()
        return None

    await status_msg.edit(content=f"✅ Re-rendered {len(completed)} clip(s).")
    await channel.send("━━━━━━━━━━━━━━━━━━━━━\n**Updated clips:**")
    for shot_num, narration, clip_path in completed:
        size_mb = clip_path.stat().st_size / (1024 * 1024)
        if size_mb > 24:
            await channel.send(f"• Shot {shot_num} — too large ({size_mb:.1f} MB), saved locally")
            continue
            
        # Re-fetch the shot info to get the prompt
        shot = next((s for s in script["shots"] if s.get("shot_number") == shot_num), {})
        
        # Truncate motion prompt for Discord (1024 char field cap)
        motion_text = (
            shot.get("motion_prompt", "").strip()
            or shot.get("action", "").strip()
            or shot.get("visual_description", "").strip()
            or "(no prompt)"
        )[:900]
        if len(motion_text) >= 900:
            motion_text += "..."

        from modules.embed_styles import themed_embed
        from modules.approval_buttons import ClipControlView
        embed = themed_embed("video", f"Shot {shot_num} (revised)", f"🎙️ *{narration[:200]}*")
        embed.color = discord.Color.gold()  # distinguish revisions
        embed.add_field(name="🎥 Motion Prompt", value=f"```{motion_text}```", inline=False)
        embed.set_footer(text=f"{size_mb:.2f} MB")
        clip_view = ClipControlView(
            script_id=script_id, shot_number=shot_num, owner_id=owner_id,
        )
        try:
            await channel.send(
                embed=embed,
                file=discord.File(str(clip_path), filename=clip_path.name),
                view=clip_view,
            )
        except discord.HTTPException as e:
            await channel.send(f"⚠️ Could not upload Shot {shot_num}: `{e}`")

    from modules import approval_buttons
    view = approval_buttons.VideoApprovalView(script_id=script_id, owner_id=owner_id)
    approval_msg = await channel.send(
        f"**Video clips for `{script_id}`** — updated, awaiting decision.",
        view=view,
    )

    await _free_vram_async()

    return {
        "approval_msg_id": approval_msg.id,
        "script_id": script_id,
        "owner_id": owner_id,
        "channel_id": channel.id,
    }