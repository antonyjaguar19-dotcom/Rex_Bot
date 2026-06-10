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
from modules import generation_meta as gm
from modules import health_monitor as hm

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

    total_shots = len(image_by_shot)
    for idx, shot in enumerate(shots, start=1):
        shot_num = shot.get("shot_number")
        if shot_num not in image_by_shot:
            log.warning(f"Skipping shot {shot_num} — no storyboard image")
            continue

        narration = shot.get("narration", "").strip()
        # Prefer motion_prompt (new schema), fallback to action/description, last resort narration
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

        progress(f"Shot {shot_num}: TTS + video gen...", idx, total_shots)

        try:
            clip_path = await asyncio.to_thread(
                clip_gen.generate_clip,
                shot_id=f"{script_id}_shot{shot_num}",
                narration=narration,
                action_prompt=action,
                storyboard_image=image_path,
                output_filename=f"clip_{script_id}_shot{shot_num}.mp4",
            )
        except Exception as e:
            log.exception(f"Clip gen failed for shot {shot_num}")
            failed_shots.append((shot_num, str(e)[:200]))
            await channel.send(
                f"⚠️ Shot {shot_num} failed: `{type(e).__name__}: {str(e)[:200]}`"
            )
            continue

        completed_clips.append((shot_num, narration, clip_path))

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
            embed = themed_embed("video", f"Shot {shot_num}", f"🎙️ *{narration[:200]}*")
            embed.add_field(name="🎥 Motion Prompt", value=f"```{motion_text}```", inline=False)
            embed.set_footer(
                text=f"{clip_path.stat().st_size/(1024*1024):.2f} MB · {video_backend_name}"
            )
            try:
                await channel.send(
                    embed=embed,
                    file=discord.File(str(clip_path), filename=clip_path.name),
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

    summary_lines = [f"━━━━━━━━━━━━━━━━━━━━━",
                     f"**🎥 {title} — Video Clips Complete**"]
    for shot_num, narration, clip_path in completed_clips:
        size_mb = clip_path.stat().st_size / (1024 * 1024)
        summary_lines.append(f"• Shot {shot_num} — {size_mb:.2f} MB")
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
    total_size_mb = sum(p.stat().st_size for _, _, p in completed_clips) / (1024 * 1024)
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
        "completed_clips": [(n, str(p)) for n, _, p in completed_clips],
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
        action = (shot.get("motion_prompt", "").strip()
                  or shot.get("action", "").strip()
                  or shot.get("visual_description", "").strip()
                  or narration)
        action = (
            action
            + " No speech, no dialog, no mouth movement, no lip sync. "
            + "Characters do not speak. Narration is voiceover added separately."
        )
        try:
            clip_path = await asyncio.to_thread(
                clip_gen.generate_clip,
                shot_id=f"{script_id}_shot{shot_num}_v2",
                narration=narration,
                action_prompt=action,
                storyboard_image=image_by_shot[shot_num],
                output_filename=f"clip_{script_id}_shot{shot_num}_v2.mp4",
            )
        except Exception as e:
            log.exception(f"Re-gen failed for shot {shot_num}")
            await channel.send(f"⚠️ Shot {shot_num} re-gen failed: `{e}`")
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
        embed = themed_embed("video", f"Shot {shot_num} (revised)", f"🎙️ *{narration[:200]}*")
        embed.color = discord.Color.gold()  # distinguish revisions
        embed.add_field(name="🎥 Motion Prompt", value=f"```{motion_text}```", inline=False)
        embed.set_footer(text=f"{size_mb:.2f} MB")
        
        try:
            await channel.send(
                embed=embed,
                file=discord.File(str(clip_path), filename=clip_path.name),
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