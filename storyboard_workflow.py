"""
Claw Bot — Storyboard Discord Workflow

Bridges storyboard_generator with Discord's approval/revision pattern.
Reuses ✅/❌/⏹️/💾 from scripts. Frees VRAM after each job.
"""

import asyncio
import json
import logging
import re
import sys
import time as _t
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import discord

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import storyboard_generator as sg
from modules import image_backend as ib
from modules import model_registry
from modules import gpu_utils
from modules import generation_meta as gm
from modules import health_monitor as hm
from modules import runtime_settings as rs

log = logging.getLogger("claw_bot.storyboard_workflow")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
STORYBOARDS_DIR = PROJECT_ROOT / "04_Outputs" / "storyboards"
SCRIPTS_DIR = PROJECT_ROOT / "04_Outputs" / "scripts"
APPROVED_STORYBOARDS_DIR = PROJECT_ROOT / "04_Outputs" / "approved_storyboards"


def load_script(script_id: str) -> dict:
    """Load a script JSON by ID."""
    path = SCRIPTS_DIR / f"script_{script_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


async def _free_vram_async():
    """Call the blocking cleanup on a worker thread; log result."""
    try:
        cleanup = await asyncio.to_thread(gpu_utils.cleanup_after_job, True, False)
        after = cleanup.get("after", {})
        free_gb = after.get("free_gb")
        if free_gb is not None:
            log.info(f"VRAM cleanup: free now {free_gb} GB")
    except Exception as e:
        log.warning(f"VRAM cleanup failed (non-fatal): {e}")


async def generate_and_post_storyboard(
    channel: discord.TextChannel,
    script_id: str,
    owner_id: int,
    requested_by_mention: str = "system",
    is_auto: bool = False,
) -> Optional[dict]:
    """
    Full workflow: generate 4 frames, post each, then post approval prompt.
    """
    try:
        script = load_script(script_id)
    except FileNotFoundError as e:
        await channel.send(f"❌ {e}")
        return None

    title = script.get("title", "Untitled")
    theme = script.get("theme", "")
    backend_cfg = model_registry.get_active("image_backend")
    backend_name = backend_cfg.get("_id", "unknown")

    _job_start = _t.time()

    prefix = (
        "🎬 **Auto-storyboarding** (script approved)"
        if is_auto
        else f"🎬 Storyboard requested by {requested_by_mention}"
    )
    status_msg = await channel.send(
        f"{prefix}\n"
        f"Script: **{title}** (`{script_id}`)\n"
        f"Theme: _{theme}_\n"
        f"Backend: `{backend_name}`\n"
        f"⏳ Starting... (~2 minutes for 4 frames)"
    )

    # Capture the main event loop now, while still on the bot's thread.
    bot_loop = asyncio.get_running_loop()
    from modules.progress_bar import ProgressTracker
    tracker = ProgressTracker(
        total=len(script.get("shots", [])),
        label="Rendering frames",
        emoji="🎨",
    )
    progress_state = {"last_update": 0.0}

    def progress(text: str, current: int, total: int):
        """Runs in generator worker thread. Schedule Discord update on bot loop."""
        now = datetime.now(timezone.utc).timestamp()
        if now - progress_state["last_update"] < 2.0 and current < total:
            return
        progress_state["last_update"] = now
        tracker.total = total
        tracker.update(current)

        new_content = (
            f"{prefix}\n"
            f"Script: **{title}** (`{script_id}`)\n"
            f"Theme: _{theme}_\n"
            f"Backend: `{backend_name}`\n"
            f"{tracker.render(extra=text)}"
        )
        asyncio.run_coroutine_threadsafe(
            status_msg.edit(content=new_content), bot_loop
        )

    try:
        result = await asyncio.to_thread(
            sg.generate_storyboard,
            script_id=script_id,
            aspect_ratio=model_registry.get_image_defaults().get("aspect_ratio", "16:9"),
            progress_callback=progress,
        )
    except Exception as e:
        log.exception("Storyboard generation failed")
        await status_msg.edit(
            content=f"❌ Storyboard generation failed: `{e}`\n"
                    f"Check the bot's console for details."
        )
        await _free_vram_async()
        return None

    if not result.success:
        await status_msg.edit(
            content=f"❌ Storyboard generation reported failure: `{result.error}`"
        )
        await _free_vram_async()
        return None

    await status_msg.edit(
        content=f"✅ Storyboard generated for **{title}** (`{script_id}`).\n"
                f"Posting frames below..."
    )

    await channel.send(f"━━━━━━━━━━━━━━━━━━━━━\n**📖 {title} — Storyboard**")

    for frame in result.frames:
        shot = next(
            (s for s in script["shots"] if s.get("shot_number") == frame.shot_number),
            {},
        )
        narration = shot.get("narration", "")
        abs_path = PROJECT_ROOT / frame.image_path
        if not abs_path.exists():
            await channel.send(f"⚠️ Missing file: `{frame.image_path}`")
            continue

        # Truncate prompt for embed (Discord field max 1024 chars)
        prompt_text = (frame.formatted_prompt or frame.prompt or "")[:1000]
        if len(prompt_text) >= 1000:
            prompt_text += "..."

        from modules.embed_styles import themed_embed
        from modules.approval_buttons import ShotControlView
        embed = themed_embed("shot", f"Shot {frame.shot_number}", f"🎙️ *{narration}*")
        if prompt_text:
            embed.add_field(name="🎨 Image Prompt", value=f"```{prompt_text}```", inline=False)
        embed.set_footer(text=f"seed={frame.seed} · {frame.backend_id}")
        discord_file = discord.File(str(abs_path), filename=f"shot{frame.shot_number}.png")
        embed.set_image(url=f"attachment://shot{frame.shot_number}.png")
        shot_view = ShotControlView(
            script_id=script_id,
            shot_number=frame.shot_number,
            owner_id=owner_id,
        )
        await channel.send(embed=embed, file=discord_file, view=shot_view)

    await _free_vram_async()
    # ── End QA + Continuity pass ──

    from modules import approval_buttons
    view = approval_buttons.StoryboardApprovalView(script_id=script_id, owner_id=owner_id)
    approval_msg = await channel.send(
        f"**Storyboard `{script_id}`** — awaiting your decision.",
        view=view,
    )

    # Log to generation_meta + refresh dashboard
    try:
        eff_style = rs.get_effective_style()
    except Exception:
        eff_style = None
    try:
        eff_aspect = rs.get_effective_aspect_ratio()
    except Exception:
        eff_aspect = None
    gm.record({
        "kind": "storyboard",
        "script_id": script_id,
        "duration_sec": round(_t.time() - _job_start, 1),
        "vram_peak_mb": gpu_utils.get_vram_stats().get("used_mb", 0),
        "settings": {
            "backend_id": backend_name,
            "frames_count": len(getattr(result, "frames", [])),
            "style": eff_style,
            "aspect_ratio": eff_aspect,
        },
        "success": True,
    })
    try:
        await hm.trigger_refresh()
    except Exception:
        pass

    # Free VRAM after the storyboard is done
    await _free_vram_async()

    return {
        "approval_msg_id": approval_msg.id,
        "script_id": script_id,
        "owner_id": owner_id,
        "channel_id": channel.id,
        "result": result,
    }


async def regenerate_shots(
    channel: discord.TextChannel,
    script_id: str,
    shot_numbers: list[int],
    owner_id: int,
    post_approval: bool = True,
) -> Optional[dict]:
    """Re-render specific shots of an existing storyboard."""
    try:
        script = load_script(script_id)
    except FileNotFoundError as e:
        await channel.send(f"❌ {e}")
        return None

    title = script.get("title", "Untitled")
    status_msg = await channel.send(
        f"🔄 Re-rendering shots `{', '.join(map(str, shot_numbers))}` of **{title}**...\n"
        f"⏳ (~{30 * len(shot_numbers)} seconds)"
    )

    all_new_frames = []
    try:
        for shot_num in shot_numbers:
            new_frames = await asyncio.to_thread(
                sg.regenerate_shot,
                script_id=script_id,
                shot_number=shot_num,
                which_frame="first",
            )
            all_new_frames.extend(new_frames)
    except Exception as e:
        log.exception("Regen failed")
        await status_msg.edit(content=f"❌ Regeneration failed: `{e}`")
        await _free_vram_async()
        return None

    await status_msg.edit(content=f"✅ Re-rendered {len(all_new_frames)} frame(s).")

    await channel.send(f"━━━━━━━━━━━━━━━━━━━━━\n**Updated frames:**")
    for frame in all_new_frames:
        shot = next(
            (s for s in script["shots"] if s.get("shot_number") == frame.shot_number),
            {},
        )
        narration = shot.get("narration", "")
        abs_path = PROJECT_ROOT / frame.image_path
        if not abs_path.exists():
            continue
        # Truncate prompt for embed (Discord field max 1024 chars)
        prompt_text = (frame.formatted_prompt or frame.prompt or "")[:1000]
        if len(prompt_text) >= 1000:
            prompt_text += "..."

        from modules.embed_styles import themed_embed
        from modules.approval_buttons import ShotControlView
        embed = themed_embed("shot", f"Shot {frame.shot_number} (revised)", f"🎙️ *{narration}*")
        embed.color = discord.Color.orange()  # mark revised distinctly
        discord_file = discord.File(str(abs_path), filename=f"shot{frame.shot_number}_revised.png")
        embed.set_image(url=f"attachment://shot{frame.shot_number}_revised.png")
        shot_view = ShotControlView(
            script_id=script_id,
            shot_number=frame.shot_number,
            owner_id=owner_id,
        )
        await channel.send(embed=embed, file=discord_file, view=shot_view)

    if not post_approval:
        await _free_vram_async()
        return None

    from modules import approval_buttons
    view = approval_buttons.StoryboardApprovalView(script_id=script_id, owner_id=owner_id)
    approval_msg = await channel.send(
        f"**Storyboard `{script_id}`** — updated, awaiting decision.",
        view=view,
    )

    await _free_vram_async()

    return {
        "approval_msg_id": approval_msg.id,
        "script_id": script_id,
        "owner_id": owner_id,
        "channel_id": channel.id,
    }


def extract_shot_numbers(notes: str, total_shots: int = None) -> list[int]:
    """Parse revision notes looking for shot references.

    Shot counts are story-driven (organic), so no hardcoded ceiling.
    Pass `total_shots` to resolve 'last'/'final' to the real final shot;
    without it, those words are ignored.
    """
    nums = set()
    text = notes.lower()

    # Match "shot 5", "shot #5", "shot 12" — multi-digit safe, finds ALL occurrences
    for m in re.finditer(r"shot\s*#?\s*(\d+)", text):
        nums.add(int(m.group(1)))

    # Same for "scene" and "frame"
    for m in re.finditer(r"scene\s*#?\s*(\d+)", text):
        nums.add(int(m.group(1)))
    for m in re.finditer(r"frame\s*#?\s*(\d+)", text):
        nums.add(int(m.group(1)))

    # Ordinal words
    word_map = {
        "first": 1, "1st": 1,
        "second": 2, "2nd": 2,
        "third": 3, "3rd": 3,
        "fourth": 4, "4th": 4,
        "fifth": 5, "5th": 5,
        "sixth": 6, "6th": 6,
        "seventh": 7, "7th": 7,
        "eighth": 8, "8th": 8,
        "ninth": 9, "9th": 9,
        "tenth": 10, "10th": 10,
    }
    for word, n in word_map.items():
        if re.search(rf"\b{word}\s+(shot|scene|frame)", text):
            nums.add(n)

    # "last" / "final" — only resolvable if we know total shot count
    if total_shots and re.search(r"\b(last|final)\s+(shot|scene|frame)", text):
        nums.add(total_shots)

    # Filter to valid range if total_shots known; else keep as-is and let caller validate
    if total_shots:
        nums = {n for n in nums if 1 <= n <= total_shots}

    return sorted(nums)


def list_recent_storyboards(limit: int = 10) -> list[dict]:
    """Return metadata for recent storyboard manifests."""
    if not STORYBOARDS_DIR.exists():
        return []
    manifests = sorted(
        STORYBOARDS_DIR.glob("*/storyboard.json"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )[:limit]
    results = []
    for m in manifests:
        try:
            data = json.loads(m.read_text(encoding="utf-8"))
            results.append({
                "script_id": data.get("script_id"),
                "backend_id": data.get("backend_id"),
                "frames_count": len(data.get("frames", [])),
                "generated_at": data.get("generated_at", ""),
                "success": data.get("success", False),
            })
        except Exception:
            pass
    return results