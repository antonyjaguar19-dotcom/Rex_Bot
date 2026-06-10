"""
Claw Bot Agent — Phase 3 FINAL
Adds: pause/resume feedback so you can revisit revisions later.
"""

import functools
import os
import sys
import asyncio
import logging
import logging.handlers
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import discord
from discord.ext import commands
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

sys.path.insert(0, str(Path(__file__).parent))
from modules.script_generator import (
    generate_script, revise_script, format_for_discord, OUTPUTS_DIR,
    rewrite_narration,
)
from modules.theme_bank import get_theme_of_the_day, get_random_theme, get_theme_count
from modules.safety_filter import check_safety, safety_summary_for_discord
from modules import pending_feedback as pf
from modules import storyboard_workflow as sw
from modules import video_workflow as vw
from modules import gpu_utils
from modules import runtime_settings as rs
from modules.health_monitor import StatusDashboard
from modules import model_registry
from modules import prompt_polisher as pp
from modules import shot_tailor as st
from modules import prompt_approval as pap
from modules import dashboard_nicegui as ui_dashboard
from modules import agent as agent_memory
from modules import agent_router
from modules import control_panel
from modules import channel_cleanup
from modules import approval_buttons
from modules import job_lock
from modules.file_utils import atomic_write_json


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_DIR   = PROJECT_ROOT / "05_Config"
LOGS_DIR     = PROJECT_ROOT / "06_Logs"

load_dotenv(dotenv_path=CONFIG_DIR / "secrets.env")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not DISCORD_BOT_TOKEN:
    print("ERROR: DISCORD_BOT_TOKEN missing in 05_Config/secrets.env")
    sys.exit(1)

BOT_VERSION = "0.7.0-styles"
DISCORD_MSG_LIMIT = 1900
IST = ZoneInfo("Asia/Kolkata")
DAILY_GEN_HOUR = 9
MAX_REVISIONS = 10
FEEDBACK_TIMEOUT_SEC = 300   # 5 min — then auto-pause (not abandon)

# Agent listens only in these channel names
AGENT_CHANNELS = {"claw-bot"}

STATS_FILE = CONFIG_DIR / "stats.json"


# ============================================================
# LOGGING
# ============================================================

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        # Rotate at 10 MB, keep 5 backups — the old single FileHandler grew
        # without bound across multi-hour render sessions.
        logging.handlers.RotatingFileHandler(
            LOGS_DIR / "claw_bot.log", encoding="utf-8",
            maxBytes=10 * 1024 * 1024, backupCount=5,
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("claw_bot")


# ============================================================
# STATS (future-proof loader)
# ============================================================

DEFAULT_STATS = {
    "generated": 0, "approved": 0, "rejected": 0,
    "safety_blocked": 0, "revisions": 0, "loops_stopped": 0,
    "paused": 0, "resumed": 0,
}


def load_stats() -> dict:
    data = DEFAULT_STATS.copy()
    if STATS_FILE.exists():
        try:
            data.update(json.loads(STATS_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    for k, v in DEFAULT_STATS.items():
        data.setdefault(k, v)
    return data


def save_stats(stats: dict):
    atomic_write_json(STATS_FILE, stats)


STATS = load_stats()


# ============================================================
# DISCORD SETUP
# ============================================================

intents = discord.Intents.default()
intents.message_content = True
intents.members         = True
intents.presences       = True
intents.reactions       = True

bot = commands.Bot(command_prefix="!", intents=intents)
BOT_START_TIME = None

# ── Persistence helper ─────────────────────────────────────────────────────
# Survives restart: on bot startup the saved dicts are reloaded and each
# pending approval message has its view re-attached via bot.add_view(), so
# the buttons still work even though the process was killed mid-flow.
PENDING_STATE_FILE = CONFIG_DIR / "pending_state.json"


class _PersistentDict(dict):
    """dict that auto-persists itself to disk on every mutation.

    Cheap because the dict is tiny (a handful of pending approvals) and
    msgpack/JSON write to a local SSD takes microseconds.
    """

    def __init__(self, key: str):
        super().__init__()
        self._key = key

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        _save_pending_state()

    def __delitem__(self, k):
        super().__delitem__(k)
        _save_pending_state()

    def pop(self, k, *args):
        result = super().pop(k, *args)
        _save_pending_state()
        return result

    def clear(self):
        super().clear()
        _save_pending_state()


PENDING_APPROVALS = _PersistentDict("script")
PENDING_STORYBOARD_APPROVALS = _PersistentDict("storyboard")
PENDING_VIDEO_APPROVALS = _PersistentDict("video")


def _save_pending_state():
    """Snapshot all pending approval dicts to JSON.

    Errors here must never crash the bot — at worst we lose persistence
    until next mutation succeeds.
    """
    try:
        snapshot = {
            "script": {str(k): v for k, v in PENDING_APPROVALS.items()},
            "storyboard": {str(k): v for k, v in PENDING_STORYBOARD_APPROVALS.items()},
            "video": {str(k): v for k, v in PENDING_VIDEO_APPROVALS.items()},
        }
        PENDING_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(PENDING_STATE_FILE, snapshot, default=str)
    except Exception as e:
        log.warning(f"Could not persist pending state: {e}")


def _load_pending_state() -> dict:
    if not PENDING_STATE_FILE.exists():
        return {"script": {}, "storyboard": {}, "video": {}}
    try:
        data = json.loads(PENDING_STATE_FILE.read_text(encoding="utf-8"))
        return {
            "script": data.get("script", {}),
            "storyboard": data.get("storyboard", {}),
            "video": data.get("video", {}),
        }
    except Exception as e:
        log.warning(f"Could not load pending state ({e}); starting fresh")
        return {"script": {}, "storyboard": {}, "video": {}}


scheduler = AsyncIOScheduler(timezone=IST)


# ============================================================
# HELPERS
# ============================================================

def _gpu_job(label: str):
    """Gate a pipeline entrypoint behind the shared GPU job lock.

    Cross-frontend: the same lock is claimed by the dashboard's workers, so
    a Discord command can't start rendering while the web UI is mid-job and
    vice versa. The first positional arg must be a channel, ctx, or
    interaction (anything with .send or .channel.send) so the busy notice
    has somewhere to go.
    """
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(first, *args, **kwargs):
            target = getattr(first, "channel", None) or first
            if not job_lock.acquire(f"discord:{label}"):
                try:
                    await target.send(
                        f"⏳ GPU busy with **{job_lock.holder_label()}** — "
                        f"try again when it finishes."
                    )
                except Exception:
                    log.warning("Could not deliver GPU-busy notice")
                return None
            try:
                from modules import config_check
                disk_ok, free_gb = config_check.check_disk_space()
                if not disk_ok:
                    try:
                        await target.send(
                            f"⚠️ Low disk: only **{free_gb} GB** free — "
                            f"clear space in 04_Outputs before rendering."
                        )
                    except Exception:
                        pass
                    return None
                return await fn(first, *args, **kwargs)
            finally:
                job_lock.release()
        return wrapper
    return deco


async def send_long_message(channel, text: str):
    if len(text) <= DISCORD_MSG_LIMIT:
        await channel.send(text)
        return
    chunks, current = [], ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > DISCORD_MSG_LIMIT:
            chunks.append(current); current = line
        else:
            current = (current + "\n" + line) if current else line
    if current:
        chunks.append(current)
    for chunk in chunks:
        await channel.send(chunk)


def get_channel_by_name(guild, name: str):
    return discord.utils.find(
        lambda c: c.name.lower() == name.lower(),
        guild.text_channels
    )


async def send_transient(channel, content: str, delete_after_seconds: int = 15):
    """Post a short status note that auto-deletes after N seconds. Reduces clutter."""
    try:
        msg = await channel.send(content, delete_after=delete_after_seconds)
        return msg
    except Exception as e:
        log.warning(f"send_transient failed: {e}")
        return None


async def _delete_pending_storyboard_approvals(channel, script_id: str):
    """Delete every still-pending storyboard approval message for this script_id.

    Used before re-posting a fresh approval after a regen so the old one
    (with its stale buttons) doesn't linger above the new frames.
    """
    stale_msg_ids = [
        mid for mid, entry in PENDING_STORYBOARD_APPROVALS.items()
        if entry.get("script_id") == script_id
    ]
    for mid in stale_msg_ids:
        try:
            msg = await channel.fetch_message(mid)
            await msg.delete()
        except discord.NotFound:
            pass  # already gone
        except Exception as e:
            log.warning(f"Could not delete old storyboard approval {mid}: {e}")
        PENDING_STORYBOARD_APPROVALS.pop(mid, None)


async def _delete_pending_video_approvals(channel, script_id: str):
    """Same as above but for video approval messages."""
    stale_msg_ids = [
        mid for mid, entry in PENDING_VIDEO_APPROVALS.items()
        if entry.get("script_id") == script_id
    ]
    for mid in stale_msg_ids:
        try:
            msg = await channel.fetch_message(mid)
            await msg.delete()
        except discord.NotFound:
            pass
        except Exception as e:
            log.warning(f"Could not delete old video approval {mid}: {e}")
        PENDING_VIDEO_APPROVALS.pop(mid, None)

# ============================================================
# BUTTON HANDLERS — called by approval_buttons.py
# Mirror the on_reaction_add logic so existing reactions still work
# as a fallback for old messages.
# ============================================================

async def _btn_script_approve(interaction, script, owner_id):
    global STATS
    channel = interaction.channel
    script_id = script.get("_id") or script.get("script_id")
    STATS["approved"] += 1; save_stats(STATS)
    rev_num = script.get("revision_number", 1)
    note = f" after {rev_num - 1} revision(s)" if rev_num > 1 else ""
    await channel.send(
        f"✅ **Script `{script_id}` APPROVED** by {interaction.user.mention}{note}.\n"
        f"🎨 Auto-starting storyboard generation..."
    )
    approved_dir = OUTPUTS_DIR.parent / "approved_scripts"
    approved_dir.mkdir(parents=True, exist_ok=True)
    (approved_dir / f"{script_id}.approved").write_text(
        f"approved_by={interaction.user}\napproved_at={datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )
    pf.remove(script_id)
    PENDING_APPROVALS.pop(interaction.message.id, None)

    storyboards_channel = get_channel_by_name(channel.guild, "storyboards") or channel
    asyncio.create_task(_polish_then_storyboard(
        channel, storyboards_channel, script, interaction.user.id, interaction.user.mention
    ))


async def _btn_script_reject(interaction, script, owner_id, feedback_text):
    global STATS
    channel = interaction.channel
    STATS["rejected"] += 1; save_stats(STATS)
    PENDING_APPROVALS.pop(interaction.message.id, None)

    # Disable the buttons (modal already responded, so use followup edit)
    try:
        view = approval_buttons.ScriptApprovalView(script=script, owner_id=owner_id)
        for c in view.children: c.disabled = True
        view.resolved = True
        await interaction.message.edit(
            content=interaction.message.content + f"\n\n❌ **Rejected by {interaction.user.mention}** — revising…",
            view=view,
        )
    except Exception as e:
        log.warning(f"Could not disable script reject buttons: {e}")

    # Run the existing revision flow with the feedback text
    asyncio.create_task(_do_revision(channel, script, feedback_text, interaction.user.id))


async def _btn_script_stop(interaction, script, owner_id):
    global STATS
    channel = interaction.channel
    script_id = script.get("_id") or script.get("script_id")
    STATS["loops_stopped"] += 1; save_stats(STATS)
    await channel.send(
        f"🛑 Story `{script_id}` stopped by {interaction.user.mention}. No further revisions."
    )
    pf.remove(script_id)
    PENDING_APPROVALS.pop(interaction.message.id, None)


async def _btn_storyboard_approve(interaction, script_id, owner_id):
    global STATS
    sb_channel = interaction.channel
    STATS["storyboards_approved"] = STATS.get("storyboards_approved", 0) + 1
    save_stats(STATS)
    await sb_channel.send(
        f"✅ **Storyboard `{script_id}` APPROVED** by {interaction.user.mention}.\n"
        f"🎥 Auto-starting video clip generation..."
    )
    approved_dir = OUTPUTS_DIR.parent / "approved_storyboards"
    approved_dir.mkdir(parents=True, exist_ok=True)
    (approved_dir / f"{script_id}.approved").write_text(
        f"approved_by={interaction.user}\napproved_at={datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )
    PENDING_STORYBOARD_APPROVALS.pop(interaction.message.id, None)

    videos_channel = get_channel_by_name(sb_channel.guild, "videos") or sb_channel
    asyncio.create_task(_run_video_pipeline(
        videos_channel, script_id, interaction.user.id, interaction.user.mention, is_auto=True
    ))


async def _btn_storyboard_reject(interaction, script_id, owner_id, feedback_text):
    global STATS
    sb_channel = interaction.channel
    STATS["storyboards_rejected"] = STATS.get("storyboards_rejected", 0) + 1
    save_stats(STATS)
    PENDING_STORYBOARD_APPROVALS.pop(interaction.message.id, None)

    try:
        view = approval_buttons.StoryboardApprovalView(script_id=script_id, owner_id=owner_id)
        for c in view.children: c.disabled = True
        view.resolved = True
        await interaction.message.edit(
            content=interaction.message.content + f"\n\n❌ **Rejected by {interaction.user.mention}** — revising…",
            view=view,
        )
    except Exception as e:
        log.warning(f"Could not disable storyboard reject buttons: {e}")

    # _run_storyboard_revision_loop expects a user object; we pass interaction.user
    asyncio.create_task(_run_storyboard_revision_loop_with_text(
        sb_channel, script_id, interaction.user, feedback_text
    ))


async def _btn_storyboard_stop(interaction, script_id, owner_id):
    sb_channel = interaction.channel
    await sb_channel.send(
        f"🛑 Storyboard `{script_id}` stopped by {interaction.user.mention}."
    )
    PENDING_STORYBOARD_APPROVALS.pop(interaction.message.id, None)


async def _btn_storyboard_pause(interaction, script_id, owner_id):
    sb_channel = interaction.channel
    await sb_channel.send(
        f"💾 Storyboard `{script_id}` paused. "
        f"Resume by rejecting again or use `!regen_shot {script_id} <shot_num>`."
    )
    PENDING_STORYBOARD_APPROVALS.pop(interaction.message.id, None)


async def _btn_shot_edit_prompt(interaction, script_id, shot_number, owner_id, new_prompt):
    """Edit the user-approved image prompt then regen the shot."""
    channel = interaction.channel
    try:
        shot_n = int(shot_number)
    except (TypeError, ValueError):
        await channel.send(f"❌ Invalid shot number: `{shot_number}`")
        return

    new_prompt = (new_prompt or "").strip()
    if not new_prompt:
        await channel.send(f"❌ Shot {shot_n}: empty prompt — nothing to regen.")
        return

    # Persist new prompt + fresh seed so the regen produces a different image
    import random as _rnd
    try:
        state = pap.load_approved_prompts(script_id) or {
            "script_id": script_id, "prompts": {}
        }
        prompts = state.setdefault("prompts", {})
        entry = prompts.setdefault(str(shot_n), {})
        entry["image_prompt"] = new_prompt
        entry["image_seed"] = _rnd.randint(1, 2_147_483_647)
        entry["approved"] = True
        pap._save(state)
        # Refresh in-memory cache so the regen picks up the new value
        pap._ACTIVE_STATES[script_id] = state
    except Exception as e:
        log.exception("Could not persist updated image prompt")
        await channel.send(f"❌ Could not save prompt: `{e}`")
        return

    await channel.send(
        f"✏️ **Shot {shot_n}** image prompt updated · new seed `{entry['image_seed']}` · "
        f"regenerating now…"
    )
    await _delete_pending_storyboard_approvals(channel, script_id)
    try:
        result_info = await sw.regenerate_shots(
            channel=channel,
            script_id=script_id,
            shot_numbers=[shot_n],
            owner_id=owner_id,
            post_approval=True,
        )
        if result_info and result_info.get("approval_msg_id"):
            PENDING_STORYBOARD_APPROVALS[result_info["approval_msg_id"]] = {
                "script_id": result_info["script_id"],
                "owner_id": result_info["owner_id"],
            }
    except Exception as e:
        log.exception("Edit-prompt regen failed")
        await channel.send(f"❌ Regen failed: `{e}`")


async def _btn_clip_edit_prompt(interaction, script_id, shot_number, owner_id, new_prompt):
    """Edit the user-approved motion prompt then regen the video clip."""
    channel = interaction.channel
    try:
        shot_n = int(shot_number)
    except (TypeError, ValueError):
        await channel.send(f"❌ Invalid shot number: `{shot_number}`")
        return

    new_prompt = (new_prompt or "").strip()
    if not new_prompt:
        await channel.send(f"❌ Shot {shot_n}: empty motion prompt.")
        return

    import random as _rnd
    try:
        state = pap.load_approved_prompts(script_id) or {
            "script_id": script_id, "prompts": {}
        }
        prompts = state.setdefault("prompts", {})
        entry = prompts.setdefault(str(shot_n), {})
        entry["motion_prompt"] = new_prompt
        entry["motion_seed"] = _rnd.randint(1, 2_147_483_647)
        entry["approved"] = True
        pap._save(state)
        pap._ACTIVE_STATES[script_id] = state
    except Exception as e:
        log.exception("Could not persist updated motion prompt")
        await channel.send(f"❌ Could not save prompt: `{e}`")
        return

    await channel.send(
        f"✏️ **Shot {shot_n}** motion prompt updated · new seed `{entry['motion_seed']}` · "
        f"regenerating clip now…"
    )
    await _delete_pending_video_approvals(channel, script_id)
    try:
        result_info = await vw.regenerate_video_shots(
            channel=channel,
            script_id=script_id,
            shot_numbers=[shot_n],
            owner_id=owner_id,
        )
        if result_info and result_info.get("approval_msg_id"):
            PENDING_VIDEO_APPROVALS[result_info["approval_msg_id"]] = {
                "script_id": result_info["script_id"],
                "owner_id": result_info["owner_id"],
                "channel_id": channel.id,
            }
    except Exception as e:
        log.exception("Edit-motion regen failed")
        await channel.send(f"❌ Clip regen failed: `{e}`")


@_gpu_job("clip regen")
async def _btn_clip_regen(interaction, script_id, shot_number, owner_id):
    """Re-render a single video clip as-is (no prompt change, fresh seed)."""
    channel = interaction.channel
    try:
        shot_n = int(shot_number)
    except (TypeError, ValueError):
        await channel.send(f"❌ Invalid shot number: `{shot_number}`")
        return

    # Bump seed so we get a different sample
    import random as _rnd
    try:
        state = pap.load_approved_prompts(script_id) or {
            "script_id": script_id, "prompts": {}
        }
        entry = state.setdefault("prompts", {}).setdefault(str(shot_n), {})
        entry["motion_seed"] = _rnd.randint(1, 2_147_483_647)
        pap._save(state)
        pap._ACTIVE_STATES[script_id] = state
    except Exception as e:
        log.warning(f"Could not bump motion seed for shot {shot_n}: {e}")

    await channel.send(f"🔁 **Re-rendering shot {shot_n} clip** of `{script_id}`…")
    await _delete_pending_video_approvals(channel, script_id)
    try:
        result_info = await vw.regenerate_video_shots(
            channel=channel,
            script_id=script_id,
            shot_numbers=[shot_n],
            owner_id=owner_id,
        )
        if result_info and result_info.get("approval_msg_id"):
            PENDING_VIDEO_APPROVALS[result_info["approval_msg_id"]] = {
                "script_id": result_info["script_id"],
                "owner_id": result_info["owner_id"],
                "channel_id": channel.id,
            }
    except Exception as e:
        log.exception("Clip regen failed")
        await channel.send(f"❌ Clip regen failed: `{e}`")


@_gpu_job("storyboard shot regen")
async def _btn_shot_regen(interaction, script_id, shot_number, owner_id, notes=""):
    """Per-shot regen button handler (button under each storyboard frame)."""
    channel = interaction.channel
    from modules import storyboard_workflow as sw
    try:
        shot_n = int(shot_number)
    except (TypeError, ValueError):
        await channel.send(f"❌ Invalid shot number: `{shot_number}`")
        return

    note_line = f"\n📝 Notes: _{notes}_" if notes and notes.strip() else ""
    await channel.send(
        f"🔁 **Regenerating shot {shot_n}** of `{script_id}`…{note_line}"
    )

    # Delete the old storyboard approval message (and its stale buttons), if any,
    # so the new one posts fresh at the bottom of the chat.
    await _delete_pending_storyboard_approvals(channel, script_id)

    try:
        result_info = await sw.regenerate_shots(
            channel=channel,
            script_id=script_id,
            shot_numbers=[shot_n],
            owner_id=owner_id,
            post_approval=True,   # post a fresh approval at the bottom
        )
        if result_info and result_info.get("approval_msg_id"):
            PENDING_STORYBOARD_APPROVALS[result_info["approval_msg_id"]] = {
                "script_id": result_info["script_id"],
                "owner_id": result_info["owner_id"],
            }
    except Exception as e:
        log.exception("Per-shot regen failed")
        await channel.send(f"❌ Regen failed: `{e}`")


async def _btn_video_approve(interaction, script_id, owner_id):
    global STATS
    v_channel = interaction.channel
    STATS["videos_approved"] = STATS.get("videos_approved", 0) + 1
    save_stats(STATS)
    await v_channel.send(
        f"✅ **Video clips for `{script_id}` APPROVED** by {interaction.user.mention}."
    )
    approved_dir = OUTPUTS_DIR.parent / "approved_videos"
    approved_dir.mkdir(parents=True, exist_ok=True)
    (approved_dir / f"{script_id}.approved").write_text(
        f"approved_by={interaction.user}\napproved_at={datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )
    PENDING_VIDEO_APPROVALS.pop(interaction.message.id, None)

    # ── Two-stage: low-res preview FIRST, upscale gated on approval ──
    asyncio.create_task(
        _run_preview_then_final(v_channel, script_id, interaction.user.id, interaction.user.mention)
    )


async def _btn_video_reject(interaction, script_id, owner_id, feedback_text):
    global STATS
    v_channel = interaction.channel
    STATS["videos_rejected"] = STATS.get("videos_rejected", 0) + 1
    save_stats(STATS)
    PENDING_VIDEO_APPROVALS.pop(interaction.message.id, None)

    try:
        view = approval_buttons.VideoApprovalView(script_id=script_id, owner_id=owner_id)
        for c in view.children: c.disabled = True
        view.resolved = True
        await interaction.message.edit(
            content=interaction.message.content + f"\n\n❌ **Rejected by {interaction.user.mention}** — feedback noted.",
            view=view,
        )
    except Exception as e:
        log.warning(f"Could not disable video reject buttons: {e}")

    await v_channel.send(
        f"❌ Video clips for `{script_id}` rejected. Feedback:\n> {feedback_text}\n\n"
        f"Use `!regen_video_shot {script_id} <shot_num>` to re-render specific clips."
    )


async def _btn_video_stop(interaction, script_id, owner_id):
    v_channel = interaction.channel
    await v_channel.send(
        f"🛑 Video clips for `{script_id}` stopped by {interaction.user.mention}."
    )
    PENDING_VIDEO_APPROVALS.pop(interaction.message.id, None)


async def _btn_video_pause(interaction, script_id, owner_id):
    v_channel = interaction.channel
    await v_channel.send(
        f"💾 Video clips for `{script_id}` paused."
    )
    PENDING_VIDEO_APPROVALS.pop(interaction.message.id, None)


async def _run_storyboard_revision_loop_with_text(channel, script_id, user, feedback_text):
    """Modal-fed revision: parse shot numbers from feedback and regen those shots.
    Falls back to free-form revision loop if no shot numbers found."""
    try:
        from modules import storyboard_workflow as sw

        # Load storyboard to know total shot count
        try:
            from modules import storyboard_generator as sg
            sb = sg.get_storyboard_status(script_id)
            total_shots = sb.total_frames if sb else 10
        except Exception:
            total_shots = 10

        shot_nums = sw.extract_shot_numbers(feedback_text, total_shots=total_shots)

        if shot_nums:
            await channel.send(
                f"🔁 Re-rendering shots `{shot_nums}` for `{script_id}` based on your feedback…"
            )
            await _delete_pending_storyboard_approvals(channel, script_id)
            try:
                result_info = await sw.regenerate_shots(
                    channel=channel,
                    script_id=script_id,
                    shot_numbers=shot_nums,
                    owner_id=user.id,
                )
                if result_info and result_info.get("approval_msg_id"):
                    PENDING_STORYBOARD_APPROVALS[result_info["approval_msg_id"]] = {
                        "script_id": result_info["script_id"],
                        "owner_id": result_info["owner_id"],
                    }
            except Exception as e:
                log.exception("regenerate_shots failed")
                await channel.send(f"❌ Regen failed: `{e}`")
        else:
            # No shot numbers in feedback — fall back to free-form chat-based revision
            await channel.send(
                f"📝 Got your feedback (no specific shot numbers detected).\n"
                f"> {feedback_text}\n\n"
                f"Reply in chat with the shot numbers you want re-rendered "
                f"(e.g. `redo shot 2 and 4`)."
            )
            asyncio.create_task(_run_storyboard_revision_loop(channel, script_id, user))
    except Exception as e:
        log.exception("revision shim failed")
        await channel.send(f"❌ Revision setup failed: `{e}`")

# ============================================================
# TWO-STAGE ASSEMBLY: low-res preview → approval → high-res
# ============================================================

# Track pending final-assembly approvals
PENDING_FINAL_APPROVALS: dict = {}  # msg_id -> {"script_id", "owner_id", "channel_id"}


class _FinalAssemblyView(discord.ui.View):
    """Approval buttons after low-res preview is posted."""

    def __init__(self, script_id: str, owner_id: int):
        super().__init__(timeout=86400)
        self.script_id = script_id
        self.owner_id = owner_id
        self.resolved = False

    async def interaction_check(self, interaction):
        if self.resolved:
            await interaction.response.send_message("ℹ️ Already decided.", ephemeral=True)
            return False
        if interaction.user.id != self.owner_id and not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message(
                "🔒 Only the requester (or an admin) can decide.", ephemeral=True
            )
            return False
        return True

    async def _lock(self, interaction, label: str):
        self.resolved = True
        for c in self.children:
            c.disabled = True
        msg = interaction.message
        new_content = (msg.content or "") + f"\n\n{label}"
        try:
            await interaction.response.edit_message(content=new_content, view=self)
        except discord.InteractionResponded:
            await msg.edit(content=new_content, view=self)

    @discord.ui.button(label="✅ Approve & Upscale", style=discord.ButtonStyle.success)
    async def approve(self, interaction, button):
        await self._lock(interaction, f"✅ **Approved by {interaction.user.mention}** — upscaling now…")
        PENDING_FINAL_APPROVALS.pop(interaction.message.id, None)
        asyncio.create_task(
            _run_upscale_and_final(interaction.channel, self.script_id, self.owner_id)
        )

    @discord.ui.button(label="📥 Skip Upscale", style=discord.ButtonStyle.secondary)
    async def skip(self, interaction, button):
        await self._lock(interaction, f"📥 **Kept as low-res** by {interaction.user.mention}.")
        PENDING_FINAL_APPROVALS.pop(interaction.message.id, None)
        await interaction.channel.send(
            f"📁 Low-res files are the final output for `{self.script_id}`.\n"
            f"Skipping upscale per your choice."
        )

    @discord.ui.button(label="❌ Reject", style=discord.ButtonStyle.danger)
    async def reject(self, interaction, button):
        await self._lock(interaction, f"❌ **Rejected by {interaction.user.mention}**.")
        PENDING_FINAL_APPROVALS.pop(interaction.message.id, None)
        await interaction.channel.send(
            f"❌ Final video for `{self.script_id}` rejected.\n"
            f"Use `!regen_video_shot {self.script_id} <shot_num>` to redo specific clips, "
            f"then `!assemble {self.script_id}` to retry."
        )


@_gpu_job("preview assembly")
async def _run_preview_then_final(channel, script_id: str, owner_id: int, owner_mention: str):
    """Stage 1: Assemble low-res preview from original clips, post to Discord for approval."""
    from modules import assembly as asm

    status = await channel.send(f"🎬 Building **low-res preview** for `{script_id}`…")

    try:
        # Assemble from non-upscaled clips. Default `assemble_final` uses upscaled
        # if present, but at this point upscale hasn't run yet — so it uses originals.
        asm_result = await asyncio.to_thread(asm.assemble_final, script_id, None)
    except Exception as e:
        log.exception("Preview assembly failed")
        await status.edit(content=f"❌ Preview assembly failed: `{e}`")
        return

    summary = (
        f"🎬 **Low-res preview ready** — {asm_result['shot_count']} shots, "
        f"{asm_result['total_duration_sec']:.1f}s\n"
        f"_Review below. Approve to run upscale + final render._"
    )
    await status.edit(content=summary)

    # Try to attach the 9x16 preview (Shorts format) — smaller is more uploadable
    preview_file = asm_result["9x16"]
    size_mb = preview_file.stat().st_size / (1024 * 1024)
    if size_mb <= 9:
        try:
            await channel.send(file=discord.File(str(preview_file)))
        except discord.HTTPException as ue:
            await channel.send(
                f"📁 `{preview_file.name}` ({size_mb:.1f} MB) — upload failed ({ue.status}). "
                f"Saved at:\n`{preview_file}`"
            )
    else:
        await channel.send(
            f"📁 `{preview_file.name}` ({size_mb:.1f} MB) — too large to upload here.\n"
            f"Saved at:\n`{preview_file}`"
        )

    # Post the approval message with buttons
    upscale_on = rs.get_upscale_enabled()
    note = (
        "📈 Upscale is **ENABLED** — Approve to run high-res render."
        if upscale_on else
        "📉 Upscale is **DISABLED** in settings — Approve to finalize as-is."
    )
    view = _FinalAssemblyView(script_id=script_id, owner_id=owner_id)
    approval = await channel.send(
        f"**Final preview for `{script_id}`** — your call.\n{note}",
        view=view,
    )
    PENDING_FINAL_APPROVALS[approval.id] = {
        "script_id": script_id, "owner_id": owner_id, "channel_id": channel.id,
    }


@_gpu_job("upscale + final render")
async def _run_upscale_and_final(channel, script_id: str, owner_id: int):
    """Stage 2: Upscale originals (if enabled) then re-assemble at high-res."""
    upscale_on = rs.get_upscale_enabled()

    if upscale_on:
        from modules import upscaler as up
        status_msg = await channel.send(
            f"📈 Upscaling clips for `{script_id}` (~1-2 min/shot)…"
        )

        bot_loop = asyncio.get_running_loop()
        last_update = {"t": 0.0, "tick": 0}
        try:
            from modules.progress_bar import render_indeterminate
        except Exception:
            render_indeterminate = lambda tick: ""

        def progress(text: str):
            import time as _t
            now = _t.time()
            if now - last_update["t"] < 2.0:
                return
            last_update["t"] = now
            last_update["tick"] += 1
            bar = render_indeterminate(last_update["tick"])
            asyncio.run_coroutine_threadsafe(
                status_msg.edit(content=f"📈 **Upscaling** `{script_id}`\n{bar} · {text}"),
                bot_loop,
            )

        try:
            result = await asyncio.to_thread(up.upscale_storyboard_videos, script_id, progress)
            await channel.send(up.format_upscale_summary(result))
        except Exception as e:
            log.exception("Upscale failed in two-stage flow")
            await channel.send(
                f"⚠️ Upscale failed: `{e}`\n"
                f"Keeping low-res. Run manually with `!upscale {script_id}` once fixed."
            )
            return
    else:
        await channel.send(f"📉 Upscale disabled — proceeding to high-res assembly directly.")

    # Re-assemble (this time using upscaled clips if upscale ran)
    from modules import assembly as asm
    await channel.send(f"🎬 Re-assembling final video for `{script_id}`…")
    try:
        asm_result = await asyncio.to_thread(asm.assemble_final, script_id, None)
    except Exception as e:
        log.exception("Final assembly failed")
        await channel.send(f"⚠️ Final assembly failed: `{e}`")
        return

    summary = (
        f"✅ **Final video ready** — {asm_result['shot_count']} shots, "
        f"{asm_result['total_duration_sec']:.1f}s · ⏱ assembly {asm_result.get('wall_sec', 0):.0f}s\n"
        f"📂 `{asm_result['9x16'].name}` (Shorts)\n"
        f"📂 `{asm_result['16x9'].name}` (Horizontal)"
    )
    await channel.send(summary)

    # Show paths — high-res files almost always exceed Discord limit
    for key in ("9x16", "16x9"):
        f = asm_result[key]
        size_mb = f.stat().st_size / (1024 * 1024)
        if size_mb <= 9:
            try:
                await channel.send(file=discord.File(str(f)))
            except discord.HTTPException as ue:
                await channel.send(
                    f"📁 `{f.name}` ({size_mb:.1f} MB) — too large to upload ({ue.status}). "
                    f"File at:\n`{f}`"
                )
        else:
            await channel.send(
                f"📁 `{f.name}` ({size_mb:.1f} MB) — preview too large.\n"
                f"File at:\n`{f}`"
            )
    await channel.send("*Ready for YouTube upload (Phase 6C).*")


async def _post_with_approval_reactions(channel, script: dict, owner_id: int):
    """Post script + approval prompt with action buttons."""
    await send_long_message(channel, format_for_discord(script))

    rev_num = script.get("revision_number", 1)
    rev_info = f" (revision v{rev_num})" if rev_num > 1 else ""

    view = approval_buttons.ScriptApprovalView(script=script, owner_id=owner_id)
    approval_msg = await channel.send(
        f"**Script `{script['script_id']}`**{rev_info} — awaiting your decision.",
        view=view,
    )

    PENDING_APPROVALS[approval_msg.id] = {"script": script, "owner_id": owner_id}
    # Unload Llama from VRAM so image generation has room
    try:
        await asyncio.to_thread(gpu_utils.free_ollama_vram)
    except Exception as e:
        log.warning(f"Ollama unload after script gen failed: {e}")
    return approval_msg


@_gpu_job("script generation")
async def _generate_and_post(channel, theme: str, requested_by_id: int,
                             requested_by_mention: str, *, is_auto: bool = False):
    global STATS
    prefix = "🌅 **Daily Auto-Story**" if is_auto else f"📝 Request from {requested_by_mention}"
    status_msg = await channel.send(f"{prefix}\nTheme: **{theme}**\n⏳ Writing... (20-40 sec)")

    try:
        _script_t0 = time.perf_counter()
        script = await asyncio.to_thread(generate_script, theme)
        _script_secs = time.perf_counter() - _script_t0
        STATS["generated"] += 1
        save_stats(STATS)
    except Exception as e:
        log.exception(f"Script generation failed: {e}")
        await status_msg.edit(content=f"❌ Generation failed: `{e}`")
        return

    is_safe, blocked, _ = check_safety(script)
    if not is_safe:
        STATS["safety_blocked"] += 1
        save_stats(STATS)
        await status_msg.edit(
            content=f"🚨 Script blocked by safety filter\n"
                    f"Theme: `{theme}`\nBlocked: `{', '.join(blocked)}`"
        )
        return

    # Track this as the channel's current script so agent tools can find it
    try:
        agent_memory.set_current_script(str(channel.id), script.get("_id", ""))
        agent_memory.set_stage(str(channel.id), "script_generated")
    except Exception as e:
        log.warning(f"Could not set current script in memory: {e}")

    try:
        await status_msg.delete()
    except Exception:
        pass
    _sm, _ss = divmod(int(round(_script_secs)), 60)
    _sdur = f"{_sm}m {_ss}s" if _sm else f"{_ss}s"
    await channel.send(f"📝 Script written · ⏱ {_sdur}")
    await _post_with_approval_reactions(channel, script, requested_by_id)


async def _run_revision_loop(channel, original_script: dict, owner: discord.User):
    """
    Ask for notes via chat. User can:
      - give feedback -> revise
      - type 'pause' or react 💾 -> save for later
      - type 'stop' -> abandon
      - do nothing -> auto-pause on timeout (NOT abandon)
    """
    global STATS

    current_rev = original_script.get("revision_number", 1)
    if current_rev >= MAX_REVISIONS:
        await channel.send(
            f"⚠️ Revision cap reached ({MAX_REVISIONS}). "
            f"Use `!generate_script <theme>` to start fresh."
        )
        return

    sid = original_script.get("_id") or original_script.get("script_id")
    prompt_msg = await channel.send(
        f"{owner.mention} ❌ Script `{sid}` rejected.\n"
        f"**What should I change?** (reply within {FEEDBACK_TIMEOUT_SEC // 60} min)\n\n"
        f"💡 *Your options:*\n"
        f"• **Reply with notes** → I'll revise right now\n"
        f"• Type **`pause`** (or react 💾) → save for later, come back anytime\n"
        f"• Type **`stop`** → abandon this story\n"
        f"• Do nothing → I'll auto-pause after {FEEDBACK_TIMEOUT_SEC // 60} min (not lost!)\n\n"
        f"💡 *Tip for good notes:*\n"
        f"• **Surgical:** \"In shot 4, have them hugging instead of smiling\" — only shot 4 changes\n"
        f"• **Full:** \"Make the tone warmer throughout\" — whole story rewrites\n"
        f"• **Avoid:** \"Change shot 4\" — too vague, model can't act on it"
    )
    await prompt_msg.add_reaction("💾")

    def check_msg(m: discord.Message) -> bool:
        return m.author.id == owner.id and m.channel.id == channel.id

    def check_react(reaction, user):
        return (
            user.id == owner.id
            and reaction.message.id == prompt_msg.id
            and str(reaction.emoji) == "💾"
        )

    # Race: either a message or a 💾 reaction
    msg_task = asyncio.create_task(bot.wait_for("message", check=check_msg))
    react_task = asyncio.create_task(bot.wait_for("reaction_add", check=check_react))

    try:
        done, pending = await asyncio.wait(
            [msg_task, react_task],
            timeout=FEEDBACK_TIMEOUT_SEC,
            return_when=asyncio.FIRST_COMPLETED,
        )
    except Exception as e:
        log.error(f"Feedback wait error: {e}")
        for t in (msg_task, react_task):
            t.cancel()
        return

    # Cancel whichever didn't fire
    for t in pending:
        t.cancel()

    # Timeout → auto-pause (not abandon)
    if not done:
        pf.add(original_script, owner.id, channel.id, reason="timeout")
        STATS["paused"] += 1
        save_stats(STATS)
        await channel.send(
            f"⏱️ No feedback in {FEEDBACK_TIMEOUT_SEC // 60} min. "
            f"Script `{sid}` **paused** (not lost).\n"
            f"Resume anytime with `!resume_feedback {sid}` or see all with `!pending`."
        )
        log.info(f"Script {sid} auto-paused on timeout.")
        return

    result = done.pop().result()

    # Case 1: 💾 reaction clicked
    if isinstance(result, tuple):   # reaction_add yields (reaction, user)
        pf.add(original_script, owner.id, channel.id, reason="paused")
        STATS["paused"] += 1
        save_stats(STATS)
        await channel.send(
            f"💾 Script `{sid}` **paused**. Resume anytime with "
            f"`!resume_feedback {sid}` or `!pending` to see all."
        )
        log.info(f"Script {sid} manually paused.")
        return

    # Case 2: Message reply
    feedback_msg: discord.Message = result
    notes = feedback_msg.content.strip()
    notes_lower = notes.lower()

    # Pause keyword
    if notes_lower in {"pause", "save", "later"}:
        pf.add(original_script, owner.id, channel.id, reason="paused")
        STATS["paused"] += 1
        save_stats(STATS)
        await channel.send(
            f"💾 Script `{sid}` **paused**. Resume anytime with "
            f"`!resume_feedback {sid}`."
        )
        log.info(f"Script {sid} paused via keyword.")
        return

    # Stop keyword
    if notes_lower in {"stop", "cancel", "abort", "quit", "abandon"}:
        STATS["loops_stopped"] += 1
        save_stats(STATS)
        await channel.send(f"🛑 Script `{sid}` abandoned. No revision generated.")
        log.info(f"Script {sid} stopped by keyword.")
        return

    # Safety check on notes
    notes_blob = {"title": "", "theme": notes, "setting": "", "moral": "",
                  "characters": [], "shots": []}
    is_safe, blocked, _ = check_safety(notes_blob)
    if not is_safe:
        await channel.send(
            f"🚨 Notes contain unsafe words: `{', '.join(blocked)}`. "
            f"Rephrase and reject again."
        )
        return

    # Proceed with revision
    await _do_revision(channel, original_script, notes, owner.id)


@_gpu_job("script revision")
async def _do_revision(channel, original_script: dict, notes: str, owner_id: int):
    """Perform the actual revision + post."""
    global STATS
    status = await channel.send(
        f"✏️ Revising with your notes:\n> *{notes[:200]}*\n"
        f"🧠 Thinking pass (Qwen reads feedback)…\n"
        f"⏳ ~60-90 sec"
    )
    try:
        new_script = await asyncio.to_thread(revise_script, original_script, notes)
        STATS["revisions"] += 1
        STATS["generated"] += 1
        save_stats(STATS)
    except Exception as e:
        log.error(f"Revision failed: {e}")
        await status.edit(content=f"❌ Revision failed: `{e}`. Reject again to try once more.")
        return

    is_safe, blocked_out, _ = check_safety(new_script)
    if not is_safe:
        STATS["safety_blocked"] += 1
        save_stats(STATS)
        await status.edit(
            content=f"🚨 Revision blocked by safety filter (`{', '.join(blocked_out)}`). "
                    f"Reject again with different notes."
        )
        return

    await status.delete()
    await _post_with_approval_reactions(channel, new_script, owner_id)


# ============================================================
# DAILY JOB
# ============================================================

async def daily_auto_generation():
    log.info("Daily auto-generation triggered.")
    theme = get_theme_of_the_day()
    for guild in bot.guilds:
        target = get_channel_by_name(guild, "scripts")
        if target:
            await _generate_and_post(
                target, theme,
                requested_by_id=guild.owner_id,
                requested_by_mention="daily scheduler",
                is_auto=True,
            )


# ============================================================
# SYNC BRIDGE — Web → Discord
# ============================================================
# Discord → Web already syncs (dashboard polls disk every 1.5s). This closes
# the other direction: when the dashboard regenerates a clip/frame, it writes
# an event via modules.sync_bridge; this poller re-posts the updated media to
# the matching Discord channel so both front-ends stay in lockstep.

SYNC_POLL_SEC = 5


async def _handle_sync_event(ev: dict):
    """Re-post dashboard-originated media into the right Discord channel."""
    if ev.get("source") != "dashboard":
        return
    etype = ev.get("type")
    data = ev.get("data", {})
    sid = data.get("script_id")
    guild = bot.guilds[0] if bot.guilds else None
    if not guild:
        return
    owner_id = guild.owner_id or 0

    if etype == "video_regen":
        ch = get_channel_by_name(guild, "videos")
        path = Path(data.get("clip_path", ""))
        shot = data.get("shot")
        if not ch or not path.exists():
            return
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > 24:
            await ch.send(
                f"🔄 Shot {shot} re-rendered from dashboard — too large to upload "
                f"({size_mb:.1f} MB). Saved at `{path}`."
            )
            return
        from modules.embed_styles import themed_embed
        from modules.approval_buttons import ClipControlView
        embed = themed_embed("video", f"Shot {shot} (dashboard regen)",
                             "🔄 Updated from the web dashboard.")
        embed.color = discord.Color.gold()
        embed.set_footer(text=f"{size_mb:.2f} MB · synced from dashboard")
        await ch.send(
            embed=embed,
            file=discord.File(str(path), filename=path.name),
            view=ClipControlView(script_id=sid, shot_number=shot, owner_id=owner_id),
        )

    elif etype == "storyboard_regen":
        ch = get_channel_by_name(guild, "storyboards")
        path = Path(data.get("image_path", ""))
        shot = data.get("shot")
        if not ch or not path.exists():
            return
        embed = discord.Embed(
            title=f"🔄 Shot {shot} frame (dashboard regen)",
            description="Updated from the web dashboard.",
            color=discord.Color.gold(),
        )
        embed.set_image(url=f"attachment://{path.name}")
        await ch.send(embed=embed,
                      file=discord.File(str(path), filename=path.name))

    elif etype in ("video_done", "storyboard_done", "final_done"):
        ch = get_channel_by_name(guild, "videos") or get_channel_by_name(guild, "claw-bot")
        if ch:
            label = {"video_done": "🎥 Video render",
                     "storyboard_done": "🎨 Storyboard",
                     "final_done": "🎬 Final assembly"}.get(etype, etype)
            await ch.send(f"🔄 {label} for `{sid}` finished on the dashboard.")


async def _sync_bridge_loop():
    """Background poller: drain dashboard events → Discord. Skips backlog."""
    from modules import sync_bridge as sbr
    await bot.wait_until_ready()
    cursor = sbr.latest_id()      # ignore events from before this run
    sbr.set_cursor(cursor)
    log.info(f"Sync bridge poller started (cursor={cursor})")
    while not bot.is_closed():
        try:
            await asyncio.sleep(SYNC_POLL_SEC)
            for ev in sbr.read_new(cursor):
                cursor = ev["id"]
                try:
                    await _handle_sync_event(ev)
                except Exception as e:
                    log.warning(f"sync event {ev.get('type')} #{ev.get('id')} failed: {e}")
            sbr.set_cursor(cursor)
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.warning(f"sync bridge loop error: {e}")


# ============================================================
# EVENTS
# ============================================================

@bot.event
async def on_ready():
    global BOT_START_TIME
    BOT_START_TIME = datetime.now(timezone.utc)
    log.info(f"Logged in as {bot.user}")

    if not scheduler.running:
        scheduler.add_job(
            daily_auto_generation,
            trigger=CronTrigger(hour=DAILY_GEN_HOUR, minute=0, timezone=IST),
            id="daily_story", replace_existing=True,
        )
        scheduler.start()

    for guild in bot.guilds:
        ch = get_channel_by_name(guild, "status")
        if ch:
            nj = scheduler.get_job("daily_story")
            next_str = nj.next_run_time.strftime("%Y-%m-%d %H:%M %Z") if nj else "N/A"
            pending_count = pf.count()
            pending_line = (
                f"💾 Paused feedback: `{pending_count}` (use `!pending` to see)\n"
                if pending_count else ""
            )
            await ch.send(
                f"🤖 **Claw Bot v{BOT_VERSION}** online.\n"
                f"📅 Next auto-story: `{next_str}`\n"
                f"🎯 Theme bank: `{get_theme_count()}` themes\n"
                f"{pending_line}"
                f"🎨 **Styles:** storybook, cartoon, anime, watercolor, pixelart (use `!list_styles`)\n"
                f"📖 **New:** Type `!commands` for the full command reference.\n"
                f"Scripts auto-trigger storyboards on approval."
            )
            try:
                dashboard = StatusDashboard(
                    bot=bot,
                    bot_version=BOT_VERSION,
                    stats_getter=lambda: STATS,
                )
                from modules import health_monitor as hm
                hm.set_instance(dashboard)
                await dashboard.start(ch)
                bot._dashboard = dashboard   # keep reference alive
            except Exception as e:
                log.warning(f"Could not start status dashboard: {e}")

        # --- Auto-spawn the control panel in #claw-bot ---
        panel_ch = get_channel_by_name(guild, "claw-bot")
        if panel_ch:
            try:
                cmds = {
                    "generate_script":     cmd_generate_script,
                    "today_script":        cmd_today_script,
                    "suggest_theme":       cmd_suggest_theme,
                    "list_scripts":        cmd_list_scripts,
                    "show_script":         cmd_show_script,
                    "rewrite_narration":   cmd_rewrite_narration,
                    "repolish":            cmd_repolish,
                    "generate_storyboard": cmd_generate_storyboard,
                    "list_storyboards":    cmd_list_storyboards,
                    "regen_shot":          cmd_regen_shot,
                    "edit_prompts":        cmd_edit_prompts,
                    "generate_video":      cmd_generate_video,
                    "regen_video_shot":    cmd_regen_video_shot,
                    "upscale":             cmd_upscale,
                    "assemble":            cmd_assemble,
                    "set_style":           cmd_set_style,
                    "set_resolution":      cmd_set_resolution,
                    "set_video_resolution": cmd_set_video_resolution,
                    "set_steps":           cmd_set_steps,
                    "set_cfg":             cmd_set_cfg,
                    "current_settings":    cmd_current_settings,
                    "list_styles":         cmd_list_styles,
                    "reset_settings":      cmd_reset_settings,
                    "set_voice":           cmd_set_voice,
                    "list_voices":         cmd_list_voices,
                    "set_clip_length":     cmd_set_clip_length,
                    "set_sync_mode":       cmd_set_sync_mode,
                    "set_transition":      cmd_set_transition,
                    "set_upscale":         cmd_set_upscale,
                    "set_music_mood":      cmd_set_music_mood,
                    "regenerate_music":    cmd_regenerate_music,
                    "switch_model":        cmd_switch_model,
                    "stats":               cmd_stats,
                    "pending":             cmd_pending,
                    "resume_feedback":     cmd_resume_feedback,
                    "drop_feedback":       cmd_drop_feedback,
                    "restart_bot":         cmd_restart_bot,
                    "shutdown_bot":        cmd_shutdown_bot,
                }
                control_panel.register_views(bot, cmds)
                await control_panel.ensure_panel(panel_ch)
            except Exception as e:
                log.warning(f"Could not spawn control panel: {e}")

    # --- Register approval-button handlers (runs once, outside guild loop) ---
    try:
        approval_buttons.register_handlers({
            "script_approve":     _btn_script_approve,
            "script_reject":      _btn_script_reject,
            "script_stop":        _btn_script_stop,
            "storyboard_approve": _btn_storyboard_approve,
            "storyboard_reject":  _btn_storyboard_reject,
            "storyboard_stop":    _btn_storyboard_stop,
            "storyboard_pause":   _btn_storyboard_pause,
            "video_approve":      _btn_video_approve,
            "video_reject":       _btn_video_reject,
            "video_stop":         _btn_video_stop,
            "video_pause":        _btn_video_pause,
            "shot_regen":         _btn_shot_regen,
            "shot_edit_prompt":   _btn_shot_edit_prompt,
            "clip_edit_prompt":   _btn_clip_edit_prompt,
            "clip_regen":         _btn_clip_regen,
        })
    except Exception as e:
        log.warning(f"Could not register approval handlers: {e}")

    # --- Reattach persistent approval views from prior bot run ---
    try:
        await _restore_pending_approvals()
    except Exception as e:
        log.warning(f"Could not restore pending approvals: {e}")

    # --- Auto-launch browser dashboard ---
    try:
        ui_dashboard.launch_dashboard(port=7860, host="127.0.0.1", open_browser=True)
        log.info("Dashboard launched at http://127.0.0.1:7860")
    except Exception as e:
        log.warning(f"Dashboard launch failed: {e}")

    # --- Start Web → Discord sync bridge poller (once) ---
    if not getattr(bot, "_sync_loop_started", False):
        bot._sync_loop_started = True
        bot.loop.create_task(_sync_bridge_loop())


async def _restore_pending_approvals():
    """On bot startup, walk the persisted pending dicts and re-attach a
    persistent view to each Discord message so the buttons still work.

    Discord routes interactions to a View via custom_id. As long as the
    custom_ids match (they do — defined on each Button), bot.add_view() with
    the correct message_id makes everything wire back up automatically.
    """
    saved = _load_pending_state()
    n_restored = 0
    n_dropped = 0

    # Script approvals
    for msg_id_str, entry in (saved.get("script") or {}).items():
        try:
            msg_id = int(msg_id_str)
        except (TypeError, ValueError):
            continue
        script = entry.get("script") or {}
        owner_id = int(entry.get("owner_id") or 0)
        if not script or not owner_id:
            n_dropped += 1
            continue
        try:
            view = approval_buttons.ScriptApprovalView(script=script, owner_id=owner_id)
            bot.add_view(view, message_id=msg_id)
            PENDING_APPROVALS[msg_id] = {"script": script, "owner_id": owner_id}
            n_restored += 1
        except Exception as e:
            log.warning(f"Could not restore script approval {msg_id}: {e}")
            n_dropped += 1

    # Storyboard approvals
    for msg_id_str, entry in (saved.get("storyboard") or {}).items():
        try:
            msg_id = int(msg_id_str)
        except (TypeError, ValueError):
            continue
        script_id = entry.get("script_id")
        owner_id = int(entry.get("owner_id") or 0)
        if not script_id or not owner_id:
            n_dropped += 1
            continue
        try:
            view = approval_buttons.StoryboardApprovalView(script_id=script_id, owner_id=owner_id)
            bot.add_view(view, message_id=msg_id)
            PENDING_STORYBOARD_APPROVALS[msg_id] = {"script_id": script_id, "owner_id": owner_id}
            n_restored += 1
        except Exception as e:
            log.warning(f"Could not restore storyboard approval {msg_id}: {e}")
            n_dropped += 1

    # Video approvals
    for msg_id_str, entry in (saved.get("video") or {}).items():
        try:
            msg_id = int(msg_id_str)
        except (TypeError, ValueError):
            continue
        script_id = entry.get("script_id")
        owner_id = int(entry.get("owner_id") or 0)
        if not script_id or not owner_id:
            n_dropped += 1
            continue
        try:
            view = approval_buttons.VideoApprovalView(script_id=script_id, owner_id=owner_id)
            bot.add_view(view, message_id=msg_id)
            PENDING_VIDEO_APPROVALS[msg_id] = {
                "script_id": script_id, "owner_id": owner_id,
                "channel_id": entry.get("channel_id"),
            }
            n_restored += 1
        except Exception as e:
            log.warning(f"Could not restore video approval {msg_id}: {e}")
            n_dropped += 1

    log.info(
        f"Pending approvals restored: {n_restored} reattached, "
        f"{n_dropped} dropped (incomplete/invalid)"
    )


async def _maybe_repost_panel_in_claw_bot(message: discord.Message):
    """If a message lands in #claw-bot, nudge the panel back to the bottom."""
    try:
        if message.channel and getattr(message.channel, "name", None) == "claw-bot":
            await control_panel.maybe_repost_panel(message.channel)
    except Exception as e:
        log.debug(f"repost check failed: {e}")


@bot.event
async def on_message(message: discord.Message):
    """Route non-command messages in agent channels through the conversational agent."""
    # If anything lands in #claw-bot (including the bot's own posts), nudge the panel down
    asyncio.create_task(_maybe_repost_panel_in_claw_bot(message))

    # Always ignore the bot's own messages for command processing
    if message.author == bot.user or message.author.bot:
        return

    # Always let !commands work (even in agent channels)
    if message.content.startswith(bot.command_prefix):
        await bot.process_commands(message)
        return

    # Only intercept in designated agent channels
    if not message.channel.name or message.channel.name not in AGENT_CHANNELS:
        await bot.process_commands(message)
        return

    # Route through the agent
    asyncio.create_task(_handle_agent_message(message))


async def _handle_agent_message(message: discord.Message):
    """Run the router, execute any tool, post the reply."""
    channel = message.channel
    user_text = message.content.strip()
    if not user_text:
        return

    channel_id = str(channel.id)

    try:
        # Show "Claw Bot is typing..." while Qwen thinks
        async with channel.typing():
            decision = await asyncio.to_thread(
                agent_router.route, channel_id, user_text
            )

        # Save user turn FIRST (preserves chronological order)
        agent_memory.append_turn(channel_id, "user", user_text,
                                 meta={"author": str(message.author)})

        reply_text = decision.get("reply") or "..."
        tool_call = decision.get("tool_call", {}) or {}
        tool_name = tool_call.get("name", "no_tool")
        tool_args = tool_call.get("args", {}) or {}

        # Send the conversational reply
        await send_long_message(channel, reply_text)

        # Save assistant turn
        agent_memory.append_turn(
            channel_id, "assistant", reply_text,
            meta={"tool_called": tool_name, "tool_args": tool_args},
        )

        # Dispatch tool if needed — and record that it's been kicked off
        if tool_name and tool_name != "no_tool":
            try:
                await _dispatch_agent_tool(message, tool_name, tool_args)
                # Record tool completion so router knows it's done
                agent_memory.append_turn(
                    channel_id, "tool",
                    f"[{tool_name} dispatched successfully — output already shown to user above]",
                    meta={"tool_name": tool_name, "status": "dispatched"},
                )
            except Exception as e:
                log.exception(f"Tool dispatch failed: {e}")
                agent_memory.append_turn(
                    channel_id, "tool",
                    f"[{tool_name} FAILED: {e}]",
                    meta={"tool_name": tool_name, "status": "failed"},
                )

    except Exception as e:
        log.exception(f"Agent message handling failed: {e}")
        await channel.send(f"⚠️ My brain hiccuped: `{e}`")


async def _dispatch_agent_tool(message: discord.Message, tool_name: str, args: dict):
    """Execute a tool the router asked for."""
    channel = message.channel
    channel_id = str(channel.id)

    if tool_name == "no_tool" or not tool_name:
        return

    # ----- Helper: resolve script_id from args, or fall back to current -----
    def _resolve_script_id() -> str | None:
        explicit = args.get("script_id")
        if explicit and isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        return agent_memory.get_current_script(channel_id)

    # ----- generate_script -----
    if tool_name == "generate_script":
        theme = args.get("theme") or ""
        if not theme:
            await channel.send("⚠️ I wanted to make a story but didn't catch the theme — could you say it again?")
            return
        asyncio.create_task(_generate_and_post(
            channel, theme, requested_by_id=message.author.id,
            requested_by_mention=message.author.mention,
            is_auto=False,
        ))
        return

    # ----- revise_script -----
    if tool_name == "revise_script":
        feedback = args.get("feedback") or ""
        if not feedback:
            await channel.send("⚠️ I want to revise but didn't catch what to change — say it again?")
            return
        script_id = _resolve_script_id()
        if not script_id:
            await channel.send("⚠️ No script to revise yet — generate one first.")
            return
        script_file = OUTPUTS_DIR / f"script_{script_id}.json"
        if not script_file.exists():
            await channel.send(f"⚠️ Couldn't find script `{script_id}`.")
            return
        try:
            original = json.loads(script_file.read_text(encoding="utf-8"))
            asyncio.create_task(_do_revision(
                channel, original, feedback, message.author.id
            ))
        except Exception as e:
            await channel.send(f"⚠️ Revision failed: `{e}`")
        return

    # ----- start_storyboard -----
    if tool_name == "start_storyboard":
        script_id = _resolve_script_id()
        if not script_id:
            await channel.send("⚠️ No current script — generate or approve one first.")
            return
        if not (OUTPUTS_DIR / f"script_{script_id}.json").exists():
            await channel.send(f"⚠️ Script `{script_id}` not found.")
            return
        storyboards_channel = get_channel_by_name(channel.guild, "storyboards") or channel
        asyncio.create_task(_run_storyboard_pipeline(
            storyboards_channel, script_id, message.author.id,
            message.author.mention, is_auto=False,
        ))
        return

    # ----- regenerate_shot -----
    if tool_name == "regenerate_shot":
        shot_numbers = args.get("shot_numbers") or []
        if not isinstance(shot_numbers, list) or not shot_numbers:
            await channel.send("⚠️ Which shot(s) should I redo?")
            return
        script_id = _resolve_script_id()
        if not script_id:
            await channel.send("⚠️ No current script — which storyboard?")
            return
        try:
            shot_ints = [int(n) for n in shot_numbers]
        except (TypeError, ValueError):
            await channel.send("⚠️ Shot numbers should be integers.")
            return
        try:
            await sw.regenerate_shots(
                channel=channel,
                script_id=script_id,
                shot_numbers=shot_ints,
                owner_id=message.author.id,
            )
        except AttributeError:
            await channel.send("⚠️ Storyboard regen not exposed by sw module — use `!regen_shot`.")
        except Exception as e:
            await channel.send(f"⚠️ Regen failed: `{e}`")
        return

    # ----- start_video -----
    if tool_name == "start_video":
        script_id = _resolve_script_id()
        if not script_id:
            await channel.send("⚠️ No current script — generate one first.")
            return
        videos_channel = get_channel_by_name(channel.guild, "videos") or channel
        asyncio.create_task(_run_video_pipeline(
            videos_channel, script_id, message.author.id,
            message.author.mention, is_auto=False,
        ))
        return

    # ----- list_recent_scripts -----
    if tool_name == "list_recent_scripts":
        files = sorted(OUTPUTS_DIR.glob("script_*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)[:10]
        if not files:
            await channel.send("📭 No stories yet.")
            return
        lines = ["📚 **Recent stories:**"]
        for f in files:
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                title = d.get("title", "Untitled")
                sid = d.get("_id", f.stem.replace("script_", ""))
                lines.append(f"• `{sid}` — {title}")
            except Exception:
                continue
        await channel.send("\n".join(lines))
        return

    log.warning(f"Agent requested unknown tool: {tool_name}")
    await channel.send(f"⚠️ My router asked me to do something I don't know yet (`{tool_name}`).")
    
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"⚠️ Missing argument. Usage: `!{ctx.command.name} <args>`")
        return
    log.error(f"Command error: {error}")
    await ctx.send(f"⚠️ Something went wrong: `{error}`")


@bot.event
async def on_reaction_add(reaction, user):
    global STATS
    if user == bot.user:
        return

    msg_id = reaction.message.id

    # Bail out only if message is in NONE of the approval dicts.
    if (
        msg_id not in PENDING_APPROVALS
        and msg_id not in PENDING_STORYBOARD_APPROVALS
        and msg_id not in PENDING_VIDEO_APPROVALS
    ):
        return

    # Skip script handler if msg isn't a pending script approval
    if msg_id not in PENDING_APPROVALS:
        entry = None
        script = None
        owner_id = None
    else:
        entry = PENDING_APPROVALS[msg_id]
        script = entry["script"]
        owner_id = entry["owner_id"]

        if user.id != owner_id and not user.guild_permissions.administrator:
            return

    emoji = str(reaction.emoji)
    channel = reaction.message.channel
    script_id = (script.get("_id") or script.get("script_id")) if script else None

    if script is not None:
        if emoji == "✅":
            STATS["approved"] += 1; save_stats(STATS)
            rev_num = script.get("revision_number", 1)
            note = f" after {rev_num - 1} revision(s)" if rev_num > 1 else ""
            await channel.send(
                f"✅ **Script `{script_id}` APPROVED** by {user.mention}{note}.\n"
                f"🎨 Auto-starting storyboard generation..."
            )
            log.info(f"Script {script_id} approved by {user}. Auto-triggering storyboard.")
            approved_dir = OUTPUTS_DIR.parent / "approved_scripts"
            approved_dir.mkdir(parents=True, exist_ok=True)
            (approved_dir / f"{script_id}.approved").write_text(
                f"approved_by={user}\napproved_at={datetime.now(timezone.utc).isoformat()}\n",
                encoding="utf-8",
            )
            pf.remove(script_id)
            del PENDING_APPROVALS[msg_id]

            # Polish prompts with Qwen, then auto-trigger storyboard
            storyboards_channel = get_channel_by_name(channel.guild, "storyboards") or channel
            asyncio.create_task(_polish_then_storyboard(
                channel, storyboards_channel, script, user.id, user.mention
            ))

        elif emoji == "❌":
            STATS["rejected"] += 1; save_stats(STATS)
            del PENDING_APPROVALS[msg_id]
            asyncio.create_task(_run_revision_loop(channel, script, user))

        elif emoji == "⏹️":
            STATS["loops_stopped"] += 1; save_stats(STATS)
            await channel.send(
                f"🛑 Story `{script_id}` stopped by {user.mention}. No further revisions."
            )
            pf.remove(script_id)
            del PENDING_APPROVALS[msg_id]
        return

    # ------- Storyboard approvals -------
    if msg_id in PENDING_STORYBOARD_APPROVALS:
        sb_entry = PENDING_STORYBOARD_APPROVALS[msg_id]
        if user.id != sb_entry["owner_id"] and not user.guild_permissions.administrator:
            return
        sb_script_id = sb_entry["script_id"]
        sb_channel = reaction.message.channel

        if emoji == "✅":
            STATS["storyboards_approved"] = STATS.get("storyboards_approved", 0) + 1
            save_stats(STATS)
            await sb_channel.send(
                f"✅ **Storyboard `{sb_script_id}` APPROVED** by {user.mention}.\n"
                f"🎥 Auto-starting video clip generation..."
            )
            APPROVED_STORYBOARDS_DIR = OUTPUTS_DIR.parent / "approved_storyboards"
            APPROVED_STORYBOARDS_DIR.mkdir(parents=True, exist_ok=True)
            (APPROVED_STORYBOARDS_DIR / f"{sb_script_id}.approved").write_text(
                f"approved_by={user}\napproved_at={datetime.now(timezone.utc).isoformat()}\n",
                encoding="utf-8",
            )
            del PENDING_STORYBOARD_APPROVALS[msg_id]

            # Auto-trigger video generation in #videos channel
            videos_channel = get_channel_by_name(sb_channel.guild, "videos") or sb_channel
            asyncio.create_task(_run_video_pipeline(
                videos_channel, sb_script_id, user.id, user.mention, is_auto=True
            ))

        elif emoji == "❌":
            STATS["storyboards_rejected"] = STATS.get("storyboards_rejected", 0) + 1
            save_stats(STATS)
            del PENDING_STORYBOARD_APPROVALS[msg_id]
            asyncio.create_task(_run_storyboard_revision_loop(sb_channel, sb_script_id, user))

        elif emoji == "⏹️":
            await sb_channel.send(
                f"🛑 Storyboard `{sb_script_id}` stopped by {user.mention}."
            )
            del PENDING_STORYBOARD_APPROVALS[msg_id]

        elif emoji == "💾":
            await sb_channel.send(
                f"💾 Storyboard `{sb_script_id}` paused. "
                f"Resume by rejecting again or use `!regen_shot {sb_script_id} <shot_num>`."
            )
            del PENDING_STORYBOARD_APPROVALS[msg_id]
        return

    # ------- Video clip approvals -------
    if msg_id in PENDING_VIDEO_APPROVALS:
        v_entry = PENDING_VIDEO_APPROVALS[msg_id]
        if user.id != v_entry["owner_id"] and not user.guild_permissions.administrator:
            return
        v_script_id = v_entry["script_id"]
        v_channel = reaction.message.channel

        if emoji == "✅":
            STATS["videos_approved"] = STATS.get("videos_approved", 0) + 1
            save_stats(STATS)
            await v_channel.send(
                f"✅ **Video clips for `{v_script_id}` APPROVED** by {user.mention}."
            )
            APPROVED_VIDEOS_DIR = OUTPUTS_DIR.parent / "approved_videos"
            APPROVED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
            (APPROVED_VIDEOS_DIR / f"{v_script_id}.approved").write_text(
                f"approved_by={user}\napproved_at={datetime.now(timezone.utc).isoformat()}\n",
                encoding="utf-8",
            )
            del PENDING_VIDEO_APPROVALS[msg_id]

            # ── Auto-upscale: replaces 480p originals with 4x upscaled versions ──
            from modules import upscaler as up
            await v_channel.send(
                f"📈 Auto-upscaling clips for `{v_script_id}` (~1-2 min/shot)..."
            )
            try:
                result = await asyncio.to_thread(up.upscale_storyboard_videos, v_script_id, None)
                await v_channel.send(up.format_upscale_summary(result))

                # ── Auto-assemble: stitch upscaled clips into final 9x16 + 16x9 ──
                from modules import assembly as asm
                await v_channel.send(f"🎬 Auto-assembling final video for `{v_script_id}`...")
                asm_result = None
                try:
                    asm_result = await asyncio.to_thread(asm.assemble_final, v_script_id, None)
                except Exception as e:
                    log.exception("Auto-assembly failed")
                    await v_channel.send(
                        f"⚠️ Auto-assembly failed: `{e}`\n"
                        f"Run manually with `!assemble {v_script_id}` once fixed."
                    )

                if asm_result:
                    summary = (
                        f"✅ **Assembly complete** — {asm_result['shot_count']} shots, "
                        f"{asm_result['total_duration_sec']:.1f}s · ⏱ assembly {asm_result.get('wall_sec', 0):.0f}s\n"
                        f"📂 `{asm_result['9x16'].name}` (Shorts)\n"
                        f"📂 `{asm_result['16x9'].name}` (Horizontal)"
                    )
                    await v_channel.send(summary)

                    # Upload attempt: catch oversize uploads independently
                    for key in ("9x16", "16x9"):
                        f = asm_result[key]
                        size_mb = f.stat().st_size / (1024 * 1024)
                        # Discord limit is 10 MB on unboosted servers, 25 MB on tier-1+
                        # Safer to skip preview for anything over 9 MB and just show path
                        if size_mb <= 9:
                            try:
                                await v_channel.send(file=discord.File(str(f)))
                            except discord.HTTPException as ue:
                                await v_channel.send(
                                    f"📁 `{f.name}` ({size_mb:.1f} MB) — too large to upload "
                                    f"({ue.status}). File saved at:\n`{f}`"
                                )
                        else:
                            await v_channel.send(
                                f"📁 `{f.name}` ({size_mb:.1f} MB) — preview too large. "
                                f"File saved at:\n`{f}`"
                            )
                    await v_channel.send("*Ready for Phase 6C (YouTube upload) — coming next.*")
            except Exception as e:
                log.exception("Auto-upscale failed")
                await v_channel.send(
                    f"⚠️ Auto-upscale failed: `{e}`\n"
                    f"Run manually with `!upscale {v_script_id}` once fixed."
                )

        elif emoji == "❌":
            STATS["videos_rejected"] = STATS.get("videos_rejected", 0) + 1
            save_stats(STATS)
            del PENDING_VIDEO_APPROVALS[msg_id]
            await v_channel.send(
                f"❌ Video clips rejected. Tell me which shots to redo "
                f"(e.g. `!regen_video_shot {v_script_id} 2`)."
            )

        elif emoji == "⏹️":
            await v_channel.send(
                f"🛑 Video pass for `{v_script_id}` stopped by {user.mention}."
            )
            del PENDING_VIDEO_APPROVALS[msg_id]

        elif emoji == "💾":
            await v_channel.send(
                f"💾 Video clips for `{v_script_id}` paused. Resume with "
                f"`!regen_video_shot {v_script_id} <shot_num>`."
            )
            del PENDING_VIDEO_APPROVALS[msg_id]
        return

# ============================================================
# STORYBOARD PIPELINE HELPER
# ============================================================

@_gpu_job("prompt generation")
async def _polish_then_storyboard(
    status_channel, storyboards_channel, script: dict,
    owner_id: int, owner_mention: str
):
    """Approval-gated prompt flow:
      1. Health gate (ComfyUI + Ollama alive)
      2. Generate image + motion prompts for every shot via Qwen
      3. Post per-shot Discord embeds with Edit/Reseed/Approve-Shot buttons
      4. User reviews, edits, locks each shot, then clicks Approve-All
      5. Approve-All → storyboard pipeline (which reads approved prompts + seeds)

    Shot tailor + polish were removed from the auto pipeline — the user now
    approves prompts directly. Polisher still available via !repolish for
    manual use.
    """
    script_id = script.get("_id") or script.get("script_id")

    # Pre-pipeline health gate — fail fast with a clear message rather than
    # dying mid-render with a cryptic ConnectionError.
    comfy_up, comfy_msg = gpu_utils.check_comfyui_alive()
    ollama_up, ollama_msg = gpu_utils.check_ollama_alive()
    if not (comfy_up and ollama_up):
        await status_channel.send(
            f"❌ Pipeline blocked — required service down:\n"
            f"• ComfyUI: `{comfy_msg}`\n"
            f"• Ollama:  `{ollama_msg}`\n"
            f"Start the missing service and re-approve the script."
        )
        return

    # Define what happens on Approve-All — kick off storyboard generation.
    async def _on_approve_all(interaction):
        await storyboards_channel.send(
            f"🚀 **Prompts approved for `{script_id}`** by {interaction.user.mention} — "
            f"starting storyboard render…"
        )
        asyncio.create_task(_run_storyboard_pipeline(
            storyboards_channel, script_id, owner_id, owner_mention, is_auto=True
        ))

    async def _on_cancel(interaction):
        await storyboards_channel.send(
            f"🛑 Prompt approval for `{script_id}` cancelled by {interaction.user.mention}."
        )

    try:
        await pap.post_approval_ui(
            channel=storyboards_channel,
            script=script,
            owner_id=owner_id,
            on_approve_all=_on_approve_all,
            on_cancel=_on_cancel,
        )
    except Exception as e:
        log.exception(f"Prompt approval setup failed: {e}")
        await status_channel.send(
            f"❌ Could not post prompt approvals for `{script_id}`: `{e}`\n"
            f"Falling back to legacy direct storyboard render."
        )
        asyncio.create_task(_run_storyboard_pipeline(
            storyboards_channel, script_id, owner_id, owner_mention, is_auto=True
        ))

@_gpu_job("storyboard render")
async def _run_storyboard_pipeline(
    channel, script_id: str, owner_id: int, owner_mention: str, *, is_auto: bool = False
):
    """Generate a storyboard and register its approval message."""
    global STATS
    result_info = await sw.generate_and_post_storyboard(
        channel=channel,
        script_id=script_id,
        owner_id=owner_id,
        requested_by_mention=owner_mention,
        is_auto=is_auto,
    )
    if result_info is None:
        return  # failure was already posted inside the workflow
    # Register for approval tracking
    PENDING_STORYBOARD_APPROVALS[result_info["approval_msg_id"]] = {
        "script_id": result_info["script_id"],
        "owner_id": result_info["owner_id"],
    }
    STATS["storyboards_generated"] = STATS.get("storyboards_generated", 0) + 1
    save_stats(STATS)


# ============================================================
# VIDEO PIPELINE HELPER
# ============================================================

@_gpu_job("video render")
async def _run_video_pipeline(
    channel,
    script_id: str,
    owner_id: int,
    requested_by_mention: str,
    is_auto: bool = False,
):
    """Run video clip generation and register the approval message."""
    global STATS
    result = await vw.generate_and_post_video(
        channel=channel,
        script_id=script_id,
        owner_id=owner_id,
        requested_by_mention=requested_by_mention,
        is_auto=is_auto,
    )
    if result and result.get("approval_msg_id"):
        PENDING_VIDEO_APPROVALS[result["approval_msg_id"]] = {
            "script_id": script_id,
            "owner_id": owner_id,
            "channel_id": channel.id,
        }
        STATS["videos_generated"] = STATS.get("videos_generated", 0) + 1
        save_stats(STATS)


@_gpu_job("video shot regen")
async def _run_video_regen(channel, script_id: str, shot_numbers: list, owner_id: int):
    """Re-render specific video clips and register the approval."""
    # Wipe stale video approval messages so the new one lands at the bottom
    await _delete_pending_video_approvals(channel, script_id)

    result = await vw.regenerate_video_shots(
        channel=channel,
        script_id=script_id,
        shot_numbers=shot_numbers,
        owner_id=owner_id,
    )
    if result and result.get("approval_msg_id"):
        PENDING_VIDEO_APPROVALS[result["approval_msg_id"]] = {
            "script_id": script_id,
            "owner_id": owner_id,
            "channel_id": channel.id,
        }


async def _run_storyboard_revision_loop(channel, script_id: str, owner):
    """Ask for feedback, parse shot numbers, regenerate those shots."""
    prompt_msg = await channel.send(
        f"{owner.mention} ❌ Storyboard `{script_id}` rejected.\n"
        f"**What should I change?** (reply within 5 min, or type `stop` to cancel)\n\n"
        f"💡 Tips:\n"
        f"• **Surgical:** \"redo shot 2\" → only shot 2 regenerates\n"
        f"• **Multiple:** \"redo shot 1 and shot 3\"\n"
        f"• **All:** \"regenerate everything\" or \"redo all shots\"\n"
        f"• **Pause:** type `pause` to save for later\n"
        f"• **Cancel:** type `stop`"
    )

    def check(m):
        return m.author.id == owner.id and m.channel.id == channel.id

    try:
        feedback_msg = await bot.wait_for("message", check=check, timeout=FEEDBACK_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        await channel.send(f"⏱️ No feedback in {FEEDBACK_TIMEOUT_SEC // 60} min. Storyboard loop paused.")
        return

    notes = feedback_msg.content.strip().lower()
    if notes in {"stop", "cancel", "abort"}:
        await channel.send(f"🛑 Storyboard `{script_id}` abandoned.")
        return
    if notes in {"pause", "save", "later"}:
        await channel.send(f"💾 Storyboard `{script_id}` feedback paused. Resume later with a new `!regen_shot` command.")
        return

    # Parse which shots to redo (organic shot count — load from script)
    try:
        _script = sw.load_script(script_id)
        total_shots = len(_script.get("shots", []))
    except Exception:
        total_shots = None

    shot_nums = sw.extract_shot_numbers(notes, total_shots=total_shots)
    if not shot_nums:
        # Assume "redo all"
        if any(kw in notes for kw in ["all", "everything", "whole", "entire"]):
            shot_nums = list(range(1, (total_shots or 4) + 1))
        else:
            await channel.send(
                "⚠️ I couldn't detect which shot(s) to redo. "
                "Try: *\"redo shot 2\"* or *\"redo all shots\"*. Please reject again with clearer notes."
            )
            return

    result_info = await sw.regenerate_shots(
        channel=channel,
        script_id=script_id,
        shot_numbers=shot_nums,
        owner_id=owner.id,
    )
    if result_info:
        PENDING_STORYBOARD_APPROVALS[result_info["approval_msg_id"]] = {
            "script_id": result_info["script_id"],
            "owner_id": result_info["owner_id"],
        }


# ============================================================
# COMMANDS — Basic
# ============================================================

@bot.command(name="hello")
async def cmd_hello(ctx):
    await ctx.send(f"👋 Hello {ctx.author.mention}! Claw Bot at your service.")


@bot.command(name="ping")
async def cmd_ping(ctx):
    await ctx.send(f"🏓 Pong! ({round(bot.latency * 1000)} ms)")


@bot.command(name="status")
async def cmd_status(ctx):
    if BOT_START_TIME is None:
        await ctx.send("Just starting up.")
        return
    uptime = datetime.now(timezone.utc) - BOT_START_TIME
    h, rem = divmod(int(uptime.total_seconds()), 3600)
    m, s   = divmod(rem, 60)
    nj = scheduler.get_job("daily_story")
    next_str = nj.next_run_time.strftime("%Y-%m-%d %H:%M %Z") if nj else "N/A"
    await ctx.send(
        f"📊 **Claw Bot Status**\n"
        f"• Version: `{BOT_VERSION}`\n"
        f"• Uptime: `{h}h {m}m {s}s`\n"
        f"• Pending approvals: `{len(PENDING_APPROVALS)}`\n"
        f"• Paused feedback: `{pf.count()}`\n"
        f"• Next auto-story: `{next_str}`"
    )


# ============================================================
# COMMANDS — Lifecycle (restart / shutdown)
# ============================================================

def _spawn_relaunch():
    """Spawn a detached PowerShell that waits for THIS process to exit, then
    relaunches the bot in a fresh console window. Ollama + ComfyUI keep running
    (untouched) so the restart only reloads the bot's .py code. Zero overlap:
    the relauncher blocks on Wait-Process until our PID dies before starting."""
    import subprocess
    bot_script = Path(__file__).resolve()
    agent_dir = bot_script.parent
    py = sys.executable  # venv python while running
    ps_cmd = (
        f"Wait-Process -Id {os.getpid()} -ErrorAction SilentlyContinue; "
        f"Start-Sleep -Milliseconds 800; "
        f"$env:PYTHONPATH = '{agent_dir}'; "
        f"Set-Location '{PROJECT_ROOT}'; "
        f"& '{py}' '{bot_script}'"
    )
    CREATE_NEW_CONSOLE = 0x00000010
    subprocess.Popen(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
        creationflags=CREATE_NEW_CONSOLE,
        close_fds=True,
    )


async def cmd_restart_bot(ctx):
    """Relaunch the bot to load .py edits. Spawns a fresh console, then exits."""
    await ctx.send("♻️ **Restarting bot…** New console window will open. Back in ~15s.")
    log.info("Restart requested via control panel.")
    try:
        _spawn_relaunch()
    except Exception as e:
        log.exception("Relaunch spawn failed")
        await ctx.send(f"❌ Could not spawn relauncher: `{e}`")
        return
    try:
        await bot.close()
    finally:
        os._exit(0)


async def cmd_shutdown_bot(ctx):
    """Stop the bot completely. Boot again from the launcher (.bat / launch script)."""
    await ctx.send("🛑 **Shutting down.** Restart from the launcher when you need me.")
    log.info("Shutdown requested via control panel.")
    try:
        await bot.close()
    finally:
        os._exit(0)


# ============================================================
# COMMANDS — Generation
# ============================================================

@bot.command(name="generate_script", aliases=["gs", "script"])
async def cmd_generate_script(ctx, *, theme: str = None):
    if theme is None:
        theme = get_random_theme()
        await ctx.send(f"🎲 Using random theme: **{theme}**")

    theme_blob = {"title": "", "theme": theme, "setting": "", "moral": "",
                  "characters": [], "shots": []}
    is_safe, blocked, _ = check_safety(theme_blob)
    if not is_safe:
        await ctx.send(
            f"🚨 Theme blocked. Unsafe words: `{', '.join(blocked)}`.\n"
            f"Try `!suggest_theme` for ideas."
        )
        return

    target = get_channel_by_name(ctx.guild, "scripts") or ctx.channel
    if ctx.channel.id != target.id:
        await send_transient(ctx.channel, f"📝 Generating... posting in {target.mention}.", 10)

    await _generate_and_post(
        target, theme,
        requested_by_id=ctx.author.id,
        requested_by_mention=ctx.author.mention,
    )


@bot.command(name="today_script", aliases=["today"])
async def cmd_today_script(ctx):
    theme = get_theme_of_the_day()
    await ctx.send(
        f"🌅 **Today's theme:** {theme}\n"
        f"(Auto-generates daily at {DAILY_GEN_HOUR:02d}:00 IST, "
        f"or: `!generate_script {theme}`)"
    )


@bot.command(name="suggest_theme", aliases=["suggest"])
async def cmd_suggest_theme(ctx):
    theme = get_random_theme()
    await send_transient(
        ctx.channel,
        f"🎲 **Theme suggestion:** {theme}\nUse: `!generate_script {theme}`",
        30,
    )


@bot.command(name="list_scripts", aliases=["ls"])
async def cmd_list_scripts(ctx):
    if not OUTPUTS_DIR.exists():
        await ctx.send("📭 No scripts yet.")
        return
    files = sorted(OUTPUTS_DIR.glob("script_*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)[:10]
    if not files:
        await ctx.send("📭 No scripts yet.")
        return
    lines = ["**📚 Recent Scripts:**"]
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            rev = data.get("revision_number", 1)
            rev_tag = f" [v{rev}]" if rev > 1 else ""
            lines.append(f"• `{data['script_id']}`{rev_tag} — **{data['title']}** ({data['theme']})")
        except Exception:
            lines.append(f"• `{f.stem}` — *(unreadable)*")
    await ctx.send("\n".join(lines))


@bot.command(name="show_script", aliases=["show"])
async def cmd_show_script(ctx, script_id: str):
    target_file = OUTPUTS_DIR / f"script_{script_id}.json"
    if not target_file.exists():
        await ctx.send(f"❌ No script `{script_id}`. Try `!list_scripts`.")
        return
    try:
        data = json.loads(target_file.read_text(encoding="utf-8"))
    except Exception as e:
        await ctx.send(f"❌ Parse error: `{e}`")
        return
    await send_long_message(ctx.channel, format_for_discord(data))


@bot.command(name="rewrite_narration", aliases=["rewrite_narr", "narr_rewrite"])
async def cmd_rewrite_narration(ctx, script_id: str = None, shot_num=None,
                                *instr, instruction: str = None):
    """AI-rewrite one shot's narration. `!rewrite_narration <id> <shot> [hint...]`"""
    if not script_id or shot_num is None:
        await ctx.send("Usage: `!rewrite_narration <script_id> <shot_num> [instruction]`")
        return
    try:
        shot_n = int(str(shot_num).strip())
    except ValueError:
        await ctx.send(f"❌ Shot number must be an integer, got `{shot_num}`.")
        return
    hint = (instruction or " ".join(instr) or "").strip()

    path = OUTPUTS_DIR / f"script_{script_id}.json"
    if not path.exists():
        await ctx.send(f"❌ No script `{script_id}`. Try `!list_scripts`.")
        return
    try:
        script = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        await ctx.send(f"❌ Parse error: `{e}`")
        return
    shot = next((s for s in script.get("shots", [])
                 if s.get("shot_number") == shot_n), None)
    if not shot:
        await ctx.send(f"❌ Shot `{shot_n}` not in script `{script_id}`.")
        return

    old = (shot.get("narration") or "").strip()
    status = await ctx.send(f"🪄 Rewriting narration for shot `{shot_n}`…")
    try:
        new = await asyncio.to_thread(
            rewrite_narration, old, hint,
            shot=shot, title=script.get("title", ""), moral=script.get("moral", ""),
        )
    except Exception as e:
        await status.edit(content=f"❌ Rewrite failed: `{e}`")
        return
    if not new or new == old:
        await status.edit(content="⚠️ AI returned no change.")
        return
    shot["narration"] = new
    try:
        atomic_write_json(path, script)
    except Exception as e:
        await status.edit(content=f"❌ Could not save: `{e}`")
        return
    await status.edit(
        content=(f"✅ **Shot {shot_n} narration rewritten** (`{script_id}`)\n"
                 f"**Before:** {old or '_(empty)_'}\n"
                 f"**After:** {new}\n"
                 f"_Re-render this shot's video to hear it._")
    )


# ============================================================
# COMMANDS — Pending / Resume / Drop
# ============================================================

@bot.command(name="pending")
async def cmd_pending(ctx):
    """Show all scripts currently awaiting your feedback."""
    entries = pf.list_all()
    if not entries:
        await ctx.send("📭 No scripts awaiting feedback.")
        return

    lines = [f"**💾 Paused Feedback ({len(entries)}):**"]
    for e in entries:
        paused_when = e["paused_at"].split("T")[0]   # just the date part
        reason_tag = " *(auto-paused)*" if e["reason"] == "timeout" else ""
        lines.append(
            f"• `{e['script_id']}` — **{e['title']}** "
            f"*(paused {paused_when}){reason_tag}*"
        )
    lines.append("")
    lines.append("Resume any with: `!resume_feedback <script_id>`")
    lines.append("Discard any with: `!drop_feedback <script_id>`")
    await ctx.send("\n".join(lines))


@bot.command(name="resume_feedback", aliases=["resume"])
async def cmd_resume_feedback(ctx, script_id: str):
    """Reopen the feedback prompt for a paused script."""
    global STATS
    entry = pf.get(script_id)
    if entry is None:
        await ctx.send(f"❌ No paused feedback for `{script_id}`. See `!pending`.")
        return

    # Only the original owner (or admin) can resume
    if ctx.author.id != entry["owner_id"] and not ctx.author.guild_permissions.administrator:
        await ctx.send("⚠️ Only the original requester can resume this feedback.")
        return

    # Remove from pending store and restart the revision loop here
    pf.remove(script_id)
    STATS["resumed"] += 1
    save_stats(STATS)

    await ctx.send(
        f"🔁 Resuming feedback for `{script_id}`..."
    )
    # Re-post the script first so the user has context
    await send_long_message(ctx.channel, format_for_discord(entry["script"]))
    asyncio.create_task(_run_revision_loop(ctx.channel, entry["script"], ctx.author))


@bot.command(name="drop_feedback", aliases=["drop"])
async def cmd_drop_feedback(ctx, script_id: str):
    """Permanently discard a paused script."""
    entry = pf.get(script_id)
    if entry is None:
        await ctx.send(f"❌ No paused feedback for `{script_id}`.")
        return
    if ctx.author.id != entry["owner_id"] and not ctx.author.guild_permissions.administrator:
        await ctx.send("⚠️ Only the original requester can drop this.")
        return
    pf.remove(script_id)
    await ctx.send(f"🗑️ Dropped paused feedback for `{script_id}`.")


# ============================================================
# COMMANDS — Stats
# ============================================================

@bot.command(name="stats")
async def cmd_stats(ctx):
    total = STATS["generated"]; appr = STATS["approved"]; rej = STATS["rejected"]
    blk = STATS["safety_blocked"]; revs = STATS["revisions"]; stops = STATS["loops_stopped"]
    paused = STATS["paused"]; resumed = STATS["resumed"]
    pending_approval = len(PENDING_APPROVALS)
    pending_feedback = pf.count()
    rate = (appr / (appr + rej) * 100) if (appr + rej) > 0 else 0
    await ctx.send(
        f"📊 **Claw Bot All-Time Stats**\n"
        f"• Total generations: `{total}` (including `{revs}` revisions)\n"
        f"• ✅ Approved: `{appr}`\n"
        f"• ❌ Rejected: `{rej}`\n"
        f"• 💾 Paused: `{paused}` | 🔁 Resumed: `{resumed}`\n"
        f"• 🛑 Loops stopped: `{stops}`\n"
        f"• 🚨 Safety-blocked: `{blk}`\n"
        f"• ⏳ Pending approval: `{pending_approval}` | Paused feedback: `{pending_feedback}`\n"
        f"• 📈 Approval rate: `{rate:.1f}%`"
    )


# ============================================================
# MAIN
# ============================================================

# ============================================================
# COMMANDS — Help & Settings
# ============================================================

@bot.command(name="commands", aliases=["help_me", "h"])
async def cmd_help(ctx):
    """Full command reference with examples + active settings."""
    from modules import script_generator as sg_mod
    from modules import runtime_settings as rs_mod

    available_styles = sg_mod.get_available_style_ids()
    active_style = rs_mod.get_effective_style()
    active_ratio = rs_mod.get_effective_aspect_ratio()
    active_steps = rs_mod.get_effective_steps()
    active_cfg = rs_mod.get_effective_cfg()

    try:
        img_cfg = model_registry.get_active("image_backend")
        active_backend = img_cfg.get("_id", "unknown")
    except Exception:
        active_backend = "unknown"

    e = discord.Embed(
        title="🤖 Claw Bot — Command Reference",
        description=(
            f"**Active settings:**\n"
            f"🎨 Style: `{active_style}`  ·  🔌 Model: `{active_backend}`\n"
            f"📐 Aspect: `{active_ratio}`  ·  ⚙️ Steps: `{active_steps}`  ·  🎲 CFG: `{active_cfg}`\n"
            f"_(Overrides persist until you change them again.)_"
        ),
        color=discord.Color.blue(),
    )

    e.add_field(
        name="📝 Script Generation",
        value=(
            "`!generate_script <theme>` — new story from a theme\n"
            "  ↳ *Example:* `!generate_script a shy rabbit learning to share`\n"
            "`!today_script` — today's theme of the day\n"
            "`!suggest_theme` — random theme suggestion\n"
            "`!list_scripts` — recent scripts with IDs\n"
            "`!show_script <id>` — display a specific script"
        ),
        inline=False,
    )

    e.add_field(
        name="🎬 Storyboard",
        value=(
            "`!generate_storyboard <id>` or `!gsb <id>` — render images for a script\n"
            "  ↳ *Example:* `!gsb 20260420_213414`\n"
            "`!list_storyboards` or `!lsb` — recent storyboards\n"
            "`!regen_shot <id> <shot#>` — re-render a single shot (same prompt)\n"
            "  ↳ *Example:* `!regen_shot 20260420_213414 2`\n"
            "`!edit_prompts <id> [shot#…]` — EDIT a shot's prompt then regen\n"
            "  ↳ *Example:* `!edit_prompts 20260420_213414 2` (fix a bad shot)"
        ),
        inline=False,
    )

    e.add_field(
        name="🎨 Style & Look",
        value=(
            f"`!list_styles` — all {len(available_styles)} styles with descriptions\n"
            "`!set_style <name>` — force a default style\n"
            "  ↳ *Example:* `!set_style anime`\n"
            "`!set_style auto` — let LLM pick (default)\n"
            f"  ↳ Available: " + ", ".join(f"`{s}`" for s in available_styles)
        ),
        inline=False,
    )

    e.add_field(
        name="⚙️ Image Tuning",
        value=(
            "`!set_resolution <ratio>` — 16:9, 9:16, or 1:1\n"
            "  ↳ *Example:* `!set_resolution 9:16` (vertical for Shorts)\n"
            "`!set_video_resolution <p>` — video quality: 480p/720p/1080p/reset\n"
            "  ↳ *Example:* `!set_video_resolution 720p` (sharper, less noise)\n"
            "`!set_steps <n>` — override sampler steps (higher = better, slower)\n"
            "  ↳ *Example:* `!set_steps 12`\n"
            "`!set_cfg <n>` — guidance scale (1.0-7.0 typical)\n"
            "  ↳ *Example:* `!set_cfg 1.5`\n"
            "`!reset_settings` — clear all overrides"
        ),
        inline=False,
    )

    e.add_field(
        name="🔌 Model Control",
        value=(
            "`!switch_model` — show active and available image models\n"
            "`!switch_model <id>` — change image model\n"
            "  ↳ *Example:* `!switch_model comfyui_zimage_turbo`\n"
            "`!current_settings` — full active config + stats"
        ),
        inline=False,
    )

    e.add_field(
        name="📊 System",
        value=(
            "`!status` — bot status summary\n"
            "`!ping` — is the bot alive?\n"
            "`!stats` — session statistics\n"
            "`!pending` — paused feedback loops\n"
            "`!resume_feedback <id>` — resume a paused loop\n"
            "`!drop_feedback <id>` — discard paused feedback"
        ),
        inline=False,
    )

    e.add_field(
        name="🧭 Reaction Controls",
        value=(
            "On any script or storyboard approval message:\n"
            "✅ Approve  ·  ❌ Reject & revise  ·  ⏹️ Stop  ·  💾 Pause"
        ),
        inline=False,
    )

    e.set_footer(text="📖 Live dashboard in #status · Scripts auto-trigger storyboards on ✅")
    await ctx.send(embed=e)


@bot.command(name="list_styles", aliases=["styles"])
async def cmd_list_styles(ctx):
    """Show all available visual styles."""
    from modules import script_generator as sg_mod
    styles = sg_mod._load_styles().get("available", {})
    default = sg_mod.get_default_style()
    if not styles:
        await ctx.send("⚠️ No styles configured. Check `05_Config/styles.json`.")
        return
    lines = ["**🎨 Available Visual Styles**", ""]
    for sid, info in styles.items():
        marker = " _(default)_" if sid == default else ""
        lines.append(f"**`{sid}`**{marker} — {info.get('display_name', sid)}")
        lines.append(f"   _{info.get('description', '')}_")
        lines.append(f"   💡 Best for: {info.get('best_for', '')}")
        lines.append("")
    lines.append("Scripts auto-pick the right style. You can override via revision feedback: ")
    lines.append("_\"Use anime style instead\"_ or _\"Change to watercolor.\"_")
    await ctx.send("\n".join(lines))

# ============================================================
# COMMANDS — Runtime Tuning
# ============================================================

@bot.command(name="set_style")
async def cmd_set_style(ctx, style_id: str = None):
    """Override the default style (or 'auto' to let LLM pick)."""
    from modules import script_generator as sg_mod
    if style_id is None:
        current = rs.get_style_override() or "auto (LLM decides)"
        await ctx.send(f"🎨 Current style override: `{current}`\nUse `!set_style <name>` or `!set_style auto`.")
        return

    style_id = style_id.lower().strip()
    if style_id == "auto":
        rs.clear_style_override()
        await ctx.send("🎨 Style override cleared. LLM will pick per story.")
        return

    available = sg_mod.get_available_style_ids()
    if style_id not in available:
        await ctx.send(
            f"❌ Unknown style `{style_id}`.\n"
            f"Available: " + ", ".join(f"`{s}`" for s in available) + "\n"
            f"Or `!set_style auto` to let LLM decide."
        )
        return

    rs.set_style_override(style_id)
    await ctx.send(f"✅ Style set to `{style_id}`. Will apply to next generation.")


@bot.command(name="set_resolution", aliases=["set_aspect"])
async def cmd_set_resolution(ctx, aspect: str = None):
    """Set aspect ratio: 16:9, 9:16, or 1:1."""
    valid = {"16:9", "9:16", "1:1"}
    if aspect is None:
        current = rs.get_effective_aspect_ratio()
        await ctx.send(f"📐 Current aspect ratio: `{current}`\nValid options: `16:9`, `9:16`, `1:1`")
        return

    if aspect not in valid:
        await ctx.send(f"❌ Invalid aspect `{aspect}`. Use one of: `16:9`, `9:16`, `1:1`")
        return

    rs.set_resolution_override(aspect)
    await ctx.send(f"✅ Aspect ratio set to `{aspect}`.")


@bot.command(name="set_video_resolution", aliases=["set_video_res", "video_res", "set_vres"])
async def cmd_set_video_resolution(ctx, preset: str = None):
    """Set video generation resolution: 480p, 720p, 1080p, or 'reset'.

    Higher = sharper / less noise, but slower + more VRAM. 1080p I2V is heavy.
    Applies to the next video render. 'reset' = back to the model's own default.
    """
    valid = list(rs.VIDEO_RES_PRESETS.keys())
    if preset is None:
        cur = rs.get_video_resolution_override()
        dims = rs.get_effective_video_resolution()
        label = f"`{cur}` ({dims[0]}x{dims[1]})" if cur and dims else "model default"
        await ctx.send(
            f"📺 Current video resolution: {label}\n"
            f"Options: {', '.join(f'`{p}`' for p in valid)} · `reset`"
        )
        return

    p = preset.strip().lower()
    if p in ("reset", "default", "auto", "clear"):
        rs.clear_video_resolution_override()
        await ctx.send("✅ Video resolution reset to the model's own default.")
        return
    if p not in valid:
        await ctx.send(f"❌ Invalid `{preset}`. Use one of: {', '.join(valid)} · `reset`")
        return

    rs.set_video_resolution_override(p)
    dims = rs.get_effective_video_resolution()
    await ctx.send(
        f"✅ Video resolution set to `{p}` ({dims[0]}x{dims[1]}). "
        f"Applies to the next video render."
    )


@bot.command(name="set_steps")
async def cmd_set_steps(ctx, steps: int = None):
    """Override the sampler step count. Use `!set_steps reset` to clear."""
    if steps is None:
        current = rs.get_effective_steps()
        override = rs.get_steps_override()
        label = "override" if override is not None else "backend default"
        await ctx.send(f"⚙️ Current steps: `{current}` ({label})")
        return

    if steps < 1 or steps > 100:
        await ctx.send(f"⚠️ Steps should be 1-100. (Turbo: 4-8, Base: 20-30 typical.)")
        return

    rs.set_steps_override(steps)
    await ctx.send(f"✅ Steps set to `{steps}`. Will apply to next image generation.")


@bot.command(name="set_cfg")
async def cmd_set_cfg(ctx, cfg: float = None):
    """Override the CFG guidance scale."""
    if cfg is None:
        current = rs.get_effective_cfg()
        override = rs.get_cfg_override()
        label = "override" if override is not None else "backend default"
        await ctx.send(f"🎲 Current CFG: `{current}` ({label})")
        return

    if cfg < 0.1 or cfg > 20.0:
        await ctx.send(f"⚠️ CFG should be 0.1-20.0. (Turbo: 1.0, Base: 4.0-7.0 typical.)")
        return

    rs.set_cfg_override(cfg)
    await ctx.send(f"✅ CFG set to `{cfg}`. Will apply to next image generation.")

@bot.command(name="set_music_mood", aliases=["music_mood", "set_mood"])
async def cmd_set_music_mood(ctx, mood: str = None):
    """Override the music mood. Pass 'reset' to clear and let the LLM pick."""
    from modules import runtime_settings as rs
    from modules.music_generator import VALID_MOODS

    if mood is None:
        current = rs.get_music_mood_override()
        moods_list = " · ".join(VALID_MOODS)
        if current:
            await ctx.send(
                f"🎵 Current music mood override: **{current}**\n"
                f"Available: {moods_list}\n"
                f"Use `!set_music_mood <mood>` to change, or `!set_music_mood reset` to let the LLM pick."
            )
        else:
            await ctx.send(
                f"🎵 No override active — LLM picks per script.\n"
                f"Available: {moods_list}"
            )
        return

    if mood.lower() == "reset":
        rs.clear_music_mood_override()
        await ctx.send("✅ Music mood override cleared. LLM will pick per script.")
        return

    try:
        rs.set_music_mood_override(mood.lower())
        await ctx.send(f"✅ Music mood override set to **{mood.lower()}**.")
    except ValueError as e:
        await ctx.send(f"⚠️ {e}")


@bot.command(name="regenerate_music", aliases=["regen_music", "remix_music"])
async def cmd_regenerate_music(ctx, script_id: str = None):
    """Re-render music for an already-assembled video. Re-mixes both 9x16 and 16x9."""
    if script_id is None:
        await ctx.send("⚠️ Usage: `!regenerate_music <script_id>`")
        return

    from modules import music_generator as mg
    from modules import assembly as asm
    from pathlib import Path

    # Verify the assembled finals exist
    final_9x16 = asm.FINAL_DIR / f"final_{script_id}_9x16.mp4"
    final_16x9 = asm.FINAL_DIR / f"final_{script_id}_16x9.mp4"
    if not final_9x16.exists() or not final_16x9.exists():
        await ctx.send(
            f"❌ Finals not found for `{script_id}`.\n"
            f"Run `!assemble {script_id}` first."
        )
        return

    # Probe total duration via ffprobe
    import subprocess
    try:
        r = subprocess.run(
            [str(asm.FFPROBE_EXE), "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(final_9x16)],
            capture_output=True, text=True, check=True,
        )
        total_dur = float(r.stdout.strip())
    except Exception as e:
        await ctx.send(f"❌ Could not probe duration: `{e}`")
        return

    mood = mg.get_effective_mood()
    status_msg = await ctx.send(
        f"🎵 Regenerating music for `{script_id}`\n"
        f"Mood: **{mood}**  ·  Duration: {total_dur:.1f}s\n"
        f"⏳ Loading MusicGen..."
    )

    bot_loop = asyncio.get_running_loop()
    last_update = {"t": 0.0}

    from modules.progress_bar import render_indeterminate
    last_update["tick"] = 0

    def progress(text: str):
        import time as _t
        now = _t.time()
        if now - last_update["t"] < 2.0:
            return
        last_update["t"] = now
        last_update["tick"] += 1
        bar = render_indeterminate(last_update["tick"])
        asyncio.run_coroutine_threadsafe(
            status_msg.edit(content=f"🎵 **Generating music** for `{script_id}`\n{bar} · {text}"),
            bot_loop
        )

    try:
        # Generate new track
        music_track = await asyncio.to_thread(
            mg.generate_music, mood, total_dur, script_id, progress
        )
        if music_track is None or not music_track.exists():
            await ctx.send("❌ Music generation failed — see logs.")
            return

        # Re-mix both finals
        await status_msg.edit(content=f"🎵 Mixing music into both finals...")
        for aspect_key, final_path in (("9x16", final_9x16), ("16x9", final_16x9)):
            tmp_mixed = final_path.with_suffix(".mixed.mp4")
            await asyncio.to_thread(
                asm._mix_music_under_narration,
                final_path, music_track, tmp_mixed, -12.0,
            )
            final_path.unlink(missing_ok=True)
            tmp_mixed.rename(final_path)

        await ctx.send(
            f"✅ **Music regenerated** for `{script_id}` (mood: **{mood}**)\n"
            f"📂 `{final_9x16.name}` · `{final_16x9.name}`"
        )
        # Attach if under Discord limit
        for f in (final_9x16, final_16x9):
            size_mb = f.stat().st_size / (1024 * 1024)
            if size_mb <= 9:
                try:
                    await ctx.send(file=discord.File(str(f)))
                except discord.HTTPException as ue:
                    await ctx.send(
                        f"📁 `{f.name}` ({size_mb:.1f} MB) — too large to upload. "
                        f"File saved at:\n`{f}`"
                    )
            else:
                await ctx.send(
                    f"📁 `{f.name}` ({size_mb:.1f} MB) — preview too large. "
                    f"File saved at:\n`{f}`"
                )
    except Exception as e:
        log.exception("Regenerate music failed")
        await ctx.send(f"❌ Regeneration failed: `{e}`")
        
@bot.command(name="set_voice")
async def cmd_set_voice(ctx, voice_id: str = None):
    """Override Kokoro narration voice. Use `!list_voices` to see options."""
    if voice_id is None:
        current = rs.get_effective_voice()
        override = rs.get_voice_override()
        label = "override" if override is not None else "default"
        await ctx.send(f"🎙️ Current voice: `{current}` ({label})")
        return

    if voice_id.lower() == "reset":
        rs.clear_voice_override()
        await ctx.send(f"🔄 Voice override cleared. Reverting to default (`af_heart`).")
        return

    rs.set_voice_override(voice_id)
    await ctx.send(f"✅ Voice set to `{voice_id}`. Will apply to next clip generation.")


@bot.command(name="list_voices", aliases=["voices"])
async def cmd_list_voices(ctx):
    """List available Kokoro TTS voices."""
    voices = {
        "af_heart":   "🇺🇸 American female · warm, child-friendly (default)",
        "af_bella":   "🇺🇸 American female · soft, gentle",
        "af_nicole":  "🇺🇸 American female · whispered, intimate",
        "af_sky":     "🇺🇸 American female · bright, upbeat",
        "am_adam":    "🇺🇸 American male · clear, neutral",
        "am_michael": "🇺🇸 American male · warm, confident",
        "bf_emma":    "🇬🇧 British female · natural",
        "bf_isabella":"🇬🇧 British female · warm",
        "bm_george":  "🇬🇧 British male · classic",
        "bm_lewis":   "🇬🇧 British male · deep",
    }
    embed = discord.Embed(
        title="🎙️ Available Kokoro Voices",
        description="Use `!set_voice <voice_id>` to switch.",
        color=discord.Color.blue(),
    )
    for vid, desc in voices.items():
        embed.add_field(name=vid, value=desc, inline=False)
    embed.set_footer(text=f"Current: {rs.get_effective_voice()}")
    await ctx.send(embed=embed)


@bot.command(name="set_clip_length")
async def cmd_set_clip_length(ctx, seconds: float = None):
    """Override default clip length in seconds. Use `!set_clip_length reset` to clear."""
    if seconds is None:
        override = rs.get_clip_length_override()
        if override is not None:
            await ctx.send(f"⏱️ Current clip length override: `{override}s`")
        else:
            await ctx.send(
                "⏱️ No clip length override. Each clip matches its narration "
                "(Strategy A: TTS-first sync)."
            )
        return

    if isinstance(seconds, str) and seconds.lower() == "reset":
        rs.clear_clip_length_override()
        await ctx.send("🔄 Clip length override cleared. Length now matches narration.")
        return

    if seconds < 1.0 or seconds > 30.0:
        await ctx.send(f"⚠️ Clip length should be 1.0–30.0 seconds.")
        return

    rs.set_clip_length_override(seconds)
    await ctx.send(f"✅ Clip length set to `{seconds}s`. Will apply to next clip generation.")


@bot.command(name="set_sync_mode")
async def cmd_set_sync_mode(ctx, mode: str = None):
    """Set audio-video sync strategy. `strict` matches exactly; `loose` adds buffer."""
    if mode is None:
        current = rs.get_effective_sync_mode()
        override = rs.get_sync_mode_override()
        label = "override" if override is not None else "default"
        await ctx.send(
            f"🎚️ Current sync mode: `{current}` ({label})\n"
            f"• `strict` — video frames = exact narration length\n"
            f"• `loose` — video frames = narration + 0.5s buffer"
        )
        return

    mode = mode.lower()
    if mode == "reset":
        rs.clear_sync_mode_override()
        await ctx.send("🔄 Sync mode override cleared. Reverting to default (`strict`).")
        return

    if mode not in ("strict", "loose"):
        await ctx.send(f"⚠️ Sync mode must be `strict` or `loose`.")
        return

    rs.set_sync_mode_override(mode)
    await ctx.send(f"✅ Sync mode set to `{mode}`.")


@bot.command(name="set_transition", aliases=["transition", "set_xfade"])
async def cmd_set_transition(ctx, mode: str = None):
    """Set how shots join in the final video. `crossfade` (0.3s dissolve, silent-
    padded so narration never overlaps) or `cut` (immediate hard cut, full length)."""
    if mode is None:
        current = rs.get_effective_transition_mode()
        override = rs.get_transition_mode_override()
        label = "override" if override is not None else "default"
        await ctx.send(
            f"🎞️ Current transition: `{current}` ({label})\n"
            f"• `crossfade` — 0.3s dissolve; each shot gets a silent tail so "
            f"narration never mixes into the next shot\n"
            f"• `cut` — immediate hard cut at each clip's full length; zero "
            f"narration loss"
        )
        return

    mode = mode.lower()
    if mode == "reset":
        rs.clear_transition_mode_override()
        await ctx.send("🔄 Transition override cleared. Reverting to default (`crossfade`).")
        return

    if mode not in ("crossfade", "cut"):
        await ctx.send("⚠️ Transition must be `crossfade` or `cut`.")
        return

    rs.set_transition_mode_override(mode)
    await ctx.send(f"✅ Transition set to `{mode}`.")


@bot.command(name="set_upscale", aliases=["upscale_toggle"])
async def cmd_set_upscale(ctx, value: str = None):
    """Enable/disable auto-upscale in the pipeline. Usage: !set_upscale on|off"""
    if value is None:
        current = rs.get_upscale_enabled()
        await ctx.send(
            f"📈 Auto-upscale: **{'ON' if current else 'OFF'}**\n"
            f"Use `!set_upscale on` or `!set_upscale off` to change."
        )
        return
    v = value.strip().lower()
    if v in ("on", "true", "yes", "1", "enable", "enabled"):
        rs.set_upscale_enabled(True)
        await ctx.send("📈 Upscale **ENABLED**. High-res render will run after approval.")
    elif v in ("off", "false", "no", "0", "disable", "disabled"):
        rs.set_upscale_enabled(False)
        await ctx.send("📉 Upscale **DISABLED**. Preview-quality only.")
    else:
        await ctx.send(f"⚠️ Invalid value `{value}`. Use `on` or `off`.")


@bot.command(name="set_reference_mode", aliases=["refmode", "set_ref"])
async def cmd_set_reference_mode(ctx, value: str = None):
    """Toggle cast-sheet character reference for storyboards. Usage: !set_reference_mode on|off"""
    if value is None:
        cur = rs.get_reference_mode_enabled()
        await ctx.send(
            f"🎯 Character reference mode: **{'ON' if cur else 'OFF'}**\n"
            f"ON = a cast sheet anchors every shot to the same character look.\n"
            f"Use `!set_reference_mode on` or `!set_reference_mode off`."
        )
        return
    v = value.strip().lower()
    if v in ("on", "true", "yes", "1", "enable", "enabled"):
        rs.set_reference_mode_enabled(True)
        await ctx.send("🎯 Reference mode **ON** — storyboards will anchor to a cast sheet for consistency.")
    elif v in ("off", "false", "no", "0", "disable", "disabled"):
        rs.set_reference_mode_enabled(False)
        await ctx.send("🎯 Reference mode **OFF** — back to prompt-only (locked tokens).")
    else:
        await ctx.send("⚠️ Use `on` or `off`.")


@bot.command(name="reset_settings")
async def cmd_reset_settings(ctx):
    """Clear all runtime overrides, revert to backend defaults + LLM choices."""
    rs.clear_all_overrides()
    await ctx.send(
        "🔄 All runtime overrides cleared.\n"
        "• Style → LLM picks per story\n"
        "• Resolution → backend default\n"
        "• Steps / CFG → backend default\n"
        "• Voice → `af_heart`\n"
        "• Sync mode → `strict`\n"
        "• Clip length → matches narration"
    )


@bot.command(name="info", aliases=["quickinfo"])
async def cmd_info(ctx):
    """Quick text snapshot of current settings (old !panel)."""
    try:
        img_cfg = model_registry.get_active("image_backend")
        backend_id = img_cfg.get("_id", "unknown")
    except Exception:
        backend_id = "unknown"

    style = rs.get_effective_style()
    ratio = rs.get_effective_aspect_ratio()
    steps = rs.get_effective_steps()
    cfg = rs.get_effective_cfg()
    overrides = rs.get_all_overrides()

    def label(key, fallback="default"):
        return "override" if key in overrides else fallback

    e = discord.Embed(title="🎛️ Quick Info", color=discord.Color.purple())
    e.add_field(name="🎨 Style", value=f"`{style}` _{label('style', 'LLM-picked')}_", inline=True)
    e.add_field(name="🔌 Model", value=f"`{backend_id}`", inline=True)
    e.add_field(name="📐 Aspect", value=f"`{ratio}` _{label('aspect_ratio')}_", inline=True)
    e.add_field(name="⚙️ Steps", value=f"`{steps}` _{label('steps')}_", inline=True)
    e.add_field(name="🎲 CFG", value=f"`{cfg}` _{label('cfg')}_", inline=True)
    e.set_footer(text="Use !panel for the interactive button UI")
    await ctx.send(embed=e)


@bot.command(name="nuke")
@commands.has_permissions(administrator=True)
async def cmd_nuke(ctx):
    """💣 Wipe ALL messages from bot channels (claw-bot, scripts, storyboards, videos, status)."""
    async def repost_panel(ch):
        await control_panel.ensure_panel(ch)

    async def repost_dashboard(ch):
        try:
            dashboard = StatusDashboard(
                bot=bot, bot_version=BOT_VERSION,
                stats_getter=lambda: STATS,
            )
            from modules import health_monitor as hm
            hm.set_instance(dashboard)
            await dashboard.start(ch)
            bot._dashboard = dashboard
        except Exception as e:
            log.warning(f"dashboard restart failed: {e}")

    await channel_cleanup.run_nuke(
        ctx,
        channel_cleanup.BOT_CHANNELS,
        get_channel_by_name,
        post_panel=repost_panel,
        post_dashboard=repost_dashboard,
    )


@bot.command(name="nuke_channel")
@commands.has_permissions(administrator=True)
async def cmd_nuke_channel(ctx, channel_name: str = None):
    """💣 Wipe ALL messages from one specific channel. Usage: !nuke_channel scripts"""
    if not channel_name:
        await ctx.send(
            "Usage: `!nuke_channel <name>`\n"
            f"Valid: {', '.join(f'`{c}`' for c in channel_cleanup.BOT_CHANNELS)}"
        )
        return
    if channel_name not in channel_cleanup.BOT_CHANNELS:
        await ctx.send(f"❌ `{channel_name}` is not a tracked bot channel.")
        return

    async def repost_panel(ch):
        await control_panel.ensure_panel(ch)

    await channel_cleanup.run_nuke_single(
        ctx, channel_name, get_channel_by_name, post_panel=repost_panel
    )


@bot.command(name="panel", aliases=["menu", "ui"])
async def cmd_panel(ctx):
    """Open a fresh copy of the control panel here. (The pinned one in #claw-bot is the default.)"""
    await control_panel.open_panel_manual(ctx)


@bot.command(name="resetpanel")
@commands.has_permissions(administrator=True)
async def cmd_resetpanel(ctx):
    """Delete every old panel in this channel and post one fresh, pinned panel."""
    deleted = 0
    async for m in ctx.channel.history(limit=200):
        if m.author.id != ctx.guild.me.id:
            continue
        for emb in m.embeds:
            if emb.title and "Claw Bot — Control Panel" in emb.title:
                try:
                    await m.delete()
                    deleted += 1
                except Exception:
                    pass
                break
    # Also clear saved state so ensure_panel posts a fresh one
    try:
        from modules.control_panel import _load_state, _save_state
        state = _load_state()
        state.pop(str(ctx.channel.id), None)
        _save_state(state)
    except Exception as e:
        log.warning(f"state clear failed: {e}")

    await control_panel.ensure_panel(ctx.channel)
    await send_transient(ctx.channel, f"🧹 Cleared {deleted} old panels. Fresh one posted.", 8)


@bot.command(name="current_settings", aliases=["settings"])
async def cmd_current_settings(ctx):
    """Show currently active backend, style defaults, and session stats."""
    from modules import script_generator as sg_mod
    try:
        img_cfg = model_registry.get_active("image_backend")
        llm_cfg = model_registry.get_active("llm_backend")
    except Exception as e:
        await ctx.send(f"❌ Could not read registry: {e}")
        return

    default_style = sg_mod.get_default_style()
    defaults = model_registry.get_image_defaults()
    try:
        vid_cfg = model_registry.get_active("video_backend") or {}
    except Exception:
        vid_cfg = {}
    vres = rs.get_video_resolution_override()
    vdims = rs.get_effective_video_resolution()
    vres_label = f"`{vres}` ({vdims[0]}x{vdims[1]})" if vres and vdims else "model default"

    lines = [
        "**⚙️ Current Settings**",
        "",
        "**🎨 Image Backend**",
        f"• Active: `{img_cfg.get('_id', 'unknown')}`",
        f"• Description: {img_cfg.get('description', '')}",
        f"• Steps: `{img_cfg.get('steps', '?')}`  ·  CFG: `{img_cfg.get('cfg', '?')}`  "
        f"·  Sampler: `{img_cfg.get('sampler', '?')}`",
        "",
        "**🎥 Video Backend**",
        f"• Active: `{vid_cfg.get('_id', 'unknown')}`",
        f"• Resolution: {vres_label}  ·  fps: `{vid_cfg.get('default_fps', '?')}`  "
        f"·  max clip: `{vid_cfg.get('max_clip_seconds', '?')}s`",
        "",
        "**🧠 LLM Backend**",
        f"• Active: `{llm_cfg.get('_id', 'unknown')}`",
        f"• Model: `{llm_cfg.get('model_name') or llm_cfg.get('model', '?')}`",
        "",
        "**🖼️ Image Defaults**",
        f"• Default aspect ratio: `{defaults.get('aspect_ratio', '16:9')}`",
        f"• Default style: `{default_style}`",
        "",
        "**📊 Session Stats**",
        f"• Scripts generated: `{STATS.get('generated', 0)}`  ·  approved: `{STATS.get('approved', 0)}`",
        f"• Storyboards generated: `{STATS.get('storyboards_generated', 0)}`  ·  approved: `{STATS.get('storyboards_approved', 0)}`",
        "",
        "Use `!switch_model <id>` to change image backend.",
        "Use `!help` for the full command list.",
    ]
    await ctx.send("\n".join(lines))

# ============================================================
# COMMANDS — Storyboard
# ============================================================

@bot.command(name="repolish")
async def cmd_repolish(ctx, script_id: str = None):
    """Re-run Qwen polish pass on an existing script."""
    if script_id is None:
        await ctx.send("⚠️ Usage: `!repolish <script_id>`")
        return

    script_file = OUTPUTS_DIR / f"script_{script_id}.json"
    if not script_file.exists():
        await ctx.send(f"❌ Script `{script_id}` not found.")
        return

    msg = await ctx.send(f"🪄 Re-polishing `{script_id}` with Qwen 2.5 14B... (~1-2 min)")
    try:
        original = json.loads(script_file.read_text(encoding="utf-8"))
        polished = await asyncio.to_thread(pp.polish_script, original)
        await msg.edit(content=f"✨ Polish complete for `{script_id}`. Run `!generate_storyboard {script_id}` to render.")
    except Exception as e:
        log.exception(f"Repolish failed: {e}")
        await msg.edit(content=f"❌ Repolish failed: `{e}`")


@bot.command(name="retry_failed", aliases=["rf"])
async def cmd_retry_failed(ctx, script_id: str = None):
    """Re-render any storyboard frames or video clips missing from disk.

    Checks the storyboard manifest + the expected clip filenames for a
    script_id. Anything missing is re-run via the same workflows the auto
    pipeline uses.
    """
    if script_id is None:
        await ctx.send("⚠️ Usage: `!retry_failed <script_id>`")
        return

    from pathlib import Path as _Path
    sb_manifest = PROJECT_ROOT / "04_Outputs" / "storyboards" / script_id / "storyboard.json"
    clips_dir = PROJECT_ROOT / "04_Outputs" / "clips"

    if not sb_manifest.exists():
        await ctx.send(f"❌ No storyboard manifest for `{script_id}`.")
        return

    manifest = json.loads(sb_manifest.read_text(encoding="utf-8"))
    frames = manifest.get("frames", [])
    expected_shots = sorted({f.get("shot_number") for f in frames if f.get("shot_number")})

    # Missing storyboard images
    missing_sb = []
    for f in frames:
        abs_p = PROJECT_ROOT / f.get("image_path", "")
        if not abs_p.exists():
            missing_sb.append(f.get("shot_number"))

    # Missing video clips
    missing_clips = []
    for sn in expected_shots:
        glob_pattern = f"clip_{script_id}_shot{sn}*.mp4"
        found = list(clips_dir.glob(glob_pattern))
        if not found:
            missing_clips.append(sn)

    if not missing_sb and not missing_clips:
        await ctx.send(f"✅ Nothing to retry for `{script_id}` — all shots present.")
        return

    summary = []
    if missing_sb:
        summary.append(f"Missing storyboards: `{missing_sb}`")
    if missing_clips:
        summary.append(f"Missing clips: `{missing_clips}`")
    await ctx.send("🔄 Retrying:\n" + "\n".join(summary))

    target_sb_channel = get_channel_by_name(ctx.guild, "storyboards") or ctx.channel
    target_v_channel = get_channel_by_name(ctx.guild, "videos") or ctx.channel

    if missing_sb:
        asyncio.create_task(sw.regenerate_shots(
            channel=target_sb_channel,
            script_id=script_id,
            shot_numbers=missing_sb,
            owner_id=ctx.author.id,
            post_approval=False,
        ))

    if missing_clips:
        asyncio.create_task(vw.regenerate_video_shots(
            channel=target_v_channel,
            script_id=script_id,
            shot_numbers=missing_clips,
            owner_id=ctx.author.id,
        ))


@bot.command(name="retailor", aliases=["rt"])
async def cmd_retailor(ctx, script_id: str = None):
    """Re-run beat-aware shot tailoring on an existing script."""
    if script_id is None:
        await ctx.send("⚠️ Usage: `!retailor <script_id>`")
        return

    script_file = OUTPUTS_DIR / f"script_{script_id}.json"
    if not script_file.exists():
        await ctx.send(f"❌ Script `{script_id}` not found.")
        return

    msg = await ctx.send(f"🎯 Re-tailoring `{script_id}` per shot beat with Qwen 2.5 14B... (~1-2 min)")
    try:
        original = json.loads(script_file.read_text(encoding="utf-8"))
        tailored = await asyncio.to_thread(st.tailor_script, original)
        try:
            await ctx.send(st.format_tailor_summary_for_discord(tailored))
        except Exception as fe:
            log.warning(f"Tailor summary post failed: {fe}")
        await msg.edit(content=f"🎯 Tailor complete for `{script_id}`. Run `!generate_storyboard {script_id}` to render.")
    except Exception as e:
        log.exception(f"Retailor failed: {e}")
        await msg.edit(content=f"❌ Retailor failed: `{e}`")

@bot.command(name="generate_storyboard", aliases=["gsb", "storyboard"])
async def cmd_generate_storyboard(ctx, script_id: str = None):
    """Manually generate a storyboard for a given script."""
    if script_id is None:
        await ctx.send("⚠️ Usage: `!generate_storyboard <script_id>`. Use `!list_scripts` to find IDs.")
        return

    # Verify script exists
    script_file = OUTPUTS_DIR / f"script_{script_id}.json"
    if not script_file.exists():
        await ctx.send(f"❌ Script `{script_id}` not found. Try `!list_scripts`.")
        return

    target = get_channel_by_name(ctx.guild, "storyboards") or ctx.channel
    if ctx.channel.id != target.id:
        await ctx.send(f"🎬 Storyboarding... posting frames in {target.mention}.")

    asyncio.create_task(_run_storyboard_pipeline(
        target, script_id, ctx.author.id, ctx.author.mention, is_auto=False
    ))


@bot.command(name="list_storyboards", aliases=["lsb"])
async def cmd_list_storyboards(ctx):
    """Show the 10 most recent storyboards."""
    entries = sw.list_recent_storyboards(limit=10)
    if not entries:
        await ctx.send("📭 No storyboards generated yet.")
        return
    lines = ["**🎬 Recent Storyboards:**"]
    for e in entries:
        status = "✅" if e["success"] else "❌"
        lines.append(
            f"• {status} `{e['script_id']}` — {e['frames_count']} frames, backend: `{e['backend_id']}`"
        )
    await ctx.send("\n".join(lines))


@bot.command(name="regen_shot", aliases=["regen"])
async def cmd_regen_shot(ctx, script_id: str = None, shot_num: int = None):
    """Regenerate a specific shot of an existing storyboard."""
    if script_id is None or shot_num is None:
        await ctx.send("⚠️ Usage: `!regen_shot <script_id> <shot_number>`")
        return
    # Validate against the actual shot count in this script (organic count)
    try:
        _total = len(sw.load_script(script_id).get("shots", []))
    except Exception:
        _total = 20
    if not 1 <= shot_num <= _total:
        await ctx.send(f"⚠️ Shot number must be 1-{_total}.")
        return

    target = get_channel_by_name(ctx.guild, "storyboards") or ctx.channel
    await _delete_pending_storyboard_approvals(target, script_id)
    result_info = await sw.regenerate_shots(
        channel=target, script_id=script_id, shot_numbers=[shot_num], owner_id=ctx.author.id
    )
    if result_info:
        PENDING_STORYBOARD_APPROVALS[result_info["approval_msg_id"]] = {
            "script_id": result_info["script_id"],
            "owner_id": result_info["owner_id"],
        }


@bot.command(name="edit_prompts", aliases=["edit_prompt", "editprompts", "reprompt", "fix_prompt"])
async def cmd_edit_prompts(ctx, script_id: str = None, *shot_nums: int):
    """Re-open prompt editing for an existing script, then regen those shots.

    Usage: !edit_prompts <script_id> [shot_num ...]
      - no shot_num  -> edit + regen ALL shots
      - one or more  -> edit + regen only those shots

    Reads the saved approved_prompts JSON (no costly LLM re-assembly), re-posts
    the per-shot Edit/Reseed embeds so a bad storyboard shot can be fixed, then
    Approve-ALL re-renders the targeted shots.
    """
    if script_id is None:
        await ctx.send(
            "⚠️ Usage: `!edit_prompts <script_id> [shot_num ...]`\n"
            "Example: `!edit_prompts abc123 3` (edit + regen shot 3)"
        )
        return
    try:
        script = sw.load_script(script_id)
    except Exception as e:
        await ctx.send(f"❌ Could not load script `{script_id}`: `{e}`")
        return
    if not script or not script.get("shots"):
        await ctx.send(f"❌ Script `{script_id}` has no shots.")
        return

    _total = len(script.get("shots", []))
    shots = list(shot_nums)
    for s in shots:
        if not 1 <= s <= _total:
            await ctx.send(f"⚠️ Shot number must be 1-{_total}.")
            return

    # Old scripts rendered before/without the approval step have no saved prompts
    # JSON — rebuild an editable one from the storyboard manifest + script (no LLM)
    # so the real prompts that produced each frame show up for editing.
    try:
        pap.backfill_from_disk(script_id, script)
    except Exception as e:
        log.warning(f"edit_prompts backfill failed for {script_id}: {e}")

    target = get_channel_by_name(ctx.guild, "storyboards") or ctx.channel
    if ctx.channel.id != target.id:
        await ctx.send(f"📝 Re-opening prompt editing for `{script_id}` in {target.mention}.")

    async def _on_regen(interaction):
        await _delete_pending_storyboard_approvals(target, script_id)
        regen_list = shots or list(range(1, _total + 1))
        await target.send(
            f"🚀 **Prompts re-approved for `{script_id}`** — re-rendering shot(s) "
            f"`{', '.join(map(str, regen_list))}`…"
        )
        result_info = await sw.regenerate_shots(
            channel=target, script_id=script_id,
            shot_numbers=regen_list, owner_id=ctx.author.id,
        )
        if result_info:
            PENDING_STORYBOARD_APPROVALS[result_info["approval_msg_id"]] = {
                "script_id": result_info["script_id"],
                "owner_id": result_info["owner_id"],
            }

    async def _on_cancel(interaction):
        await target.send(f"🛑 Prompt re-edit for `{script_id}` cancelled.")

    try:
        await pap.post_approval_ui(
            channel=target,
            script=script,
            owner_id=ctx.author.id,
            on_approve_all=_on_regen,
            on_cancel=_on_cancel,
            reuse_existing=True,
            only_shots=shots or None,
            action_word="Regenerate",
        )
    except Exception as e:
        log.exception(f"edit_prompts failed for {script_id}")
        await target.send(f"❌ Could not open prompt editor for `{script_id}`: `{e}`")


@bot.command(name="regen_video_shot", aliases=["regen_video", "rvs"])
async def cmd_regen_video_shot(ctx, script_id: str = None, *shot_nums, shot_num=None):
    """Re-render specific video clips. Usage: !regen_video_shot <script_id> <shot_num> [<shot_num> ...]"""
    # Control-panel modal passes shot_num= as a (possibly space-separated) string.
    if shot_num is not None:
        shot_nums = shot_nums + tuple(str(shot_num).split())
    # Normalize all shot identifiers to ints, drop junk.
    parsed = []
    for s in shot_nums:
        try:
            parsed.append(int(str(s).strip()))
        except (ValueError, TypeError):
            continue
    shot_nums = tuple(parsed)

    if script_id is None or not shot_nums:
        await ctx.send(
            "⚠️ Usage: `!regen_video_shot <script_id> <shot_num> [<shot_num> ...]`\n"
            "Example: `!regen_video_shot abc123 2 4`"
        )
        return

    target = get_channel_by_name(ctx.guild, "videos") or ctx.channel
    if ctx.channel.id != target.id:
        await ctx.send(f"🎥 Re-rendering shot(s) `{', '.join(map(str, shot_nums))}` "
                       f"in {target.mention}.")

    asyncio.create_task(_run_video_regen(
        target, script_id, list(shot_nums), ctx.author.id
    ))


@bot.command(name="upscale", aliases=["up", "upscale_video"])
async def cmd_upscale(ctx, script_id: str = None):
    """Upscale all video clips for a script_id via Real-ESRGAN (4x)."""
    if script_id is None:
        await ctx.send(
            "⚠️ Usage: `!upscale <script_id>`\n"
            "Upscales all clips for that script_id 4x via Real-ESRGAN."
        )
        return

    from modules import upscaler as up
    status_msg = await ctx.send(
        f"📈 Starting upscale for `{script_id}`...\n"
        f"⏳ ~1-2 min per clip on RTX 5080. Originals will be REPLACED."
    )

    # Capture event loop for thread-safe progress edits
    bot_loop = asyncio.get_running_loop()
    last_update = {"t": 0.0, "tick": 0}
    from modules.progress_bar import render_indeterminate

    def progress(text: str):
        import time as _t
        now = _t.time()
        if now - last_update["t"] < 2.0:
            return
        last_update["t"] = now
        last_update["tick"] += 1
        bar = render_indeterminate(last_update["tick"])
        asyncio.run_coroutine_threadsafe(
            status_msg.edit(content=f"📈 **Upscaling** `{script_id}`\n{bar} · {text}"),
            bot_loop
        )

    if not job_lock.acquire("discord:upscale"):
        await ctx.send(f"⏳ GPU busy with **{job_lock.holder_label()}** — try again later.")
        return
    try:
        result = await asyncio.to_thread(
            up.upscale_storyboard_videos, script_id, progress
        )
        await ctx.send(up.format_upscale_summary(result))
    except FileNotFoundError as e:
        await ctx.send(f"❌ {e}")
    except Exception as e:
        log.exception("Upscale command failed")
        await ctx.send(f"❌ Upscale failed: `{e}`")
    finally:
        job_lock.release()


@bot.command(name="assemble", aliases=["asm", "final"])
async def cmd_assemble(ctx, script_id: str = None):
    """Stitch all clips for a script_id into final 9x16 + 16x9 MP4s with crossfades."""
    if script_id is None:
        await ctx.send(
            "⚠️ Usage: `!assemble <script_id>`\n"
            "Stitches all clips into final 9x16 (Shorts) + 16x9 versions with 0.3s crossfades."
        )
        return

    from modules import assembly as asm
    status_msg = await ctx.send(f"🎬 Assembling `{script_id}`...")

    bot_loop = asyncio.get_running_loop()
    last_update = {"t": 0.0, "tick": 0}
    from modules.progress_bar import render_indeterminate

    def progress(text: str):
        import time as _t
        now = _t.time()
        if now - last_update["t"] < 2.0:
            return
        last_update["t"] = now
        last_update["tick"] += 1
        bar = render_indeterminate(last_update["tick"])
        asyncio.run_coroutine_threadsafe(
            status_msg.edit(content=f"🎬 **Assembling** `{script_id}`\n{bar} · {text}"),
            bot_loop
        )

    if not job_lock.acquire("discord:assembly"):
        await ctx.send(f"⏳ GPU busy with **{job_lock.holder_label()}** — try again later.")
        return
    try:
        result = await asyncio.to_thread(asm.assemble_final, script_id, progress)
        summary = (
            f"✅ **Assembly complete** — {result['shot_count']} shots, "
            f"{result['total_duration_sec']:.1f}s\n"
            f"📂 `{result['9x16'].name}` (Shorts)\n"
            f"📂 `{result['16x9'].name}` (Horizontal)"
        )
        await ctx.send(summary)
        for key in ("9x16", "16x9"):
            f = result[key]
            size_mb = f.stat().st_size / (1024 * 1024)
            if size_mb <= 9:
                try:
                    await ctx.send(file=discord.File(str(f)))
                except discord.HTTPException as ue:
                    await ctx.send(
                        f"📁 `{f.name}` ({size_mb:.1f} MB) — too large to upload. "
                        f"File saved at:\n`{f}`"
                    )
            else:
                await ctx.send(
                    f"📁 `{f.name}` ({size_mb:.1f} MB) — preview too large. "
                    f"File saved at:\n`{f}`"
                )
    except FileNotFoundError as e:
        await ctx.send(f"❌ {e}")
    except Exception as e:
        log.exception("Assemble command failed")
        await ctx.send(f"❌ Assembly failed: `{e}`")
    finally:
        job_lock.release()


@bot.command(name="generate_video", aliases=["gv", "video"])
async def cmd_generate_video(ctx, script_id: str = None):
    """Manually trigger video generation for a script with an existing approved storyboard."""
    if script_id is None:
        await ctx.send(
            "⚠️ Usage: `!generate_video <script_id>`\n"
            "Requires an existing approved storyboard. Use `!list_storyboards` to find IDs."
        )
        return

    # Check storyboard manifest exists
    manifest_path = (
        OUTPUTS_DIR.parent / "storyboards" / script_id / "storyboard.json"
    )
    if not manifest_path.exists():
        await ctx.send(
            f"❌ No storyboard found for `{script_id}`. "
            f"Generate one first with `!generate_storyboard {script_id}`."
        )
        return

    target = get_channel_by_name(ctx.guild, "videos") or ctx.channel
    if ctx.channel.id != target.id:
        await ctx.send(f"🎥 Starting video generation in {target.mention}.")

    asyncio.create_task(_run_video_pipeline(
        target, script_id, ctx.author.id, ctx.author.mention, is_auto=False
    ))


@bot.command(name="switch_model")
async def cmd_switch_model(ctx, backend_id: str = None, backend_type: str = None):
    """
    Switch the active backend.
    Usage:
      !switch_model                            → show all backends
      !switch_model <id>                       → auto-detect type, switch
      !switch_model <id> <image|video>         → explicit type
    """
    if backend_id is None:
        # Show all backends across all types
        lines = []
        for btype, label in [("image_backend", "🖼️ Image"), ("video_backend", "🎥 Video")]:
            try:
                current = model_registry.get_active(btype)
                available = model_registry.list_available(btype)
            except Exception:
                continue
            current_id = current["_id"] if current else "—"
            lines.append(f"**{label} backend** (active: `{current_id}`)")
            for k, v in available.items():
                marker = " (active)" if k == current_id else ""
                lines.append(f"• `{k}`{marker} — {v}")
            lines.append("")
        lines.append("Switch with: `!switch_model <backend_id>` (type auto-detected)")
        await ctx.send("\n".join(lines))
        return

    # Auto-detect backend type if not given
    if backend_type is None:
        for btype in ("image_backend", "video_backend"):
            try:
                if backend_id in model_registry.list_available(btype):
                    backend_type = btype
                    break
            except Exception:
                pass
        if backend_type is None:
            await ctx.send(
                f"❌ Backend `{backend_id}` not found in any registry. "
                f"Run `!switch_model` (no args) to list available."
            )
            return
    else:
        # Normalize the user's input ("image" -> "image_backend", etc.)
        backend_type = backend_type.lower()
        if backend_type in ("image", "img"):
            backend_type = "image_backend"
        elif backend_type in ("video", "vid"):
            backend_type = "video_backend"
        if backend_type not in ("image_backend", "video_backend"):
            await ctx.send(f"❌ Invalid type `{backend_type}`. Must be `image` or `video`.")
            return

    try:
        new_cfg = model_registry.set_active(backend_type, backend_id)
        nice_label = "image" if backend_type == "image_backend" else "video"
        # Reset cfg/steps overrides so the new model runs at ITS recommended
        # settings (each model's best config lives in models.json, not in global
        # overrides left over from the previous model).
        from modules import runtime_settings as rs_mod
        rs_mod.reset_overrides_for_model_switch()
        msg = (
            f"🔌 **Switched active {nice_label} backend to** `{backend_id}`.\n"
            f"{new_cfg.get('description', '')}\n"
            f"♻️ Tuning overrides reset — using this model's recommended settings:"
        )
        if backend_type == "video_backend":
            msg += (
                f" cfg=`{new_cfg.get('cfg', '?')}`, "
                f"steps=`{new_cfg.get('steps_full', new_cfg.get('steps', '?'))}`, "
                f"fps=`{new_cfg.get('default_fps', '?')}`, "
                f"max_clip=`{new_cfg.get('max_clip_seconds', '?')}s`."
            )
        else:
            msg += (
                f" cfg=`{new_cfg.get('cfg', '?')}`, "
                f"steps=`{new_cfg.get('steps', '?')}`."
            )
        await ctx.send(msg)
    except KeyError as e:
        await ctx.send(f"❌ Invalid backend: `{e}`. Run `!switch_model` with no args to see available.")
    except Exception as e:
        await ctx.send(f"❌ Switch failed: `{e}`")

if __name__ == "__main__":
    log.info(f"Starting Claw Bot v{BOT_VERSION}...")

    # Fail fast on broken config — a models.json typo should stop the boot
    # with a clear message, not crash a render two hours in.
    from modules import config_check
    _cfg_errors = config_check.validate_configs()
    if _cfg_errors:
        for _err in _cfg_errors:
            log.error(f"CONFIG ERROR: {_err}")
        log.error("Fix 05_Config and relaunch. Bot NOT started.")
        sys.exit(1)
    config_check.warn_on_secrets_bom()
    _disk_ok, _free_gb = config_check.check_disk_space()
    if not _disk_ok:
        log.warning(f"LOW DISK: only {_free_gb} GB free on the project drive — "
                    f"video renders may fail mid-write.")

    try:
        bot.run(DISCORD_BOT_TOKEN, log_handler=None)
    except discord.LoginFailure:
        log.error("Login failed — token invalid.")
        sys.exit(1)
    except KeyboardInterrupt:
        log.info("Shutdown requested. Bye!")