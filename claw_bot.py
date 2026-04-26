"""
Claw Bot Agent — Phase 3 FINAL
Adds: pause/resume feedback so you can revisit revisions later.
"""

import os
import sys
import asyncio
import logging
import json
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
    generate_script, revise_script, format_for_discord, OUTPUTS_DIR
)
from modules.theme_bank import get_theme_of_the_day, get_random_theme, get_theme_count
from modules.safety_filter import check_safety, safety_summary_for_discord
from modules import pending_feedback as pf
from modules import storyboard_workflow as sw
from modules import gpu_utils
from modules.health_monitor import StatusDashboard
from modules import model_registry
from modules import gpu_utils
from modules.health_monitor import StatusDashboard

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

STATS_FILE = CONFIG_DIR / "stats.json"


# ============================================================
# LOGGING
# ============================================================

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "claw_bot.log", encoding="utf-8"),
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
    STATS_FILE.write_text(json.dumps(stats, indent=2), encoding="utf-8")


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
PENDING_APPROVALS = {}   # approval_msg_id -> {"script": dict, "owner_id": int}
PENDING_STORYBOARD_APPROVALS = {}   # approval_msg_id -> {"script_id": str, "owner_id": int}
scheduler = AsyncIOScheduler(timezone=IST)


# ============================================================
# HELPERS
# ============================================================

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


async def _post_with_approval_reactions(channel, script: dict, owner_id: int):
    """Post script + approval prompt with ✅ ❌ ⏹️ reactions."""
    await send_long_message(channel, format_for_discord(script))

    rev_num = script.get("revision_number", 1)
    rev_info = f" (revision v{rev_num})" if rev_num > 1 else ""
    approval_msg = await channel.send(
        f"**Script `{script['script_id']}`**{rev_info} — awaiting your decision.\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ Approve → send to Phase 4 (storyboards, coming soon)\n"
        f"❌ Reject → tell me what to change, I'll revise\n"
        f"⏹️ Stop this story entirely (no revision)"
    )
    await approval_msg.add_reaction("✅")
    await approval_msg.add_reaction("❌")
    await approval_msg.add_reaction("⏹️")

    PENDING_APPROVALS[approval_msg.id] = {"script": script, "owner_id": owner_id}
    # Unload Llama from VRAM so image generation has room
    try:
        await asyncio.to_thread(gpu_utils.free_ollama_vram)
    except Exception as e:
        log.warning(f"Ollama unload after script gen failed: {e}")
    return approval_msg


async def _generate_and_post(channel, theme: str, requested_by_id: int,
                             requested_by_mention: str, *, is_auto: bool = False):
    global STATS
    prefix = "🌅 **Daily Auto-Story**" if is_auto else f"📝 Request from {requested_by_mention}"
    status_msg = await channel.send(f"{prefix}\nTheme: **{theme}**\n⏳ Writing... (20-40 sec)")

    try:
        script = await asyncio.to_thread(generate_script, theme)
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

    await status_msg.delete()
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

    sid = original_script.get("_id") or script.get("script_id")
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


async def _do_revision(channel, original_script: dict, notes: str, owner_id: int):
    """Perform the actual revision + post."""
    global STATS
    status = await channel.send(f"✏️ Revising with your notes:\n> *{notes[:200]}*\n⏳ Writing...")
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
                await dashboard.start(ch)
                bot._dashboard = dashboard   # keep reference alive
            except Exception as e:
                log.warning(f"Could not start status dashboard: {e}")


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
    if msg_id not in PENDING_APPROVALS:
        return

    entry = PENDING_APPROVALS[msg_id]
    script = entry["script"]
    owner_id = entry["owner_id"]

    if user.id != owner_id and not user.guild_permissions.administrator:
        return

    emoji = str(reaction.emoji)
    channel = reaction.message.channel
    script_id = script.get("_id") or script.get("script_id")

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

        # Auto-trigger storyboard in the #storyboards channel if it exists
        storyboards_channel = get_channel_by_name(channel.guild, "storyboards") or channel
        asyncio.create_task(_run_storyboard_pipeline(
            storyboards_channel, script_id, user.id, user.mention, is_auto=True
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
                f"*Ready to move to Phase 5 (video generation) — coming soon.*"
            )
            APPROVED_STORYBOARDS_DIR = OUTPUTS_DIR.parent / "approved_storyboards"
            APPROVED_STORYBOARDS_DIR.mkdir(parents=True, exist_ok=True)
            (APPROVED_STORYBOARDS_DIR / f"{sb_script_id}.approved").write_text(
                f"approved_by={user}\napproved_at={datetime.now(timezone.utc).isoformat()}\n",
                encoding="utf-8",
            )
            del PENDING_STORYBOARD_APPROVALS[msg_id]

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

# ============================================================
# STORYBOARD PIPELINE HELPER
# ============================================================

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

    # Parse which shots to redo
    shot_nums = sw.extract_shot_numbers(notes)
    if not shot_nums:
        # Assume "redo all"
        if any(kw in notes for kw in ["all", "everything", "whole", "entire"]):
            shot_nums = [1, 2, 3, 4]
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
        await ctx.send(f"📝 Generating... posting in {target.mention}.")

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
    await ctx.send(f"🎲 **Theme suggestion:** {theme}\nUse: `!generate_script {theme}`")


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
    """Full command reference."""
    from modules import script_generator as sg_mod
    available_styles = sg_mod.get_available_style_ids()
    style_list = ", ".join(f"`{s}`" for s in available_styles)

    sections = [
        "**🤖 Claw Bot Commands**",
        "",
        "**📝 Script Generation**",
        "• `!generate_script <theme>` — Generate a new script from a theme. LLM picks style/culture/shot count.",
        "  Example: `!generate_script a shy mouse learning to speak up`",
        "• `!today_script` — Generate using today's theme of the day.",
        "• `!suggest_theme` — Get a random theme suggestion.",
        "• `!list_scripts` — Show recent scripts.",
        "• `!show_script <id>` — Display a specific script.",
        "",
        "**🎬 Storyboard Generation**",
        "• `!generate_storyboard <script_id>` or `!gsb <id>` — Render images for an approved script.",
        "• `!list_storyboards` or `!lsb` — Show recent storyboards.",
        "• `!regen_shot <script_id> <shot_number>` — Re-render a single shot.",
        "",
        "**🎨 Styles**",
        f"• `!list_styles` — Show all {len(available_styles)} available styles.",
        "• Available: " + style_list,
        "",
        "**🔌 Model & Config**",
        "• `!switch_model` — Show active image backend and alternatives.",
        "• `!switch_model <backend_id>` — Switch to a different image model.",
        "• `!current_settings` — Show active backend, style, resolution, and active stats.",
        "",
        "**📊 Utility**",
        "• `!status` — Bot status summary.",
        "• `!ping` — Check bot is alive.",
        "• `!stats` — Session stats.",
        "• `!pending` — List any paused feedback.",
        "• `!resume_feedback <script_id>` — Resume a paused feedback loop.",
        "• `!drop_feedback <script_id>` — Discard paused feedback.",
        "",
        "**🧭 Reaction Controls (on script/storyboard messages)**",
        "• ✅ Approve  · ❌ Reject & revise  · ⏹️ Stop  · 💾 Pause feedback",
        "",
        "📖 Live dashboard in `#status` channel.",
    ]
    msg = "\n".join(sections)
    # Discord messages max 2000 chars — split if needed
    if len(msg) <= 1900:
        await ctx.send(msg)
    else:
        # Rare but handle it
        chunks = []
        current = ""
        for line in sections:
            if len(current) + len(line) + 1 > 1900:
                chunks.append(current)
                current = line
            else:
                current = current + "\n" + line if current else line
        if current:
            chunks.append(current)
        for c in chunks:
            await ctx.send(c)


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

    lines = [
        "**⚙️ Current Settings**",
        "",
        "**🎨 Image Backend**",
        f"• Active: `{img_cfg.get('_id', 'unknown')}`",
        f"• Description: {img_cfg.get('description', '')}",
        f"• Steps: `{img_cfg.get('steps', '?')}`  ·  CFG: `{img_cfg.get('cfg', '?')}`  "
        f"·  Sampler: `{img_cfg.get('sampler', '?')}`",
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
    if not 1 <= shot_num <= 4:
        await ctx.send("⚠️ Shot number must be 1-4.")
        return

    target = get_channel_by_name(ctx.guild, "storyboards") or ctx.channel
    result_info = await sw.regenerate_shots(
        channel=target, script_id=script_id, shot_numbers=[shot_num], owner_id=ctx.author.id
    )
    if result_info:
        PENDING_STORYBOARD_APPROVALS[result_info["approval_msg_id"]] = {
            "script_id": result_info["script_id"],
            "owner_id": result_info["owner_id"],
        }


@bot.command(name="switch_model")
async def cmd_switch_model(ctx, backend_id: str = None):
    """Switch the active image backend (e.g., comfyui_zimage_base, comfyui_sdxl_turbo)."""
    if backend_id is None:
        current = model_registry.get_active("image_backend")
        available = model_registry.list_available("image_backend")
        lines = [
            f"**🔌 Active image backend:** `{current['_id']}`",
            "",
            "**Available:**",
        ]
        for k, v in available.items():
            marker = " (active)" if k == current["_id"] else ""
            lines.append(f"• `{k}`{marker} — {v}")
        lines.append("")
        lines.append("Switch with: `!switch_model <backend_id>`")
        await ctx.send("\n".join(lines))
        return

    try:
        new_cfg = model_registry.set_active("image_backend", backend_id)
        await ctx.send(
            f"🔌 **Switched active image backend to** `{backend_id}`.\n"
            f"{new_cfg.get('description', '')}"
        )
    except KeyError as e:
        await ctx.send(f"❌ Invalid backend: `{e}`. Run `!switch_model` with no args to see available.")
    except Exception as e:
        await ctx.send(f"❌ Switch failed: `{e}`")

if __name__ == "__main__":
    log.info(f"Starting Claw Bot v{BOT_VERSION}...")
    try:
        bot.run(DISCORD_BOT_TOKEN, log_handler=None)
    except discord.LoginFailure:
        log.error("Login failed — token invalid.")
        sys.exit(1)
    except KeyboardInterrupt:
        log.info("Shutdown requested. Bye!")