"""
Claw Bot — Channel Cleanup

Wipes all messages from the bot's working channels, with a button-confirm step.
Bulk-deletes anything <14 days old, falls back to per-message delete for older.
After wipe, reposts the control panel in #claw-bot and the dashboard in #status.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import discord
from discord import ui

log = logging.getLogger("claw_bot.channel_cleanup")

# Channels this bot owns
BOT_CHANNELS = ["claw-bot", "scripts", "storyboards", "videos", "status"]


# ============================================================
# CORE PURGE
# ============================================================

async def purge_channel(channel: discord.TextChannel,
                        status_msg: Optional[discord.Message] = None) -> dict:
    """Delete every message in a channel. Returns counts."""
    bulk_cutoff = datetime.now(timezone.utc) - timedelta(days=13, hours=23)

    bulk_deleted = 0
    slow_deleted = 0
    failed = 0
    batch_count = 0

    # Phase 1: bulk delete (fast, <14d only)
    while True:
        try:
            msgs = [m async for m in channel.history(limit=100, after=bulk_cutoff)]
        except Exception as e:
            log.warning(f"history() failed in #{channel.name}: {e}")
            break

        if not msgs:
            break

        try:
            await channel.delete_messages(msgs)
            bulk_deleted += len(msgs)
            batch_count += 1
        except discord.HTTPException as e:
            # delete_messages refuses if any message is >14d — fall through to slow path
            log.warning(f"bulk delete failed in #{channel.name}: {e}")
            break

        if status_msg and batch_count % 2 == 0:
            try:
                await status_msg.edit(
                    content=f"🧹 Cleaning `#{channel.name}` — {bulk_deleted} deleted…"
                )
            except Exception:
                pass

        # If we got <100, we drained recent messages
        if len(msgs) < 100:
            break
        await asyncio.sleep(0.5)

    # Phase 2: slow delete (anything left, includes old messages)
    try:
        async for m in channel.history(limit=None):
            try:
                await m.delete()
                slow_deleted += 1
                if slow_deleted % 20 == 0:
                    if status_msg:
                        try:
                            await status_msg.edit(
                                content=f"🧹 Cleaning `#{channel.name}` — "
                                f"{bulk_deleted} fast + {slow_deleted} slow…"
                            )
                        except Exception:
                            pass
                    await asyncio.sleep(0.5)
                else:
                    await asyncio.sleep(0.2)
            except discord.NotFound:
                pass
            except Exception as e:
                failed += 1
                log.warning(f"delete failed in #{channel.name}: {e}")
    except Exception as e:
        log.warning(f"slow-pass history() failed in #{channel.name}: {e}")

    return {
        "channel": channel.name,
        "bulk": bulk_deleted,
        "slow": slow_deleted,
        "failed": failed,
        "total": bulk_deleted + slow_deleted,
    }


# ============================================================
# CONFIRMATION BUTTON UI
# ============================================================

class NukeConfirmView(ui.View):
    def __init__(self, owner_id: int, channels: list, on_confirm):
        super().__init__(timeout=60)
        self.owner_id = owner_id
        self.channels = channels
        self.on_confirm = on_confirm
        self.confirmed = False

    async def interaction_check(self, interaction):
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(
                "🔒 Only the user who started this can confirm.", ephemeral=True
            )
            return False
        return True

    @ui.button(label="💣 YES, NUKE EVERYTHING", style=discord.ButtonStyle.danger)
    async def confirm(self, interaction, button):
        self.confirmed = True
        for item in self.children:
            item.disabled = True
        await interaction.response.edit_message(
            content="💥 Starting cleanup. Don't close Discord…", view=self
        )
        self.stop()
        await self.on_confirm(interaction)

    @ui.button(label="❌ Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction, button):
        for item in self.children:
            item.disabled = True
        await interaction.response.edit_message(
            content="✅ Cancelled — nothing deleted.", view=self
        )
        self.stop()


# ============================================================
# PUBLIC API
# ============================================================

async def run_nuke(ctx, channel_names: list, get_channel_by_name,
                   post_panel=None, post_dashboard=None):
    """
    Confirm-and-wipe flow.

    channel_names: list of channel names to wipe
    get_channel_by_name: function (guild, name) -> channel
    post_panel: optional async fn(channel) -> reposts control panel
    post_dashboard: optional async fn(channel) -> reposts dashboard
    """
    guild = ctx.guild
    if guild is None:
        await ctx.send("❌ This command must run in a server.")
        return

    # Resolve channels
    targets = []
    missing = []
    for name in channel_names:
        ch = get_channel_by_name(guild, name)
        if ch:
            targets.append(ch)
        else:
            missing.append(name)

    if not targets:
        await ctx.send(f"❌ No matching channels found: {channel_names}")
        return

    desc = "\n".join(f"• `#{c.name}`" for c in targets)
    if missing:
        desc += "\n\n_Not found (skipped):_ " + ", ".join(f"`{m}`" for m in missing)

    embed = discord.Embed(
        title="💣 Channel Nuke — Confirm",
        description=(
            f"**This will delete EVERY message in:**\n{desc}\n\n"
            "⚠️ Cannot be undone. Bot needs **Manage Messages** in each channel.\n"
            "Old messages (>14 days) delete one-by-one — slower.\n\n"
            "After cleanup, the control panel and dashboard will be reposted."
        ),
        color=discord.Color.red(),
    )
    embed.set_footer(text="60 seconds to confirm")

    async def do_purge(interaction):
        status = await interaction.channel.send("🧹 Starting…")
        results = []
        for ch in targets:
            try:
                # If we're nuking the channel where this status message lives, skip it
                # to avoid deleting our own progress message.
                msg_to_keep = status.id if ch.id == status.channel.id else None
                if msg_to_keep:
                    # Wipe everything *except* the status msg by deleting all, then
                    # accepting the status msg may also vanish — we recreate after.
                    pass
                r = await purge_channel(ch, status_msg=status)
                results.append(r)
            except Exception as e:
                log.exception(f"purge failed for #{ch.name}")
                results.append({"channel": ch.name, "total": 0, "failed": -1, "error": str(e)})

        # Build summary
        lines = []
        for r in results:
            if r.get("failed") == -1:
                lines.append(f"❌ `#{r['channel']}` — {r.get('error', 'error')}")
            else:
                lines.append(
                    f"✅ `#{r['channel']}` — {r['total']} deleted "
                    f"(bulk {r['bulk']} + slow {r['slow']}, fails {r['failed']})"
                )

        summary_embed = discord.Embed(
            title="✅ Cleanup complete",
            description="\n".join(lines),
            color=discord.Color.green(),
        )

        # Re-post panel + dashboard. status_msg may have been deleted; send fresh.
        try:
            target_for_summary = get_channel_by_name(guild, "claw-bot") or status.channel
            await target_for_summary.send(embed=summary_embed)
        except Exception:
            pass

        # Repost control panel
        if post_panel:
            try:
                cb = get_channel_by_name(guild, "claw-bot")
                if cb:
                    await post_panel(cb)
            except Exception as e:
                log.warning(f"panel repost failed: {e}")

        # Repost dashboard
        if post_dashboard:
            try:
                sc = get_channel_by_name(guild, "status")
                if sc:
                    await post_dashboard(sc)
            except Exception as e:
                log.warning(f"dashboard repost failed: {e}")

    view = NukeConfirmView(ctx.author.id, targets, do_purge)
    await ctx.send(embed=embed, view=view)


async def run_nuke_single(ctx, channel_name: str, get_channel_by_name,
                          post_panel=None, post_dashboard=None):
    """Wipe just one channel."""
    await run_nuke(ctx, [channel_name], get_channel_by_name,
                   post_panel=post_panel, post_dashboard=post_dashboard)
