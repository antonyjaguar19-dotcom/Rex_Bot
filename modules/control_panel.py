"""
Claw Bot — Persistent Control Panel (button UI, default front-end)

Auto-spawns in #claw-bot on bot startup. Pinned. Shared (anyone can click).
Persistent: buttons survive bot restarts (no timeout, custom_ids).
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import discord
from discord import ui

from modules.file_utils import atomic_write_json

log = logging.getLogger("claw_bot.control_panel")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
PANEL_STATE_PATH = PROJECT_ROOT / "05_Config" / "panel_state.json"

# Dashboard URL for the 🖥️ Open Dashboard link button. Defaults to localhost
# (only reachable ON the PC). Set CLAW_DASHBOARD_URL in secrets.env to a tunnel
# URL (ngrok static / Cloudflare / Tailscale) to reach the dashboard while away.
# Load secrets.env here: this module is imported before claw_bot calls
# load_dotenv(), so without this the env var wouldn't be visible yet.
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=PROJECT_ROOT / "05_Config" / "secrets.env")
except Exception:
    pass
DASHBOARD_URL = os.environ.get("CLAW_DASHBOARD_URL", "http://127.0.0.1:7860")

# Filled in by register_views() — maps cmd_key -> async function from claw_bot.py
_CMDS: dict = {}


# ============================================================
# STATE
# ============================================================

def _load_state() -> dict:
    if not PANEL_STATE_PATH.exists():
        return {}
    try:
        return json.loads(PANEL_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(data: dict) -> None:
    PANEL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(PANEL_STATE_PATH, data)


# ============================================================
# FAKE CTX
# ============================================================

class _FakeCtx:
    def __init__(self, interaction: discord.Interaction):
        self.interaction = interaction
        self.author = interaction.user
        self.channel = interaction.channel
        self.guild = interaction.guild
        self.bot = interaction.client

    async def send(self, *args, **kwargs):
        return await self.channel.send(*args, **kwargs)

    async def reply(self, *args, **kwargs):
        return await self.channel.send(*args, **kwargs)


async def _run_cmd(interaction: discord.Interaction, cmd_key: str, *args, **kwargs):
    cmd_func = _CMDS.get(cmd_key)
    if cmd_func is None:
        await interaction.response.send_message(
            f"❌ Command `{cmd_key}` not registered.", ephemeral=True
        )
        return
    if not interaction.response.is_done():
        await interaction.response.defer(ephemeral=False, thinking=False)
    ctx = _FakeCtx(interaction)
    try:
        await cmd_func(ctx, *args, **kwargs)
    except Exception as e:
        log.exception(f"Button cmd failed: {cmd_key}")
        try:
            await interaction.followup.send(f"❌ Error: `{e}`", ephemeral=True)
        except Exception:
            pass


# ============================================================
# MODALS
# ============================================================

class ThemeModal(ui.Modal, title="✨ New Script"):
    theme = ui.TextInput(
        label="Theme (what's the story about?)",
        placeholder="e.g. sharing toys with a friend",
        max_length=200, required=True,
    )

    async def on_submit(self, interaction):
        await _run_cmd(interaction, "generate_script", theme=str(self.theme))


class ScriptIdModal(ui.Modal):
    script_id = ui.TextInput(
        label="Script ID", placeholder="e.g. 20260502_1430",
        max_length=40, required=True,
    )

    def __init__(self, title: str, cmd_key: str,
                 extra_arg_name: str = None, extra_arg_label: str = None,
                 extra_arg_placeholder: str = None):
        super().__init__(title=title)
        self.cmd_key = cmd_key
        self.extra_arg_name = extra_arg_name
        if extra_arg_name:
            self.extra_input = ui.TextInput(
                label=extra_arg_label or extra_arg_name,
                placeholder=extra_arg_placeholder or "",
                required=False, max_length=100,
            )
            self.add_item(self.extra_input)

    async def on_submit(self, interaction):
        kwargs = {"script_id": str(self.script_id)}
        if self.extra_arg_name and str(self.extra_input).strip():
            kwargs[self.extra_arg_name] = str(self.extra_input).strip()
        await _run_cmd(interaction, self.cmd_key, **kwargs)


class ValueModal(ui.Modal):
    value = ui.TextInput(label="Value", required=True, max_length=50)

    def __init__(self, title: str, label: str, placeholder: str,
                 cmd_key: str, arg_name: str, cast=str):
        super().__init__(title=title)
        self.value.label = label
        self.value.placeholder = placeholder
        self.cmd_key = cmd_key
        self.arg_name = arg_name
        self.cast = cast

    async def on_submit(self, interaction):
        try:
            val = self.cast(str(self.value).strip())
        except ValueError:
            await interaction.response.send_message(
                f"❌ Invalid value: `{self.value}`", ephemeral=True
            )
            return
        await _run_cmd(interaction, self.cmd_key, **{self.arg_name: val})


class NarrationRewriteModal(ui.Modal, title="🪄 AI Rewrite Narration"):
    script_id = ui.TextInput(
        label="Script ID", placeholder="e.g. 20260502_1430",
        required=True, max_length=40,
    )
    shot_num = ui.TextInput(
        label="Shot number", placeholder="e.g. 2",
        required=True, max_length=5,
    )
    instruction = ui.TextInput(
        label="Instruction (optional)",
        placeholder="e.g. make it funnier / shorter / calmer",
        required=False, max_length=200,
        style=discord.TextStyle.paragraph,
    )

    async def on_submit(self, interaction):
        await _run_cmd(
            interaction, "rewrite_narration",
            script_id=str(self.script_id).strip(),
            shot_num=str(self.shot_num).strip(),
            instruction=str(self.instruction).strip(),
        )


# ============================================================
# EMBEDS
# ============================================================

def _home_embed() -> discord.Embed:
    e = discord.Embed(
        title="🤖 Claw Bot — Control Panel",
        description=(
            "**Tap a category to expand.** Buttons replace typing.\n"
            "_Pinned at the top — always here when you need it._"
        ),
        color=discord.Color.blurple(),
    )
    e.add_field(name="📝 Scripts",      value="Generate · list · revise", inline=True)
    e.add_field(name="🎨 Storyboards",  value="Frames · regen", inline=True)
    e.add_field(name="🎥 Videos",       value="Render · regen · upscale", inline=True)
    e.add_field(name="⚙️ Settings",     value="Style · aspect · steps · cfg", inline=True)
    e.add_field(name="🔊 Voice & Music", value="TTS voice · BG music", inline=True)
    e.add_field(name="🔄 Models",       value="Switch backends", inline=True)
    e.add_field(name="📊 Stats",        value="Counters · status", inline=True)
    e.add_field(name="⏸️ Queue",        value="Pending · resume · drop", inline=True)
    e.add_field(name="🎬 Final",        value="Assemble · publish", inline=True)
    e.set_footer(text="Persistent panel · Claw Bot")
    return e


def _sub_embed(title: str, desc: str) -> discord.Embed:
    return discord.Embed(title=title, description=desc, color=discord.Color.blurple())


# ============================================================
# VIEWS — persistent (timeout=None, custom_id on every button)
# ============================================================

class HomeView(ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        # Link button — opens the browser dashboard. Link-style buttons carry a
        # URL (no callback / custom_id) and work inside persistent views.
        try:
            self.add_item(ui.Button(
                label="🖥️ Open Dashboard",
                style=discord.ButtonStyle.link,
                url=DASHBOARD_URL,
                row=3,
            ))
        except Exception as e:
            log.warning(f"Could not add dashboard link button: {e}")

    @ui.button(label="📝 Scripts", style=discord.ButtonStyle.primary, row=0, custom_id="cp:home:scripts")
    async def b1(self, i, b): await _switch(i, ScriptsView, "📝 Scripts", "Generate new scripts and manage existing ones.")

    @ui.button(label="🎨 Storyboards", style=discord.ButtonStyle.primary, row=0, custom_id="cp:home:stb")
    async def b2(self, i, b): await _switch(i, StoryboardsView, "🎨 Storyboards", "Generate storyboard frames per shot.")

    @ui.button(label="🎥 Videos", style=discord.ButtonStyle.primary, row=0, custom_id="cp:home:vid")
    async def b3(self, i, b): await _switch(i, VideosView, "🎥 Videos", "Render, regen, upscale clips.")

    @ui.button(label="⚙️ Settings", style=discord.ButtonStyle.secondary, row=1, custom_id="cp:home:set")
    async def b4(self, i, b): await _switch(i, SettingsView, "⚙️ Settings", "Override style, aspect ratio, steps, CFG.")

    @ui.button(label="🔊 Voice & Music", style=discord.ButtonStyle.secondary, row=1, custom_id="cp:home:vm")
    async def b5(self, i, b): await _switch(i, VoiceMusicView, "🔊 Voice & Music", "TTS voice and background music mood.")

    @ui.button(label="🔄 Models", style=discord.ButtonStyle.secondary, row=1, custom_id="cp:home:mdl")
    async def b6(self, i, b): await _switch(i, ModelsView, "🔄 Models", "Swap image / video backends.")

    @ui.button(label="📊 Stats", style=discord.ButtonStyle.success, row=2, custom_id="cp:home:stats")
    async def b7(self, i, b): await _run_cmd(i, "stats")

    @ui.button(label="⏸️ Queue", style=discord.ButtonStyle.success, row=2, custom_id="cp:home:q")
    async def b8(self, i, b): await _switch(i, QueueView, "⏸️ Queue", "Paused scripts awaiting your feedback.")

    @ui.button(label="🎬 Final", style=discord.ButtonStyle.success, row=2, custom_id="cp:home:final")
    async def b9(self, i, b): await _switch(i, FinalView, "🎬 Final Output", "Assemble all clips into the final video.")

    @ui.button(label="🛠️ System", style=discord.ButtonStyle.danger, row=3, custom_id="cp:home:sys")
    async def b10(self, i, b): await _switch(i, SystemView, "🛠️ System", "Restart (load code edits) or shut the bot down.")


async def _switch(interaction, ViewClass, title, desc):
    view = ViewClass()
    await interaction.response.edit_message(embed=_sub_embed(title, desc), view=view)


class _SubView(ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @ui.button(label="🏠 Home", style=discord.ButtonStyle.danger, row=4, custom_id="cp:sub:home")
    async def home(self, interaction, button):
        await interaction.response.edit_message(embed=_home_embed(), view=HomeView())


class ScriptsView(_SubView):
    @ui.button(label="✨ New Script", style=discord.ButtonStyle.primary, row=0, custom_id="cp:scr:new")
    async def b1(self, i, b): await i.response.send_modal(ThemeModal())

    @ui.button(label="🎯 Today's Theme", style=discord.ButtonStyle.primary, row=0, custom_id="cp:scr:today")
    async def b2(self, i, b): await _run_cmd(i, "today_script")

    @ui.button(label="💡 Suggest Theme", style=discord.ButtonStyle.secondary, row=0, custom_id="cp:scr:sug")
    async def b3(self, i, b): await _run_cmd(i, "suggest_theme")

    @ui.button(label="📋 List Scripts", style=discord.ButtonStyle.success, row=1, custom_id="cp:scr:lst")
    async def b4(self, i, b): await _run_cmd(i, "list_scripts")

    @ui.button(label="🔍 Show Script", style=discord.ButtonStyle.success, row=1, custom_id="cp:scr:show")
    async def b5(self, i, b): await i.response.send_modal(ScriptIdModal("Show Script", "show_script"))

    @ui.button(label="✨ Re-polish", style=discord.ButtonStyle.secondary, row=1, custom_id="cp:scr:rep")
    async def b6(self, i, b): await i.response.send_modal(ScriptIdModal("Re-polish Script", "repolish"))

    @ui.button(label="🪄 Rewrite Narration", style=discord.ButtonStyle.secondary, row=2, custom_id="cp:scr:narr")
    async def b7(self, i, b): await i.response.send_modal(NarrationRewriteModal())


class StoryboardsView(_SubView):
    @ui.button(label="🎨 Generate", style=discord.ButtonStyle.primary, row=0, custom_id="cp:stb:gen")
    async def b1(self, i, b): await i.response.send_modal(ScriptIdModal("Generate Storyboard", "generate_storyboard"))

    @ui.button(label="📋 List", style=discord.ButtonStyle.success, row=0, custom_id="cp:stb:lst")
    async def b2(self, i, b): await _run_cmd(i, "list_storyboards")

    @ui.button(label="🔁 Regen Shot", style=discord.ButtonStyle.secondary, row=1, custom_id="cp:stb:regen")
    async def b3(self, i, b):
        await i.response.send_modal(ScriptIdModal(
            "Regenerate Shot", "regen_shot",
            extra_arg_name="shot_num", extra_arg_label="Shot number",
            extra_arg_placeholder="e.g. 2",
        ))

class VideosView(_SubView):
    @ui.button(label="🎥 Generate", style=discord.ButtonStyle.primary, row=0, custom_id="cp:vid:gen")
    async def b1(self, i, b): await i.response.send_modal(ScriptIdModal("Generate Video", "generate_video"))

    @ui.button(label="🔁 Regen Clip", style=discord.ButtonStyle.secondary, row=0, custom_id="cp:vid:regen")
    async def b2(self, i, b):
        await i.response.send_modal(ScriptIdModal(
            "Regenerate Clip", "regen_video_shot",
            extra_arg_name="shot_num", extra_arg_label="Shot number(s)",
            extra_arg_placeholder="e.g. 2 or 2 4",
        ))

    @ui.button(label="✨ Upscale", style=discord.ButtonStyle.success, row=1, custom_id="cp:vid:up")
    async def b3(self, i, b): await i.response.send_modal(ScriptIdModal("Upscale Clips", "upscale"))

    @ui.button(label="🎬 Assemble", style=discord.ButtonStyle.success, row=1, custom_id="cp:vid:asm")
    async def b4(self, i, b): await i.response.send_modal(ScriptIdModal("Assemble Final", "assemble"))

    @ui.button(label="⏱️ Clip Length (sec/shot)", style=discord.ButtonStyle.primary, row=2, custom_id="cp:vid:clip")
    async def b5(self, i, b):
        await i.response.send_modal(ValueModal(
            "Set Clip Length", "Seconds per shot", "e.g. 8 (range 3-15)",
            "set_clip_length", "seconds", cast=float,
        ))

    @ui.button(label="🎚️ Sync Mode", style=discord.ButtonStyle.secondary, row=2, custom_id="cp:vid:sync")
    async def b6(self, i, b):
        await i.response.send_modal(ValueModal(
            "Set Sync Mode", "Mode", "strict (=TTS) / loose (=TTS+0.5s)",
            "set_sync_mode", "mode",
        ))

    @ui.button(label="🎞️ Transition", style=discord.ButtonStyle.secondary, row=2, custom_id="cp:vid:trans")
    async def b7(self, i, b):
        await i.response.send_modal(ValueModal(
            "Set Transition", "Mode", "crossfade (0.3s dissolve) / cut (hard cut)",
            "set_transition", "mode",
        ))

    @ui.button(label="📺 Video Res", style=discord.ButtonStyle.primary, row=3, custom_id="cp:vid:res")
    async def b8(self, i, b):
        await i.response.send_modal(ValueModal(
            "Set Video Resolution", "Resolution", "480p / 720p / 1080p / reset",
            "set_video_resolution", "preset",
        ))


class SettingsView(_SubView):
    @ui.button(label="🎨 Style", style=discord.ButtonStyle.primary, row=0, custom_id="cp:set:style")
    async def b1(self, i, b):
        await i.response.send_modal(ValueModal(
            "Set Style", "Style", "storybook / cartoon / anime / watercolor / pixelart",
            "set_style", "style_id",
        ))

    @ui.button(label="📐 Aspect", style=discord.ButtonStyle.primary, row=0, custom_id="cp:set:asp")
    async def b2(self, i, b):
        await i.response.send_modal(ValueModal(
            "Set Aspect Ratio", "Aspect", "16:9 / 9:16 / 1:1",
            "set_resolution", "aspect",
        ))

    @ui.button(label="⚙️ Steps", style=discord.ButtonStyle.secondary, row=1, custom_id="cp:set:steps")
    async def b3(self, i, b):
        await i.response.send_modal(ValueModal(
            "Set Steps", "Sampling steps", "e.g. 25",
            "set_steps", "steps", cast=int,
        ))

    @ui.button(label="🎲 CFG", style=discord.ButtonStyle.secondary, row=1, custom_id="cp:set:cfg")
    async def b4(self, i, b):
        await i.response.send_modal(ValueModal(
            "Set CFG", "Guidance strength", "e.g. 4.5",
            "set_cfg", "cfg", cast=float,
        ))

    @ui.button(label="📋 Show Current", style=discord.ButtonStyle.success, row=2, custom_id="cp:set:cur")
    async def b5(self, i, b): await _run_cmd(i, "current_settings")

    @ui.button(label="🎨 List Styles", style=discord.ButtonStyle.success, row=2, custom_id="cp:set:list")
    async def b6(self, i, b): await _run_cmd(i, "list_styles")

    @ui.button(label="📈 Upscale: Toggle", style=discord.ButtonStyle.primary, row=3, custom_id="cp:set:up")
    async def b8(self, i, b):
        await i.response.send_modal(ValueModal(
            "Toggle Upscale", "Upscale", "on / off",
            "set_upscale", "value",
        ))

    @ui.button(label="♻️ Reset All", style=discord.ButtonStyle.danger, row=3, custom_id="cp:set:reset")
    async def b7(self, i, b): await _run_cmd(i, "reset_settings")


class VoiceMusicView(_SubView):
    @ui.button(label="🎙️ Set Voice", style=discord.ButtonStyle.primary, row=0, custom_id="cp:vm:voice")
    async def b1(self, i, b):
        await i.response.send_modal(ValueModal(
            "Set Voice", "Voice ID", "e.g. af_heart / af_bella / am_adam",
            "set_voice", "voice_id",
        ))

    @ui.button(label="📋 List Voices", style=discord.ButtonStyle.success, row=0, custom_id="cp:vm:lst")
    async def b2(self, i, b): await _run_cmd(i, "list_voices")

    @ui.button(label="🎵 Music Mood", style=discord.ButtonStyle.primary, row=1, custom_id="cp:vm:mood")
    async def b3(self, i, b):
        await i.response.send_modal(ValueModal(
            "Set Music Mood", "Mood", "calm / cheerful / adventurous / mysterious",
            "set_music_mood", "mood",
        ))

    @ui.button(label="🔁 Regen Music", style=discord.ButtonStyle.secondary, row=1, custom_id="cp:vm:regen")
    async def b4(self, i, b): await i.response.send_modal(ScriptIdModal("Regenerate Music", "regenerate_music"))


class _ImageModelSelect(ui.Select):
    def __init__(self):
        from modules import model_registry
        try:
            current = model_registry.get_active("image_backend")
            available = model_registry.list_available("image_backend")
            current_id = current["_id"] if current else None
        except Exception:
            available, current_id = {}, None
        options = [
            discord.SelectOption(
                label=k[:100],
                description=(str(v)[:90] if v else None),
                default=(k == current_id),
            )
            for k, v in list(available.items())[:25]
        ] or [discord.SelectOption(label="(none available)")]
        super().__init__(
            placeholder="🖼️  Image backend…",
            options=options, min_values=1, max_values=1,
            custom_id="cp:mdl:sel:image",
        )

    async def callback(self, interaction):
        choice = self.values[0]
        await _run_cmd(interaction, "switch_model", backend_id=choice, backend_type="image")


class ModelsView(_SubView):
    def __init__(self):
        super().__init__()
        # Video backend is LOCKED to Wan 2.2 14B (single model, no switching).
        # Only the image backend select is exposed.
        self.add_item(_ImageModelSelect())


class QueueView(_SubView):
    @ui.button(label="📋 Pending", style=discord.ButtonStyle.primary, row=0, custom_id="cp:q:p")
    async def b1(self, i, b): await _run_cmd(i, "pending")

    @ui.button(label="▶️ Resume", style=discord.ButtonStyle.success, row=0, custom_id="cp:q:r")
    async def b2(self, i, b): await i.response.send_modal(ScriptIdModal("Resume Feedback", "resume_feedback"))

    @ui.button(label="🗑️ Drop", style=discord.ButtonStyle.danger, row=0, custom_id="cp:q:d")
    async def b3(self, i, b): await i.response.send_modal(ScriptIdModal("Drop Feedback", "drop_feedback"))


class FinalView(_SubView):
    @ui.button(label="🎬 Assemble", style=discord.ButtonStyle.primary, row=0, custom_id="cp:fin:asm")
    async def b1(self, i, b): await i.response.send_modal(ScriptIdModal("Assemble Final", "assemble"))

    @ui.button(label="✨ Upscale", style=discord.ButtonStyle.secondary, row=0, custom_id="cp:fin:up")
    async def b2(self, i, b): await i.response.send_modal(ScriptIdModal("Upscale Clips", "upscale"))


class _ConfirmView(ui.View):
    """Transient yes/no guard for destructive lifecycle actions. Not persistent
    (timeout) — these are momentary dialogs, no need to survive a restart."""
    def __init__(self, cmd_key: str):
        super().__init__(timeout=30)
        self.cmd_key = cmd_key

    @ui.button(label="✅ Confirm", style=discord.ButtonStyle.danger)
    async def confirm(self, i, b):
        await _run_cmd(i, self.cmd_key)

    @ui.button(label="✖️ Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, i, b):
        await i.response.edit_message(content="Cancelled — nothing happened.", view=None)


class SystemView(_SubView):
    @ui.button(label="♻️ Restart Bot", style=discord.ButtonStyle.primary, row=0, custom_id="cp:sys:restart")
    async def b1(self, i, b):
        await i.response.send_message(
            "♻️ **Restart the bot?**\nLoads your latest code edits. ~15s downtime. "
            "Ollama + ComfyUI keep running.",
            view=_ConfirmView("restart_bot"), ephemeral=True,
        )

    @ui.button(label="🛑 Shutdown Bot", style=discord.ButtonStyle.danger, row=0, custom_id="cp:sys:shutdown")
    async def b2(self, i, b):
        await i.response.send_message(
            "🛑 **Shut the bot down?**\nStays OFF until you relaunch from the "
            "launcher (`start_rex.bat` / launch script).",
            view=_ConfirmView("shutdown_bot"), ephemeral=True,
        )


# ============================================================
# PUBLIC API
# ============================================================

def register_views(bot: discord.Client, cmds: dict):
    """Register persistent views + command lookup. Call ONCE in on_ready."""
    global _CMDS
    _CMDS = cmds
    bot.add_view(HomeView())
    bot.add_view(ScriptsView())
    bot.add_view(StoryboardsView())
    bot.add_view(VideosView())
    bot.add_view(SettingsView())
    bot.add_view(VoiceMusicView())
    bot.add_view(ModelsView())
    bot.add_view(QueueView())
    bot.add_view(FinalView())
    bot.add_view(SystemView())
    log.info("Persistent views registered.")


async def ensure_panel(channel: discord.TextChannel) -> Optional[discord.Message]:
    """Post panel if missing, refresh + pin if existing. Returns the message."""
    state = _load_state()
    saved_id = state.get(str(channel.id))

    if saved_id:
        try:
            msg = await channel.fetch_message(int(saved_id))
            await msg.edit(embed=_home_embed(), view=HomeView())
            if not msg.pinned:
                try:
                    await msg.pin(reason="Claw Bot control panel")
                except discord.HTTPException as e:
                    log.warning(f"Could not pin panel: {e}")
            # Clean up any duplicates that snuck in
            await _purge_extra_panels(channel, keep_id=msg.id)
            log.info(f"Refreshed existing panel in #{channel.name} (msg {saved_id})")
            return msg
        except discord.NotFound:
            log.info(f"Saved panel {saved_id} not found — posting new one.")
        except Exception as e:
            log.warning(f"Could not fetch saved panel: {e}")

    try:
        msg = await channel.send(embed=_home_embed(), view=HomeView())
        try:
            await msg.pin(reason="Claw Bot control panel")
        except discord.HTTPException as e:
            log.warning(f"Could not pin panel: {e}")
        state[str(channel.id)] = msg.id
        _save_state(state)
        # Clean up any older panels left from previous bot runs
        await _purge_extra_panels(channel, keep_id=msg.id)
        log.info(f"Posted new panel in #{channel.name} (msg {msg.id})")
        return msg
    except Exception as e:
        log.exception(f"Failed to post panel: {e}")
        return None


async def open_panel_manual(ctx):
    """Manual !panel — fresh ephemeral-style copy (non-pinned)."""
    await ctx.send(embed=_home_embed(), view=HomeView())


# ============================================================
# AUTO-REPOST — keeps panel near the bottom of the channel
# ============================================================

# Per-channel cooldown so we don't repost on every single message
_REPOST_LOCK: dict = {}   # channel_id -> last repost timestamp
_REPOST_COOLDOWN = 30     # seconds — minimum gap between reposts
_REPOST_MSG_THRESHOLD = 3 # repost after this many other messages


# Per-channel async lock — prevents two repost tasks running at once
_REPOST_TASK_LOCK: dict = {}   # channel_id -> asyncio.Lock


async def _purge_extra_panels(channel: discord.TextChannel, keep_id: int):
    """Find every other panel-looking message and delete them. Keeps only keep_id."""
    try:
        async for m in channel.history(limit=50):
            if m.id == keep_id:
                continue
            if m.author.id != channel.guild.me.id:
                continue
            # Identify our panel by the embed title
            for emb in m.embeds:
                if emb.title and "Claw Bot — Control Panel" in emb.title:
                    try:
                        await m.delete()
                    except Exception:
                        pass
                    break
    except Exception as e:
        log.warning(f"purge_extra_panels failed: {e}")


async def maybe_repost_panel(channel: discord.TextChannel):
    """Reposts panel near bottom if buried. Bulletproofs against duplicates."""
    import asyncio
    import time

    # Get or create per-channel lock
    lock = _REPOST_TASK_LOCK.setdefault(channel.id, asyncio.Lock())

    # Try to grab lock without blocking — if another repost is in flight, skip
    if lock.locked():
        return

    async with lock:
        state = _load_state()
        saved_id = state.get(str(channel.id))
        if not saved_id:
            return

        now = time.time()
        last = _REPOST_LOCK.get(channel.id, 0)
        if now - last < _REPOST_COOLDOWN:
            return

        try:
            # Count messages newer than our panel
            history = [m async for m in channel.history(limit=20)]
            newer_than_panel = 0
            for m in history:
                if m.id == int(saved_id):
                    break
                newer_than_panel += 1

            if newer_than_panel < _REPOST_MSG_THRESHOLD:
                return

            # Delete old panel
            try:
                old = await channel.fetch_message(int(saved_id))
                await old.delete()
            except discord.NotFound:
                pass
            except Exception as e:
                log.warning(f"Could not delete old panel: {e}")

            # Post fresh at bottom
            msg = await channel.send(embed=_home_embed(), view=HomeView())
            try:
                await msg.pin(reason="Claw Bot control panel (auto-repost)")
            except discord.HTTPException:
                pass

            # Update state
            state[str(channel.id)] = msg.id
            _save_state(state)
            _REPOST_LOCK[channel.id] = now

            # Final safety: nuke any other panel embeds left over
            await _purge_extra_panels(channel, keep_id=msg.id)

            log.info(f"Auto-reposted panel in #{channel.name}")
        except Exception as e:
            log.warning(f"Auto-repost failed: {e}")
