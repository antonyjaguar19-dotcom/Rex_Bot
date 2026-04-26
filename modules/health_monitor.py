"""
Claw Bot — Service Health Monitor

Periodic checks on ComfyUI + Ollama + GPU.
Posts/updates a status dashboard message in #status channel.
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
import discord

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import model_registry
from modules import gpu_utils

log = logging.getLogger("claw_bot.health_monitor")


def check_comfyui() -> tuple[bool, str]:
    """Returns (up, short_status_string)."""
    try:
        cfg = model_registry.get_active("image_backend")
        url = cfg.get("server_url", "http://127.0.0.1:8188").rstrip("/")
        r = requests.get(f"{url}/system_stats", timeout=3)
        if r.status_code == 200:
            return True, "online"
        return False, f"HTTP {r.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "offline"
    except Exception as e:
        return False, f"error: {type(e).__name__}"


def check_ollama() -> tuple[bool, str]:
    """Returns (up, short_status_string)."""
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        if r.status_code == 200:
            data = r.json()
            models = data.get("models", [])
            return True, f"online ({len(models)} model(s))"
        return False, f"HTTP {r.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "offline"
    except Exception as e:
        return False, f"error: {type(e).__name__}"


def build_status_embed(bot_version: str, bot_start_time: datetime,
                       comfy_state: tuple[bool, str],
                       ollama_state: tuple[bool, str],
                       stats: dict) -> discord.Embed:
    """Build a rich embed showing current system state."""
    now = datetime.now(timezone.utc)
    uptime_sec = int((now - bot_start_time).total_seconds())
    hours, remainder = divmod(uptime_sec, 3600)
    minutes, _ = divmod(remainder, 60)
    uptime_str = f"{hours}h {minutes}m"

    comfy_up, comfy_msg = comfy_state
    ollama_up, ollama_msg = ollama_state
    all_healthy = comfy_up and ollama_up

    color = discord.Color.green() if all_healthy else (
        discord.Color.orange() if (comfy_up or ollama_up) else discord.Color.red()
    )

    embed = discord.Embed(
        title="🤖 Claw Bot — System Status",
        description=f"Last updated: <t:{int(now.timestamp())}:R>",
        color=color,
    )

    services_lines = [
        f"{'🟢' if comfy_up else '🔴'} **ComfyUI** — {comfy_msg}",
        f"{'🟢' if ollama_up else '🔴'} **Ollama** — {ollama_msg}",
        f"🟢 **Bot** — online · v{bot_version} · uptime {uptime_str}",
    ]
    embed.add_field(name="Services", value="\n".join(services_lines), inline=False)

    try:
        img_cfg = model_registry.get_active("image_backend")
        llm_cfg = model_registry.get_active("llm_backend")
        backend_lines = [
            f"🎨 **Image:** `{img_cfg['_id']}`",
            f"🧠 **LLM:** `{llm_cfg.get('_id', 'unknown')}`",
        ]
        embed.add_field(name="Active Models", value="\n".join(backend_lines), inline=True)
    except Exception as e:
        embed.add_field(name="Active Models", value=f"error: {e}", inline=True)

    vram = gpu_utils.get_vram_stats()
    if "error" not in vram:
        vram_lines = [
            f"💾 **VRAM:** {vram['used_mb']} / {vram['total_mb']} MB",
            f"🆓 **Free:** {vram['free_gb']} GB",
            f"⚡ **Util:** {vram['util_percent']}%",
        ]
        embed.add_field(name="GPU", value="\n".join(vram_lines), inline=True)
    else:
        embed.add_field(name="GPU", value=f"nvidia-smi: {vram['error']}", inline=True)

    stats_lines = [
        f"📝 Scripts: `{stats.get('generated', 0)}` gen · `{stats.get('approved', 0)}` ok · `{stats.get('rejected', 0)}` ❌",
        f"🎬 Storyboards: `{stats.get('storyboards_generated', 0)}` gen · `{stats.get('storyboards_approved', 0)}` ok · `{stats.get('storyboards_rejected', 0)}` ❌",
    ]
    embed.add_field(name="Session Stats", value="\n".join(stats_lines), inline=False)

    embed.set_footer(text="Dashboard auto-refreshes every 30s")
    return embed


class StatusDashboard:
    """Posts + auto-updates a status embed in #status channel."""

    def __init__(self, bot, bot_version: str, stats_getter):
        self.bot = bot
        self.bot_version = bot_version
        self.stats_getter = stats_getter
        self.message: Optional[discord.Message] = None
        self.start_time = datetime.now(timezone.utc)
        self._task: Optional[asyncio.Task] = None

    async def start(self, channel: discord.TextChannel):
        embed = build_status_embed(
            self.bot_version, self.start_time,
            check_comfyui(), check_ollama(),
            self.stats_getter(),
        )
        self.message = await channel.send(embed=embed)
        self._task = self.bot.loop.create_task(self._refresh_loop())
        log.info(f"Status dashboard posted in #{channel.name}")

    async def _refresh_loop(self):
        while True:
            try:
                await asyncio.sleep(30)
                if self.message is None:
                    continue
                embed = build_status_embed(
                    self.bot_version, self.start_time,
                    check_comfyui(), check_ollama(),
                    self.stats_getter(),
                )
                await self.message.edit(embed=embed)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(f"Dashboard refresh failed: {e}")

    async def stop(self):
        if self._task:
            self._task.cancel()