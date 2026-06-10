"""
Claw Bot — Embed Styles

Central palette for color-coded phase scanning.
Use themed_embed(phase, title, ...) instead of raw discord.Embed.

Phases:
  script      → Blue
  storyboard  → Purple
  shot        → Purple (storyboard frames)
  video       → Orange
  music       → Pink
  upscale     → Gold
  final       → Green (approved/done)
  preview     → Teal (low-res preview)
  approved    → Green
  rejected    → Red
  error       → Red
  warning     → Yellow
  info        → Blurple (Discord brand)
  paused      → Light grey
"""

import discord
from typing import Optional


PHASE_COLORS = {
    "script":     discord.Color.blue(),
    "storyboard": discord.Color.purple(),
    "shot":       discord.Color.purple(),
    "video":      discord.Color.orange(),
    "music":      discord.Color.from_rgb(255, 105, 180),     # hot pink
    "upscale":    discord.Color.gold(),
    "final":      discord.Color.green(),
    "preview":    discord.Color.teal(),
    "approved":   discord.Color.green(),
    "rejected":   discord.Color.red(),
    "error":      discord.Color.red(),
    "warning":    discord.Color.from_rgb(255, 191, 0),       # amber
    "info":       discord.Color.blurple(),
    "paused":     discord.Color.light_grey(),
}

PHASE_EMOJIS = {
    "script":     "📝",
    "storyboard": "🎨",
    "shot":       "🖼️",
    "video":      "🎥",
    "music":      "🎵",
    "upscale":    "📈",
    "final":      "🎬",
    "preview":    "👀",
    "approved":   "✅",
    "rejected":   "❌",
    "error":      "⚠️",
    "warning":    "⚠️",
    "info":       "ℹ️",
    "paused":     "💾",
}


def color_for(phase: str) -> discord.Color:
    return PHASE_COLORS.get(phase, discord.Color.blurple())


def emoji_for(phase: str) -> str:
    return PHASE_EMOJIS.get(phase, "•")


def themed_embed(
    phase: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    *,
    emoji_in_title: bool = True,
) -> discord.Embed:
    """Build a phase-colored embed.

    Example:
        themed_embed("video", "Shot 3", "Rendering 5s at 720p")
    """
    if title and emoji_in_title:
        em = emoji_for(phase)
        if em and not title.startswith(em):
            title = f"{em} {title}"

    return discord.Embed(
        title=title,
        description=description,
        color=color_for(phase),
    )
