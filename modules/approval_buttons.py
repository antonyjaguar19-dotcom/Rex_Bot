"""
Claw Bot — Approval Buttons (replaces ✅❌⏹️💾 reactions)

Three view classes, one per pipeline stage:
  ScriptApprovalView      — Approve / Reject / Stop
  StoryboardApprovalView  — Approve / Reject / Stop / Pause
  VideoApprovalView       — Approve / Reject / Stop / Pause

The button callbacks delegate to handler functions registered from claw_bot.py.
This keeps existing pipeline logic intact — we only swap UI from reactions to buttons.

Reject opens a modal so the user types feedback right there, no chat-typing needed.
"""

import logging
from typing import Callable, Optional

import discord
from discord import ui

log = logging.getLogger("claw_bot.approval_buttons")

# Handler registry — filled by claw_bot.py via register_handlers()
_HANDLERS: dict = {}


def register_handlers(handlers: dict):
    """
    Call once at bot startup. Pass:
      {
        "script_approve":      async fn(interaction, script, owner_id),
        "script_reject":       async fn(interaction, script, owner_id, feedback_text),
        "script_stop":         async fn(interaction, script, owner_id),
        "storyboard_approve":  async fn(interaction, script_id, owner_id),
        "storyboard_reject":   async fn(interaction, script_id, owner_id, feedback_text),
        "storyboard_stop":     async fn(interaction, script_id, owner_id),
        "storyboard_pause":    async fn(interaction, script_id, owner_id),
        "video_approve":       async fn(interaction, script_id, owner_id),
        "video_reject":        async fn(interaction, script_id, owner_id, feedback_text),
        "video_stop":          async fn(interaction, script_id, owner_id),
        "video_pause":         async fn(interaction, script_id, owner_id),
      }
    """
    global _HANDLERS
    _HANDLERS = handlers
    log.info(f"Approval-button handlers registered: {list(handlers.keys())}")


# ============================================================
# REJECT MODAL — pops up so the user can describe what to change
# ============================================================

class _RejectModal(ui.Modal):
    feedback = ui.TextInput(
        label="What should change?",
        style=discord.TextStyle.paragraph,
        placeholder="e.g. make the ending happier, change the cat to a dog…",
        max_length=1000,
        required=True,
    )

    def __init__(self, title: str, handler_key: str, payload: dict):
        super().__init__(title=title)
        self.handler_key = handler_key
        self.payload = payload

    async def on_submit(self, interaction):
        # Acknowledge IMMEDIATELY so Discord doesn't time out
        try:
            await interaction.response.defer(ephemeral=False, thinking=False)
        except Exception:
            pass

        handler = _HANDLERS.get(self.handler_key)
        if not handler:
            try:
                await interaction.followup.send(
                    f"❌ Handler `{self.handler_key}` not registered.", ephemeral=True
                )
            except Exception:
                pass
            return
        try:
            await handler(interaction, **self.payload, feedback_text=str(self.feedback))
        except Exception as e:
            log.exception("Modal handler crashed")
            try:
                await interaction.followup.send(f"❌ Error: `{e}`", ephemeral=True)
            except Exception:
                pass


# ============================================================
# BASE VIEW — common ownership / disable-after-click logic
# ============================================================

class _ApprovalBase(ui.View):
    """All approval views inherit from this. timeout=None = persists across bot restarts.

    Every button MUST set a stable custom_id for Discord to route post-restart
    interactions back to the right view.
    """

    def __init__(self, owner_id: int):
        super().__init__(timeout=None)  # persistent — survives restart
        self.owner_id = owner_id
        self.resolved = False  # set True after any decision so we don't double-click

    async def interaction_check(self, interaction):
        if self.resolved:
            await interaction.response.send_message(
                "ℹ️ This decision was already made.", ephemeral=True
            )
            return False
        # Owner OR admin can click
        if interaction.user.id != self.owner_id:
            if not interaction.user.guild_permissions.administrator:
                await interaction.response.send_message(
                    "🔒 Only the requester (or an admin) can decide on this.", ephemeral=True
                )
                return False
        return True

    async def _disable_all(self, interaction, status_emoji: str, status_text: str):
        """Grey out all buttons + add a status note. Edits the message in place."""
        self.resolved = True
        for child in self.children:
            child.disabled = True

        # Pull the original embed/content, append a status line
        msg = interaction.message
        new_content = msg.content
        if new_content:
            new_content = f"{new_content}\n\n{status_emoji} **{status_text}**"
        else:
            new_content = f"{status_emoji} **{status_text}**"

        try:
            await interaction.response.edit_message(content=new_content, view=self)
        except discord.InteractionResponded:
            # Already responded (e.g. modal); edit via followup
            await msg.edit(content=new_content, view=self)


# ============================================================
# SCRIPT APPROVAL
# ============================================================

class ScriptApprovalView(_ApprovalBase):
    def __init__(self, script: dict, owner_id: int):
        super().__init__(owner_id)
        self.script = script

    @ui.button(label="✅ Approve", style=discord.ButtonStyle.success, custom_id="script:approve")
    async def approve(self, interaction, button):
        await self._disable_all(interaction, "✅", f"Approved by {interaction.user.mention}")
        handler = _HANDLERS.get("script_approve")
        if handler:
            await handler(interaction, script=self.script, owner_id=self.owner_id)

    @ui.button(label="❌ Reject", style=discord.ButtonStyle.danger, custom_id="script:reject")
    async def reject(self, interaction, button):
        # Pop modal for feedback BEFORE disabling buttons
        modal = _RejectModal(
            title="What should change?",
            handler_key="script_reject",
            payload={"script": self.script, "owner_id": self.owner_id},
        )
        await interaction.response.send_modal(modal)
        # Mark resolved after modal sends (don't disable buttons until user submits)
        await modal.wait()
        # Modal handler will trigger the revision flow on submit

    @ui.button(label="⏹ Stop", style=discord.ButtonStyle.secondary, custom_id="script:stop")
    async def stop(self, interaction, button):
        await self._disable_all(interaction, "🛑", f"Stopped by {interaction.user.mention}")
        handler = _HANDLERS.get("script_stop")
        if handler:
            await handler(interaction, script=self.script, owner_id=self.owner_id)


# ============================================================
# STORYBOARD APPROVAL
# ============================================================

class StoryboardApprovalView(_ApprovalBase):
    def __init__(self, script_id: str, owner_id: int):
        super().__init__(owner_id)
        self.script_id = script_id

    @ui.button(label="✅ Approve", style=discord.ButtonStyle.success, custom_id="storyboard:approve")
    async def approve(self, interaction, button):
        await self._disable_all(interaction, "✅", f"Approved by {interaction.user.mention}")
        handler = _HANDLERS.get("storyboard_approve")
        if handler:
            await handler(interaction, script_id=self.script_id, owner_id=self.owner_id)

    @ui.button(label="❌ Reject", style=discord.ButtonStyle.danger, custom_id="storyboard:reject")
    async def reject(self, interaction, button):
        modal = _RejectModal(
            title="What should change in the storyboard?",
            handler_key="storyboard_reject",
            payload={"script_id": self.script_id, "owner_id": self.owner_id},
        )
        await interaction.response.send_modal(modal)
        await modal.wait()

    @ui.button(label="⏹ Stop", style=discord.ButtonStyle.secondary, custom_id="storyboard:stop")
    async def stop(self, interaction, button):
        await self._disable_all(interaction, "🛑", f"Stopped by {interaction.user.mention}")
        handler = _HANDLERS.get("storyboard_stop")
        if handler:
            await handler(interaction, script_id=self.script_id, owner_id=self.owner_id)

    @ui.button(label="💾 Pause", style=discord.ButtonStyle.secondary, custom_id="storyboard:pause")
    async def pause(self, interaction, button):
        await self._disable_all(interaction, "💾", f"Paused by {interaction.user.mention}")
        handler = _HANDLERS.get("storyboard_pause")
        if handler:
            await handler(interaction, script_id=self.script_id, owner_id=self.owner_id)


# ============================================================
# VIDEO APPROVAL
# ============================================================

class VideoApprovalView(_ApprovalBase):
    def __init__(self, script_id: str, owner_id: int):
        super().__init__(owner_id)
        self.script_id = script_id

    @ui.button(label="✅ Approve", style=discord.ButtonStyle.success, custom_id="video:approve")
    async def approve(self, interaction, button):
        await self._disable_all(interaction, "✅", f"Approved by {interaction.user.mention}")
        handler = _HANDLERS.get("video_approve")
        if handler:
            await handler(interaction, script_id=self.script_id, owner_id=self.owner_id)

    @ui.button(label="❌ Reject", style=discord.ButtonStyle.danger, custom_id="video:reject")
    async def reject(self, interaction, button):
        modal = _RejectModal(
            title="What should change in the video?",
            handler_key="video_reject",
            payload={"script_id": self.script_id, "owner_id": self.owner_id},
        )
        await interaction.response.send_modal(modal)
        await modal.wait()

    @ui.button(label="⏹ Stop", style=discord.ButtonStyle.secondary, custom_id="video:stop")
    async def stop(self, interaction, button):
        await self._disable_all(interaction, "🛑", f"Stopped by {interaction.user.mention}")
        handler = _HANDLERS.get("video_stop")
        if handler:
            await handler(interaction, script_id=self.script_id, owner_id=self.owner_id)

    @ui.button(label="💾 Pause", style=discord.ButtonStyle.secondary, custom_id="video:pause")
    async def pause(self, interaction, button):
        await self._disable_all(interaction, "💾", f"Paused by {interaction.user.mention}")
        handler = _HANDLERS.get("video_pause")
        if handler:
            await handler(interaction, script_id=self.script_id, owner_id=self.owner_id)

# ============================================================
# SHOT CONTROL VIEW — appears under each storyboard frame
# ============================================================

class _RegenShotModal(ui.Modal, title="🔁 Regenerate Shot"):
    notes = ui.TextInput(
        label="Notes (optional — leave blank to regen as-is)",
        style=discord.TextStyle.paragraph,
        placeholder="e.g. make her smile more, different background…",
        max_length=500,
        required=False,
    )

    def __init__(self, script_id: str, shot_number: int, owner_id: int):
        super().__init__()
        self.script_id = script_id
        self.shot_number = shot_number
        self.owner_id = owner_id

    async def on_submit(self, interaction):
        try:
            await interaction.response.defer(ephemeral=False, thinking=False)
        except Exception:
            pass

        handler = _HANDLERS.get("shot_regen")
        if not handler:
            await interaction.followup.send(
                "❌ Shot regen handler not registered.", ephemeral=True
            )
            return
        try:
            await handler(
                interaction,
                script_id=self.script_id,
                shot_number=self.shot_number,
                owner_id=self.owner_id,
                notes=str(self.notes),
            )
        except Exception as e:
            log.exception("Shot regen failed")
            try:
                await interaction.followup.send(f"❌ Error: `{e}`", ephemeral=True)
            except Exception:
                pass


class _EditPromptThenRegenModal(ui.Modal):
    """
    Pre-fills the user-approved prompt for a shot, accepts edits, persists,
    then dispatches the appropriate regen handler.

    kind: "image" -> updates approved_prompts.json image_prompt, calls shot_regen
          "motion" -> updates approved_prompts.json motion_prompt, calls clip_regen
    """

    def __init__(
        self, *, script_id: str, shot_number: int, owner_id: int, kind: str,
        current_prompt: str,
    ):
        title = (
            f"✏️ Edit {'image' if kind == 'image' else 'motion'} prompt — shot {shot_number}"
        )
        super().__init__(title=title[:45])
        self.script_id = script_id
        self.shot_number = shot_number
        self.owner_id = owner_id
        self.kind = kind
        self.prompt = ui.TextInput(
            label="Prompt (edit then submit to regen)",
            style=discord.TextStyle.paragraph,
            default=(current_prompt or "")[:4000],
            max_length=4000,
            required=True,
        )
        self.add_item(self.prompt)

    async def on_submit(self, interaction):
        try:
            await interaction.response.defer(ephemeral=False, thinking=False)
        except Exception:
            pass

        handler_key = "shot_edit_prompt" if self.kind == "image" else "clip_edit_prompt"
        handler = _HANDLERS.get(handler_key)
        if not handler:
            await interaction.followup.send(
                f"❌ Handler `{handler_key}` not registered.", ephemeral=True,
            )
            return
        try:
            await handler(
                interaction,
                script_id=self.script_id,
                shot_number=self.shot_number,
                owner_id=self.owner_id,
                new_prompt=str(self.prompt),
            )
        except Exception as e:
            log.exception(f"{handler_key} failed")
            try:
                await interaction.followup.send(f"❌ Error: `{e}`", ephemeral=True)
            except Exception:
                pass


class ShotControlView(ui.View):
    """Per-frame buttons under each storyboard shot.
       Edit prompt OR regen as-is.
    """

    def __init__(self, script_id: str, shot_number: int, owner_id: int):
        super().__init__(timeout=86400)
        self.script_id = script_id
        self.shot_number = shot_number
        self.owner_id = owner_id

    async def interaction_check(self, interaction):
        if interaction.user.id != self.owner_id:
            if not interaction.user.guild_permissions.administrator:
                await interaction.response.send_message(
                    "🔒 Only the requester (or an admin) can do this.", ephemeral=True
                )
                return False
        return True

    @ui.button(label="✏️ Edit Prompt & Regen", style=discord.ButtonStyle.primary)
    async def edit_prompt(self, interaction, button):
        # Lazy import to avoid module cycle
        from modules import prompt_approval as pap
        approved = pap.get_shot_prompts(self.script_id, self.shot_number) or {}
        current = (approved.get("image_prompt") or "").strip()
        modal = _EditPromptThenRegenModal(
            script_id=self.script_id,
            shot_number=self.shot_number,
            owner_id=self.owner_id,
            kind="image",
            current_prompt=current,
        )
        await interaction.response.send_modal(modal)

    @ui.button(label="🔁 Regen this shot", style=discord.ButtonStyle.secondary)
    async def regen(self, interaction, button):
        modal = _RegenShotModal(
            script_id=self.script_id,
            shot_number=self.shot_number,
            owner_id=self.owner_id,
        )
        await interaction.response.send_modal(modal)


class ClipControlView(ui.View):
    """Per-clip buttons under each generated video clip.
       Edit motion prompt OR regen as-is.
    """

    def __init__(self, script_id: str, shot_number: int, owner_id: int):
        super().__init__(timeout=86400)
        self.script_id = script_id
        self.shot_number = shot_number
        self.owner_id = owner_id

    async def interaction_check(self, interaction):
        if interaction.user.id != self.owner_id:
            if not interaction.user.guild_permissions.administrator:
                await interaction.response.send_message(
                    "🔒 Only the requester (or an admin) can do this.", ephemeral=True,
                )
                return False
        return True

    @ui.button(label="✏️ Edit Motion & Regen", style=discord.ButtonStyle.primary)
    async def edit_motion(self, interaction, button):
        from modules import prompt_approval as pap
        approved = pap.get_shot_prompts(self.script_id, self.shot_number) or {}
        current = (approved.get("motion_prompt") or "").strip()
        modal = _EditPromptThenRegenModal(
            script_id=self.script_id,
            shot_number=self.shot_number,
            owner_id=self.owner_id,
            kind="motion",
            current_prompt=current,
        )
        await interaction.response.send_modal(modal)

    @ui.button(label="🔁 Regen this clip", style=discord.ButtonStyle.secondary)
    async def regen(self, interaction, button):
        await interaction.response.defer(ephemeral=False, thinking=False)
        handler = _HANDLERS.get("clip_regen")
        if not handler:
            await interaction.followup.send(
                "❌ clip_regen handler not registered.", ephemeral=True,
            )
            return
        try:
            await handler(
                interaction,
                script_id=self.script_id,
                shot_number=self.shot_number,
                owner_id=self.owner_id,
            )
        except Exception as e:
            log.exception("clip_regen failed")
            try:
                await interaction.followup.send(f"❌ Error: `{e}`", ephemeral=True)
            except Exception:
                pass
