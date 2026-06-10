"""
Claw Bot — Prompt Approval Workflow

After a script is approved, but BEFORE any image or video is rendered, this
module asks Qwen to write the final image prompt AND the final motion prompt
for every shot, then posts them to Discord with per-shot buttons:

    [✏️ Edit Image]  [🎲 Reseed Image]
    [✏️ Edit Motion] [🎲 Reseed Motion]
    [✅ Approve Shot]

The user can edit any prompt via modal popup (auto-randomizes seed on edit)
or just change the seed without touching the prompt. Once every shot is
"Approve Shot" locked, a single "✅ Approve ALL & Render" button in the
footer message kicks off the storyboard pipeline.

Storyboard generator and clip generator read the persisted approved prompts
from disk:
    04_Outputs/approved_prompts/{script_id}.json
"""

import asyncio
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import discord
from discord import ui

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import prompt_assembler as pa
from modules import script_generator as sg

log = logging.getLogger("claw_bot.prompt_approval")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
APPROVED_PROMPTS_DIR = PROJECT_ROOT / "04_Outputs" / "approved_prompts"
SCRIPTS_DIR = PROJECT_ROOT / "04_Outputs" / "scripts"
STORYBOARDS_DIR = PROJECT_ROOT / "04_Outputs" / "storyboards"


# ==============================================================================
# DISK PERSISTENCE
# ==============================================================================

def approved_prompts_path(script_id: str) -> Path:
    return APPROVED_PROMPTS_DIR / f"{script_id}.json"


def load_approved_prompts(script_id: str) -> Optional[dict]:
    """Return the persisted approved prompts dict, or None if no approval yet."""
    path = approved_prompts_path(script_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"Could not parse approved prompts for {script_id}: {e}")
        return None


def get_shot_prompts(script_id: str, shot_number: int) -> Optional[dict]:
    """Return {image_prompt, image_seed, motion_prompt, motion_seed} for one shot."""
    data = load_approved_prompts(script_id)
    if not data:
        return None
    return (data.get("prompts") or {}).get(str(shot_number))


def _save(state: dict):
    path = approved_prompts_path(state["script_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def backfill_from_disk(script_id: str, script: Optional[dict] = None) -> Optional[dict]:
    """Build an editable approved_prompts state from already-rendered artifacts —
    the storyboard manifest (exact image prompt + seed per shot) and the script
    (motion prompt) — WITHOUT calling the LLM.

    This is what lets OLD scripts (rendered before / without the prompt-approval
    step, so they have storyboard images but no approved_prompts JSON) be edited
    and regenerated after a bot restart. If a JSON already exists it is returned
    untouched. Returns None if there is nothing on disk to rebuild from.
    """
    existing = load_approved_prompts(script_id)
    if existing and existing.get("prompts"):
        return existing

    if script is None:
        sp = SCRIPTS_DIR / f"script_{script_id}.json"
        if sp.exists():
            try:
                script = json.loads(sp.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning(f"backfill: could not read script {script_id}: {e}")
                script = None
    shots = (script or {}).get("shots", []) or []

    # Storyboard manifest → the actual image prompt + seed that produced each frame
    manifest: dict[int, dict] = {}
    man_path = STORYBOARDS_DIR / script_id / "storyboard.json"
    if man_path.exists():
        try:
            md = json.loads(man_path.read_text(encoding="utf-8"))
            for fr in md.get("frames", []):
                sn = fr.get("shot_number")
                if sn is not None:
                    manifest[int(sn)] = fr
        except Exception as e:
            log.warning(f"backfill: bad manifest for {script_id}: {e}")

    if not shots and not manifest:
        return None

    shot_by_num = {s.get("shot_number"): s for s in shots if s.get("shot_number")}
    nums = sorted(set(shot_by_num.keys()) | set(manifest.keys()))

    out = {
        "script_id": script_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "_backfilled": True,
        "prompts": {},
    }
    for sn in nums:
        sh = shot_by_num.get(sn, {})
        fr = manifest.get(sn, {})
        img_p = (
            (fr.get("prompt") or "").strip()
            or (sh.get("first_frame_prompt") or "").strip()
            or (sh.get("visual_description") or "").strip()
        )
        img_seed = fr.get("seed", -1)
        if not isinstance(img_seed, int):
            img_seed = -1
        mot_p = (
            (sh.get("motion_prompt") or "").strip()
            or (sh.get("visual_description") or "").strip()
        )
        if mot_p and "no speech" not in mot_p.lower():
            mot_p = mot_p.rstrip(".") + ". No speech. No mouth movement."
        out["prompts"][str(sn)] = {
            "image_prompt": img_p,
            "image_seed": img_seed,
            "motion_prompt": mot_p,
            "motion_seed": -1,
            "beat": (sh.get("beat") or "").strip().lower() or "hook",
            "narration": (sh.get("narration") or "").strip(),
            "approved": False,
        }

    if not out["prompts"]:
        return None
    _save(out)
    _ACTIVE_STATES[script_id] = out
    log.info(f"Backfilled approved_prompts for {script_id} from disk ({len(out['prompts'])} shots).")
    return out


def _random_seed() -> int:
    return random.randint(1, 2_147_483_647)


# ==============================================================================
# PROMPT GENERATION (one Qwen call per shot, image + motion)
# ==============================================================================

def _style_suffix_for(script: dict) -> str:
    style_id = script.get("style") or sg.get_default_style()
    style_info = sg.get_style_description(style_id)
    return style_info.get("prompt_suffix", "")


def generate_all_prompts(script: dict, progress: Optional[Callable[[str, int, int], None]] = None) -> dict:
    """
    Run prompt_assembler for every shot. Returns the state dict that gets
    persisted to disk. Image + motion prompts are generated up-front; user
    edits/reseeds happen against this state.

    Shape:
        {
          "script_id": "...",
          "created_at": iso,
          "prompts": {
            "1": {
              "image_prompt": "...",
              "image_seed": -1,
              "motion_prompt": "...",
              "motion_seed": -1,
              "beat": "atmosphere",
              "narration": "...",
              "approved": False
            },
            ...
          }
        }
    """
    script_id = script.get("_id") or script.get("script_id") or "unknown"
    shots = script.get("shots", []) or []
    style_suffix = _style_suffix_for(script)

    out = {
        "script_id": script_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompts": {},
    }

    total = len(shots)
    for idx, shot in enumerate(shots, start=1):
        shot_num = shot.get("shot_number", idx)
        beat = (shot.get("beat") or "").strip().lower() or "hook"
        if progress:
            try:
                progress(f"Shot {shot_num} ({beat}) — image prompt", idx, total)
            except Exception:
                pass

        # Image prompt
        try:
            img_p = pa.assemble_image_prompt(
                script=script, shot=shot,
                style_suffix=style_suffix, frame_type="first",
            )
        except Exception as e:
            log.warning(f"Shot {shot_num} image prompt LLM failed: {e}; using raw")
            img_p = (shot.get("first_frame_prompt") or shot.get("visual_description") or "").strip()

        # Motion prompt (uses approved image as ground truth)
        if progress:
            try:
                progress(f"Shot {shot_num} ({beat}) — motion prompt", idx, total)
            except Exception:
                pass
        try:
            mot_p = pa.assemble_motion_prompt(
                script=script, shot=shot, approved_image_prompt=img_p,
            )
        except Exception as e:
            log.warning(f"Shot {shot_num} motion prompt LLM failed: {e}; using raw")
            mot_p = (
                (shot.get("motion_prompt") or "").strip()
                or (shot.get("action") or "").strip()
                or (shot.get("visual_description") or "").strip()
            )
            if "no speech" not in mot_p.lower():
                mot_p = mot_p.rstrip(".") + ". No speech. No mouth movement."

        out["prompts"][str(shot_num)] = {
            "image_prompt": img_p,
            "image_seed": -1,
            "motion_prompt": mot_p,
            "motion_seed": -1,
            "beat": beat,
            "narration": (shot.get("narration") or "").strip(),
            "approved": False,
        }

    return out


# ==============================================================================
# DISCORD UI — per-shot buttons + modal
# ==============================================================================

# Live state cache so views share the same in-memory dict. Disk is the source
# of truth but loading + writing on every button click is wasteful.
_ACTIVE_STATES: dict[str, dict] = {}  # script_id -> state


def _get_state(script_id: str) -> Optional[dict]:
    state = _ACTIVE_STATES.get(script_id)
    if state is None:
        state = load_approved_prompts(script_id)
        if state:
            _ACTIVE_STATES[script_id] = state
    return state


# Master views keyed by script_id so per-shot buttons can refresh them.
_MASTER_VIEWS: dict[str, "_MasterApprovalView"] = {}


class _PromptEditModal(ui.Modal):
    def __init__(
        self, *, script_id: str, shot_num: int, prompt_kind: str,
        current_value: str, on_done: Callable,
    ):
        super().__init__(title=f"Edit {prompt_kind} prompt — shot {shot_num}")
        self.script_id = script_id
        self.shot_num = shot_num
        self.prompt_kind = prompt_kind  # "image" or "motion"
        self.on_done = on_done
        self.text = ui.TextInput(
            label=f"{prompt_kind.capitalize()} prompt",
            style=discord.TextStyle.paragraph,
            default=(current_value or "")[:4000],
            max_length=4000,
            required=True,
        )
        self.add_item(self.text)

    async def on_submit(self, interaction):
        try:
            await interaction.response.defer(ephemeral=False, thinking=False)
        except Exception:
            pass
        try:
            await self.on_done(interaction, str(self.text))
        except Exception as e:
            log.exception("Prompt edit modal failed")
            try:
                await interaction.followup.send(f"❌ Edit failed: `{e}`", ephemeral=True)
            except Exception:
                pass


class _SeedEditModal(ui.Modal):
    def __init__(
        self, *, script_id: str, shot_num: int, prompt_kind: str,
        current_value: int, on_done: Callable,
    ):
        super().__init__(title=f"Set {prompt_kind} seed — shot {shot_num}")
        self.script_id = script_id
        self.shot_num = shot_num
        self.prompt_kind = prompt_kind
        self.on_done = on_done
        default_text = "" if current_value in (None, -1) else str(current_value)
        self.text = ui.TextInput(
            label=f"{prompt_kind.capitalize()} seed (blank = random)",
            style=discord.TextStyle.short,
            default=default_text,
            placeholder="e.g. 12345 — leave blank for random",
            max_length=12,
            required=False,
        )
        self.add_item(self.text)

    async def on_submit(self, interaction):
        try:
            await interaction.response.defer(ephemeral=False, thinking=False)
        except Exception:
            pass
        raw = str(self.text).strip()
        try:
            if not raw:
                new_seed = -1
            else:
                new_seed = int(raw)
                if new_seed < 0:
                    new_seed = -1
        except ValueError:
            try:
                await interaction.followup.send(
                    f"❌ `{raw}` is not a valid integer.", ephemeral=True,
                )
            except Exception:
                pass
            return
        try:
            await self.on_done(interaction, new_seed)
        except Exception as e:
            log.exception("Seed edit modal failed")
            try:
                await interaction.followup.send(f"❌ Seed update failed: `{e}`", ephemeral=True)
            except Exception:
                pass


def _shot_embed(script_id: str, shot_num: int, shot_state: dict) -> discord.Embed:
    approved = shot_state.get("approved", False)
    color = discord.Color.green() if approved else discord.Color.orange()
    title = (
        f"Shot {shot_num} — {shot_state.get('beat', '?')} "
        f"{'✅ APPROVED' if approved else '✏️ EDITABLE'}"
    )
    embed = discord.Embed(title=title, color=color)

    img_p = shot_state.get("image_prompt", "")[:1000]
    img_seed = shot_state.get("image_seed", -1)
    mot_p = shot_state.get("motion_prompt", "")[:1000]
    mot_seed = shot_state.get("motion_seed", -1)

    seed_label = lambda s: "RANDOM" if s in (None, -1) else str(s)
    embed.add_field(
        name=f"🖼️ Image prompt (seed: {seed_label(img_seed)})",
        value=f"```{img_p or '(empty)'}```",
        inline=False,
    )
    embed.add_field(
        name=f"🎥 Motion prompt (seed: {seed_label(mot_seed)})",
        value=f"```{mot_p or '(empty)'}```",
        inline=False,
    )
    narr = shot_state.get("narration", "")
    if narr:
        embed.add_field(name="🎙️ Narration", value=f"_{narr[:300]}_", inline=False)
    embed.set_footer(text=f"script {script_id} · shot {shot_num}")
    return embed


class _ShotApprovalView(ui.View):
    """Per-shot buttons. One of these lives under each shot's embed."""

    def __init__(self, *, script_id: str, shot_num: int, owner_id: int, message_ref_holder: dict):
        super().__init__(timeout=86400)
        self.script_id = script_id
        self.shot_num = shot_num
        self.owner_id = owner_id
        # Holder lets us discover the message after first send (set by caller)
        self.message_ref_holder = message_ref_holder

    async def interaction_check(self, interaction):
        if interaction.user.id != self.owner_id and not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message(
                "🔒 Only the requester (or an admin) can edit.", ephemeral=True,
            )
            return False
        return True

    async def _refresh_message(self, interaction):
        state = _get_state(self.script_id)
        if not state:
            return
        shot_state = state["prompts"].get(str(self.shot_num))
        if not shot_state:
            return
        embed = _shot_embed(self.script_id, self.shot_num, shot_state)
        # Disable all buttons on this view when shot is approved
        for c in self.children:
            c.disabled = bool(shot_state.get("approved"))
        msg = self.message_ref_holder.get("msg")
        if msg is None:
            try:
                msg = interaction.message
            except Exception:
                msg = None
        if msg is not None:
            try:
                await msg.edit(embed=embed, view=self)
            except Exception as e:
                log.warning(f"Could not refresh shot {self.shot_num} message: {e}")

        # Also refresh the master view (button enable state)
        master = _MASTER_VIEWS.get(self.script_id)
        if master is not None:
            await master.refresh()

    def _update_shot(self, mutator: Callable[[dict], None]):
        state = _get_state(self.script_id)
        if not state:
            raise RuntimeError(f"No state for script {self.script_id}")
        shot_state = state["prompts"].setdefault(str(self.shot_num), {})
        # Editing a shot un-approves it (so the user must re-confirm)
        was_approved = shot_state.get("approved", False)
        mutator(shot_state)
        if was_approved:
            shot_state["approved"] = False
        _save(state)
        return shot_state

    @ui.button(label="✏️ Edit Image", style=discord.ButtonStyle.primary, row=0)
    async def edit_image(self, interaction, button):
        state = _get_state(self.script_id)
        shot_state = state["prompts"].get(str(self.shot_num), {})

        async def on_done(inter, new_text):
            def mut(s):
                s["image_prompt"] = new_text.strip()
                # Edit = auto-randomize seed (user asked for "new seed value on edit")
                s["image_seed"] = _random_seed()
            self._update_shot(mut)
            await self._refresh_message(inter)
            await inter.followup.send(
                f"✏️ Shot {self.shot_num} image prompt updated · new seed assigned.",
                ephemeral=True,
            )

        modal = _PromptEditModal(
            script_id=self.script_id, shot_num=self.shot_num,
            prompt_kind="image",
            current_value=shot_state.get("image_prompt", ""),
            on_done=on_done,
        )
        await interaction.response.send_modal(modal)

    @ui.button(label="🎲 Reseed Image", style=discord.ButtonStyle.secondary, row=0)
    async def reseed_image(self, interaction, button):
        def mut(s):
            s["image_seed"] = _random_seed()
        shot_state = self._update_shot(mut)
        await interaction.response.defer(ephemeral=False, thinking=False)
        await self._refresh_message(interaction)
        await interaction.followup.send(
            f"🎲 Shot {self.shot_num} image seed → `{shot_state['image_seed']}`.",
            ephemeral=True,
        )

    @ui.button(label="🔢 Set Image Seed", style=discord.ButtonStyle.secondary, row=0)
    async def set_image_seed(self, interaction, button):
        shot_state = _get_state(self.script_id)["prompts"].get(str(self.shot_num), {})

        async def on_done(inter, new_seed):
            def mut(s):
                s["image_seed"] = new_seed
            self._update_shot(mut)
            await self._refresh_message(inter)
            await inter.followup.send(
                f"🔢 Shot {self.shot_num} image seed set to "
                f"`{'RANDOM' if new_seed == -1 else new_seed}`.",
                ephemeral=True,
            )

        modal = _SeedEditModal(
            script_id=self.script_id, shot_num=self.shot_num,
            prompt_kind="image",
            current_value=shot_state.get("image_seed", -1),
            on_done=on_done,
        )
        await interaction.response.send_modal(modal)

    @ui.button(label="✏️ Edit Motion", style=discord.ButtonStyle.primary, row=1)
    async def edit_motion(self, interaction, button):
        shot_state = _get_state(self.script_id)["prompts"].get(str(self.shot_num), {})

        async def on_done(inter, new_text):
            def mut(s):
                s["motion_prompt"] = new_text.strip()
                s["motion_seed"] = _random_seed()
            self._update_shot(mut)
            await self._refresh_message(inter)
            await inter.followup.send(
                f"✏️ Shot {self.shot_num} motion prompt updated · new seed assigned.",
                ephemeral=True,
            )

        modal = _PromptEditModal(
            script_id=self.script_id, shot_num=self.shot_num,
            prompt_kind="motion",
            current_value=shot_state.get("motion_prompt", ""),
            on_done=on_done,
        )
        await interaction.response.send_modal(modal)

    @ui.button(label="🎲 Reseed Motion", style=discord.ButtonStyle.secondary, row=1)
    async def reseed_motion(self, interaction, button):
        def mut(s):
            s["motion_seed"] = _random_seed()
        shot_state = self._update_shot(mut)
        await interaction.response.defer(ephemeral=False, thinking=False)
        await self._refresh_message(interaction)
        await interaction.followup.send(
            f"🎲 Shot {self.shot_num} motion seed → `{shot_state['motion_seed']}`.",
            ephemeral=True,
        )

    @ui.button(label="🔢 Set Motion Seed", style=discord.ButtonStyle.secondary, row=1)
    async def set_motion_seed(self, interaction, button):
        shot_state = _get_state(self.script_id)["prompts"].get(str(self.shot_num), {})

        async def on_done(inter, new_seed):
            def mut(s):
                s["motion_seed"] = new_seed
            self._update_shot(mut)
            await self._refresh_message(inter)
            await inter.followup.send(
                f"🔢 Shot {self.shot_num} motion seed set to "
                f"`{'RANDOM' if new_seed == -1 else new_seed}`.",
                ephemeral=True,
            )

        modal = _SeedEditModal(
            script_id=self.script_id, shot_num=self.shot_num,
            prompt_kind="motion",
            current_value=shot_state.get("motion_seed", -1),
            on_done=on_done,
        )
        await interaction.response.send_modal(modal)

    @ui.button(label="✅ Approve Shot", style=discord.ButtonStyle.success, row=2)
    async def approve_shot(self, interaction, button):
        def mut(s):
            s["approved"] = True
        self._update_shot(mut)
        # _update_shot would normally un-approve, but our mutator sets True
        # *after* the un-approve check. Re-set True to be safe and save.
        state = _get_state(self.script_id)
        state["prompts"][str(self.shot_num)]["approved"] = True
        _save(state)

        await interaction.response.defer(ephemeral=False, thinking=False)
        await self._refresh_message(interaction)
        await interaction.followup.send(
            f"✅ Shot {self.shot_num} locked in. Edit again to unlock.",
            ephemeral=True,
        )


# ==============================================================================
# MASTER (Approve All) view
# ==============================================================================

class _MasterApprovalView(ui.View):
    def __init__(
        self, *, script_id: str, owner_id: int,
        on_approve_all: Callable, on_cancel: Callable,
        action_word: str = "Render",
    ):
        super().__init__(timeout=86400)
        self.script_id = script_id
        self.owner_id = owner_id
        self.on_approve_all = on_approve_all
        self.on_cancel = on_cancel
        self.action_word = action_word  # "Render" (fresh) or "Regenerate" (re-edit)
        self.message: Optional[discord.Message] = None
        _MASTER_VIEWS[script_id] = self

    async def interaction_check(self, interaction):
        if interaction.user.id != self.owner_id and not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message(
                "🔒 Only the requester (or an admin) can decide.", ephemeral=True,
            )
            return False
        return True

    def _all_approved(self) -> tuple[bool, int, int]:
        state = _get_state(self.script_id) or {}
        prompts = state.get("prompts", {}) or {}
        if not prompts:
            return False, 0, 0
        approved = sum(1 for v in prompts.values() if v.get("approved"))
        total = len(prompts)
        return approved == total, approved, total

    async def refresh(self):
        all_ok, approved, total = self._all_approved()
        for c in self.children:
            if isinstance(c, ui.Button) and c.label and c.label.startswith("✅ Approve ALL"):
                c.disabled = not all_ok
                c.label = (
                    f"✅ Approve ALL & {self.action_word} ({approved}/{total})"
                    if not all_ok else
                    f"✅ Approve ALL & {self.action_word} ({total}/{total}) — READY"
                )
        if self.message:
            try:
                await self.message.edit(view=self)
            except Exception as e:
                log.warning(f"Could not refresh master view: {e}")

    @ui.button(label="✅ Approve ALL & Render (0/0)", style=discord.ButtonStyle.success, row=0)
    async def approve_all(self, interaction, button):
        all_ok, approved, total = self._all_approved()
        if not all_ok:
            await interaction.response.send_message(
                f"⏳ Only `{approved}/{total}` shots approved. "
                f"Click ✅ Approve Shot on the rest first.",
                ephemeral=True,
            )
            return
        await interaction.response.defer(ephemeral=False, thinking=False)
        for c in self.children:
            c.disabled = True
        if self.message:
            try:
                await self.message.edit(
                    content=interaction.message.content
                    + f"\n\n🚀 Approved by {interaction.user.mention} — kicking off render.",
                    view=self,
                )
            except Exception:
                pass
        _MASTER_VIEWS.pop(self.script_id, None)
        try:
            await self.on_approve_all(interaction)
        except Exception as e:
            log.exception("on_approve_all callback failed")
            await interaction.followup.send(f"❌ Render kickoff failed: `{e}`")

    @ui.button(label="🛑 Cancel", style=discord.ButtonStyle.danger, row=0)
    async def cancel(self, interaction, button):
        await interaction.response.defer(ephemeral=False, thinking=False)
        for c in self.children:
            c.disabled = True
        if self.message:
            try:
                await self.message.edit(
                    content=interaction.message.content
                    + f"\n\n🛑 Cancelled by {interaction.user.mention}.",
                    view=self,
                )
            except Exception:
                pass
        _MASTER_VIEWS.pop(self.script_id, None)
        try:
            await self.on_cancel(interaction)
        except Exception:
            pass


# ==============================================================================
# PUBLIC API — called from claw_bot
# ==============================================================================

async def post_approval_ui(
    channel: discord.TextChannel,
    script: dict,
    owner_id: int,
    *,
    on_approve_all: Callable,
    on_cancel: Optional[Callable] = None,
    progress_msg: Optional[discord.Message] = None,
    reuse_existing: bool = False,
    only_shots: Optional[list[int]] = None,
    action_word: str = "Render",
) -> dict:
    """
    Post one prompt-edit embed+view per shot, finish with a master Approve-All
    view. Returns the persisted state dict.

    Normal (fresh) flow: generate prompts via Qwen first.

    Re-edit flow (reuse_existing=True): load the already-saved prompts from disk
    instead of regenerating — lets the user fix a bad shot's prompt and re-render
    WITHOUT losing earlier edits or paying for a full LLM re-assembly. Pass
    only_shots=[n, ...] to re-post just those shots (they get un-approved so the
    user must re-lock them); action_word="Regenerate" relabels the master button.

    Caller must provide on_approve_all(interaction) which kicks off the
    storyboard (re)render.
    """
    script_id = script.get("_id") or script.get("script_id") or "unknown"
    shots = script.get("shots", []) or []
    total = len(shots)
    if total == 0:
        await channel.send(f"❌ Script `{script_id}` has no shots — nothing to approve.")
        return {}

    bot_loop = asyncio.get_running_loop()
    last_edit = {"t": 0.0}

    def _progress(text: str, cur: int, tot: int):
        import time as _t
        now = _t.time()
        if now - last_edit["t"] < 2.0 and cur < tot:
            return
        last_edit["t"] = now
        if progress_msg is not None:
            asyncio.run_coroutine_threadsafe(
                progress_msg.edit(content=f"🧠 Prompt assembly · {cur}/{tot} · {text}"),
                bot_loop,
            )

    # --- State: reuse saved prompts (re-edit) or generate fresh ---
    state = load_approved_prompts(script_id) if reuse_existing else None
    if state and state.get("prompts"):
        if progress_msg is None:
            progress_msg = await channel.send(
                f"📝 Loading saved prompts for `{script_id}` to edit…"
            )
    else:
        if progress_msg is None:
            progress_msg = await channel.send(
                f"🧠 Generating image + motion prompts for `{script_id}` "
                f"({total} shots × 2 LLM calls = ~{total * 30}s)…"
            )
        state = await asyncio.to_thread(generate_all_prompts, script, _progress)

    # Scope to a subset of shots (re-edit a bad shot). Un-approve the targeted
    # shots so the user must re-lock them; force every OTHER shot approved so
    # they don't block the master button (we only post the targeted embeds).
    if only_shots:
        only_set = {str(n) for n in only_shots}
        for k, v in state.get("prompts", {}).items():
            v["approved"] = k not in only_set
        shots = [s for s in shots if str(s.get("shot_number")) in only_set]

    _save(state)
    _ACTIVE_STATES[script_id] = state

    try:
        await progress_msg.edit(
            content=(
                f"📝 **Prompts ready for `{script_id}`** — review each shot below.\n"
                f"Edit prompts, change seeds, or accept defaults. "
                f"Click **✅ Approve Shot** on each, then **✅ Approve ALL & {action_word}** at the bottom."
            )
        )
    except Exception:
        pass

    # Post one embed + view per shot
    for shot in shots:
        shot_num = shot.get("shot_number")
        shot_state = state["prompts"].get(str(shot_num))
        if not shot_state:
            continue
        msg_holder: dict = {"msg": None}
        view = _ShotApprovalView(
            script_id=script_id,
            shot_num=shot_num,
            owner_id=owner_id,
            message_ref_holder=msg_holder,
        )
        embed = _shot_embed(script_id, shot_num, shot_state)
        try:
            msg = await channel.send(embed=embed, view=view)
            msg_holder["msg"] = msg
        except discord.HTTPException as e:
            log.warning(f"Could not post shot {shot_num} approval embed: {e}")
            await channel.send(
                f"⚠️ Could not post shot {shot_num} embed (`{e}`). "
                f"Edit prompts later via the saved JSON: `{approved_prompts_path(script_id)}`"
            )

    # Master view at the bottom
    async def _default_cancel(_inter):
        await channel.send(f"🛑 Prompt approval for `{script_id}` cancelled.")

    master = _MasterApprovalView(
        script_id=script_id, owner_id=owner_id,
        on_approve_all=on_approve_all,
        on_cancel=on_cancel or _default_cancel,
        action_word=action_word,
    )
    master_msg = await channel.send(
        f"**Master approval for `{script_id}`** — click below once every shot is locked.",
        view=master,
    )
    master.message = master_msg
    await master.refresh()

    return state


def is_fully_approved(script_id: str) -> bool:
    state = load_approved_prompts(script_id)
    if not state:
        return False
    prompts = state.get("prompts", {}) or {}
    if not prompts:
        return False
    return all(v.get("approved") for v in prompts.values())
