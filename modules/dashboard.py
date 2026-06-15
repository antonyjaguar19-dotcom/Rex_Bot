"""
Claw Bot — Gradio Dashboard

Single-page browser UI that replaces the multi-channel Discord workflow.
Pipeline tab walks the user through: theme → script → prompts → storyboard
→ video → final, with approval gates at each stage. Discord bot keeps
running in parallel (for mobile checks / status posts).

Launched automatically from claw_bot.py on_ready in its own thread.

CSS = animated gradient header + soft pulse on action buttons. Lightweight.
"""

import json
import logging
import random
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import gradio as gr

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import script_generator as sg
from modules import prompt_approval as pap
from modules import storyboard_generator as sb_gen
from modules import clip_generator as cg
from modules import runtime_settings as rs
from modules import voice_casting as _vc
from modules import model_registry
from modules import gpu_utils
from modules import beat_policy as bp
from modules import assembly as asm
from modules.theme_bank import get_random_theme, get_theme_count

log = logging.getLogger("claw_bot.dashboard")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SCRIPTS_DIR = PROJECT_ROOT / "04_Outputs" / "scripts"
STORYBOARDS_DIR = PROJECT_ROOT / "04_Outputs" / "storyboards"
CLIPS_DIR = PROJECT_ROOT / "04_Outputs" / "clips"
APPROVED_DIR = PROJECT_ROOT / "04_Outputs" / "approved_scripts"


# ==============================================================================
# SHARED STATE — synced with disk so Discord bot + Dashboard agree
# ==============================================================================

class JobState:
    """Lightweight in-memory state. Disk is still source of truth."""

    def __init__(self):
        self.current_script_id: Optional[str] = None
        self.current_script: Optional[dict] = None
        self.log_lines: list[str] = []
        self.busy: bool = False
        self.stage: str = "idle"  # script | prompts | storyboard | video | final

    def log(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {text}"
        self.log_lines.append(line)
        self.log_lines = self.log_lines[-200:]
        log.info(text)

    def log_text(self) -> str:
        return "\n".join(self.log_lines[-30:]) or "_(idle)_"


STATE = JobState()


# ==============================================================================
# ANIMATED CSS — fun gradient, button pulse, card hover
# ==============================================================================

CUSTOM_CSS = """
:root {
    --rex-pink: #ff6f91;
    --rex-purple: #6c63ff;
    --rex-mint: #43e8d8;
    --rex-night: #1c1b2e;
}
.gradio-container {
    background: linear-gradient(120deg, #1c1b2e 0%, #2a1f4a 100%) !important;
    color: #f1edff !important;
}
#rex-header {
    background: linear-gradient(270deg, #ff6f91, #6c63ff, #43e8d8, #ff6f91);
    background-size: 600% 600%;
    animation: rexshift 14s ease infinite;
    border-radius: 18px;
    padding: 24px 28px;
    color: white;
    font-size: 24px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 12px;
    box-shadow: 0 8px 28px rgba(108,99,255,0.35);
}
@keyframes rexshift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.gr-button {
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.gr-button:hover {
    transform: translateY(-2px) scale(1.03) !important;
    box-shadow: 0 6px 18px rgba(255,111,145,0.45) !important;
}
.gr-button-primary {
    background: linear-gradient(120deg, #ff6f91, #6c63ff) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}
.gr-button-stop {
    background: linear-gradient(120deg, #ff5252, #ff1744) !important;
}
.rex-card {
    background: rgba(255,255,255,0.04);
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 10px;
    border: 1px solid rgba(255,255,255,0.06);
}
.rex-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    background: linear-gradient(120deg, #6c63ff, #43e8d8);
    color: white;
    font-size: 12px;
    font-weight: 600;
}
.gr-tabitem {
    background: transparent !important;
}
"""


# ==============================================================================
# HELPERS
# ==============================================================================

def _list_scripts() -> list[tuple[str, str]]:
    """Return (script_id, title) for every script on disk, newest first."""
    if not SCRIPTS_DIR.exists():
        return []
    files = sorted(SCRIPTS_DIR.glob("script_*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for f in files[:50]:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            sid = d.get("_id", f.stem.replace("script_", ""))
            title = d.get("title", "Untitled")
            out.append((sid, title))
        except Exception:
            continue
    return out


def _script_summary(script: dict) -> str:
    if not script:
        return "_(no script loaded)_"
    lines = [
        f"### 🎬 {script.get('title', 'Untitled')}",
        f"**ID:** `{script.get('_id', '?')}` · **Theme:** _{script.get('theme', '?')}_ "
        f"· **Style:** `{script.get('style', '?')}` · **Culture:** `{script.get('culture', '?')}`",
        "",
        f"**Setting:** {script.get('setting', '')}",
        f"**Moral:** {script.get('moral', '')}",
        "",
        "**Characters:**",
    ]
    for c in script.get("characters", []) or []:
        if isinstance(c, dict):
            lines.append(f"- **{c.get('name', '?')}** — {c.get('locked_visual_token', c.get('appearance', ''))}")
    lines.append("")
    lines.append("**Shots:**")
    for s in script.get("shots", []) or []:
        beat = (s.get("beat") or "").lower()
        lines.append(f"- _Shot {s.get('shot_number')}_ ({beat}) — {s.get('narration', '')[:120]}")
    return "\n".join(lines)


def _approved_prompts_table(script_id: Optional[str]) -> list[list]:
    if not script_id:
        return []
    state = pap.load_approved_prompts(script_id) or {}
    prompts = (state.get("prompts") or {})
    rows = []
    for k in sorted(prompts.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        p = prompts[k]
        rows.append([
            int(k),
            p.get("beat", "?"),
            "✅" if p.get("approved") else "✏️",
            p.get("image_seed", -1),
            p.get("motion_seed", -1),
            (p.get("image_prompt", "") or "")[:80] + "...",
            (p.get("motion_prompt", "") or "")[:80] + "...",
        ])
    return rows


def _storyboard_gallery(script_id: Optional[str]) -> list[str]:
    if not script_id:
        return []
    sb_dir = STORYBOARDS_DIR / script_id
    if not sb_dir.exists():
        return []
    files = sorted(sb_dir.glob("shot*_first.png"),
                   key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))
    return [str(f) for f in files]


def _video_gallery(script_id: Optional[str]) -> list[str]:
    if not script_id or not CLIPS_DIR.exists():
        return []
    files = sorted(
        CLIPS_DIR.glob(f"clip_{script_id}_shot*.mp4"),
        key=lambda p: int(''.join(c for c in p.stem.split("shot")[-1] if c.isdigit()) or 0),
    )
    # Prefer v2 (revised) over original for each shot
    by_shot: dict[int, Path] = {}
    for f in files:
        try:
            shot = int(''.join(c for c in f.stem.split("shot")[-1] if c.isdigit()))
        except Exception:
            continue
        if shot not in by_shot or "_v2" in f.name:
            by_shot[shot] = f
    return [str(by_shot[k]) for k in sorted(by_shot.keys())]


def _bg_run(fn, *args, **kwargs):
    """Run a generator function in a daemon thread so Gradio stays responsive."""
    t = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
    t.start()
    return t


def _gpu_line() -> str:
    try:
        s = gpu_utils.get_vram_stats() or {}
        used = s.get("used_mb", 0) / 1024
        free = s.get("free_mb", 0) / 1024
        total = s.get("total_mb", 0) / 1024
        return f"VRAM: **{used:.1f} GB used** · {free:.1f} GB free · {total:.1f} GB total"
    except Exception:
        return "VRAM: (unavailable)"


# ==============================================================================
# PIPELINE STAGE HANDLERS
# ==============================================================================

def do_generate_script(theme: str, culture: str, style: str):
    if STATE.busy:
        return "⏳ Another job already running.", _script_summary(STATE.current_script), gr.update()
    if not theme.strip():
        return "❌ Pick a theme first.", _script_summary(STATE.current_script), gr.update()

    STATE.busy = True
    STATE.stage = "script"
    STATE.log(f"Generating script — theme='{theme}', culture='{culture}', style='{style}'")

    try:
        culture_arg = culture if culture and culture != "(auto)" else None
        style_arg = style if style and style != "(auto)" else None
        script = sg.generate_script(theme, style_override=style_arg, culture_override=culture_arg)
        STATE.current_script = script
        STATE.current_script_id = script.get("_id") or script.get("script_id")
        STATE.log(f"Script {STATE.current_script_id} ready — '{script.get('title')}'")
        choices = [f"{sid} — {t}" for sid, t in _list_scripts()]
        return (
            f"✅ Script ready: **{script.get('title')}** (`{STATE.current_script_id}`)",
            _script_summary(script),
            gr.update(choices=choices, value=f"{STATE.current_script_id} — {script.get('title')}"),
        )
    except Exception as e:
        STATE.log(f"Script gen failed: {e}")
        return f"❌ Script gen failed: `{e}`", _script_summary(STATE.current_script), gr.update()
    finally:
        STATE.busy = False


def do_revise_script(feedback: str):
    if not STATE.current_script:
        return "❌ No script loaded.", _script_summary(None)
    if not feedback.strip():
        return "❌ Empty feedback.", _script_summary(STATE.current_script)
    STATE.busy = True
    STATE.stage = "script"
    STATE.log(f"Revising script {STATE.current_script_id}")
    try:
        new_script = sg.revise_script(STATE.current_script, feedback)
        STATE.current_script = new_script
        STATE.current_script_id = new_script.get("_id") or new_script.get("script_id")
        STATE.log(f"Revision {STATE.current_script_id} ready")
        return (
            f"✅ Revised: **{new_script.get('title')}** (`{STATE.current_script_id}`)",
            _script_summary(new_script),
        )
    except Exception as e:
        STATE.log(f"Revision failed: {e}")
        return f"❌ Revision failed: `{e}`", _script_summary(STATE.current_script)
    finally:
        STATE.busy = False


def do_load_script(picker_value: str):
    if not picker_value:
        return _script_summary(None), [], [], []
    script_id = picker_value.split(" — ")[0].strip()
    path = SCRIPTS_DIR / f"script_{script_id}.json"
    if not path.exists():
        STATE.log(f"Script {script_id} not on disk")
        return _script_summary(None), [], [], []
    try:
        STATE.current_script = json.loads(path.read_text(encoding="utf-8"))
        STATE.current_script_id = script_id
        STATE.log(f"Loaded script {script_id}")
        return (
            _script_summary(STATE.current_script),
            _approved_prompts_table(script_id),
            _storyboard_gallery(script_id),
            _video_gallery(script_id),
        )
    except Exception as e:
        STATE.log(f"Load failed: {e}")
        return _script_summary(None), [], [], []


def do_approve_script_and_gen_prompts():
    if not STATE.current_script:
        return "❌ No script loaded.", []
    if STATE.busy:
        return "⏳ Already busy.", _approved_prompts_table(STATE.current_script_id)

    STATE.busy = True
    STATE.stage = "prompts"
    sid = STATE.current_script_id
    STATE.log(f"Generating prompts for {sid} (image + motion per shot)…")
    APPROVED_DIR.mkdir(parents=True, exist_ok=True)
    (APPROVED_DIR / f"{sid}.approved").write_text(
        f"approved_by=dashboard\napproved_at={datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )

    try:
        state = pap.generate_all_prompts(STATE.current_script)
        pap._save(state)
        pap._ACTIVE_STATES[sid] = state
        STATE.log(f"Prompts ready for {sid} — {len(state.get('prompts', {}))} shots")
        return (
            f"✅ Prompts ready for **{sid}**. Edit/reseed/approve below, then Approve All.",
            _approved_prompts_table(sid),
        )
    except Exception as e:
        STATE.log(f"Prompt gen failed: {e}")
        return f"❌ Prompt gen failed: `{e}`", []
    finally:
        STATE.busy = False


def do_edit_prompt_row(table_data, shot_num, kind, new_text):
    """Edit one shot's image_prompt or motion_prompt + reseed."""
    sid = STATE.current_script_id
    if not sid:
        return "❌ No script.", _approved_prompts_table(None)
    if not new_text.strip():
        return "❌ Empty prompt.", _approved_prompts_table(sid)
    try:
        state = pap.load_approved_prompts(sid) or {"script_id": sid, "prompts": {}}
        entry = state.setdefault("prompts", {}).setdefault(str(int(shot_num)), {})
        if kind == "image":
            entry["image_prompt"] = new_text.strip()
            entry["image_seed"] = random.randint(1, 2_147_483_647)
        else:
            entry["motion_prompt"] = new_text.strip()
            entry["motion_seed"] = random.randint(1, 2_147_483_647)
        entry["approved"] = False
        pap._save(state)
        pap._ACTIVE_STATES[sid] = state
        STATE.log(f"Shot {shot_num} {kind} prompt updated, new seed assigned")
        return f"✏️ Shot {shot_num} {kind} updated.", _approved_prompts_table(sid)
    except Exception as e:
        return f"❌ Edit failed: `{e}`", _approved_prompts_table(sid)


def do_reseed_prompt(shot_num, kind):
    sid = STATE.current_script_id
    if not sid:
        return "❌ No script.", _approved_prompts_table(None)
    try:
        state = pap.load_approved_prompts(sid) or {"script_id": sid, "prompts": {}}
        entry = state.setdefault("prompts", {}).setdefault(str(int(shot_num)), {})
        new_seed = random.randint(1, 2_147_483_647)
        if kind == "image":
            entry["image_seed"] = new_seed
        else:
            entry["motion_seed"] = new_seed
        pap._save(state)
        pap._ACTIVE_STATES[sid] = state
        STATE.log(f"Shot {shot_num} {kind} reseeded → {new_seed}")
        return f"🎲 Shot {shot_num} {kind} seed → {new_seed}", _approved_prompts_table(sid)
    except Exception as e:
        return f"❌ Reseed failed: `{e}`", _approved_prompts_table(sid)


def do_approve_shot_prompt(shot_num):
    sid = STATE.current_script_id
    if not sid:
        return "❌ No script.", _approved_prompts_table(None)
    try:
        state = pap.load_approved_prompts(sid) or {"script_id": sid, "prompts": {}}
        entry = state.setdefault("prompts", {}).setdefault(str(int(shot_num)), {})
        entry["approved"] = True
        pap._save(state)
        pap._ACTIVE_STATES[sid] = state
        return f"✅ Shot {shot_num} locked.", _approved_prompts_table(sid)
    except Exception as e:
        return f"❌ Approve failed: `{e}`", _approved_prompts_table(sid)


def do_approve_all_and_run_storyboard():
    sid = STATE.current_script_id
    if not sid:
        return "❌ No script.", [], "_(idle)_"
    state = pap.load_approved_prompts(sid)
    if not state or not state.get("prompts"):
        return "❌ Prompts not generated yet. Click 'Generate Prompts' first.", [], STATE.log_text()

    # Auto-approve any unflipped shots (user can disable by approving manually)
    for k, p in state["prompts"].items():
        p["approved"] = True
    pap._save(state)
    pap._ACTIVE_STATES[sid] = state

    if STATE.busy:
        return "⏳ Already running.", [], STATE.log_text()

    STATE.busy = True
    STATE.stage = "storyboard"
    STATE.log(f"Storyboard render started for {sid}…")

    def _worker():
        try:
            aspect = rs.get_effective_aspect_ratio() or "16:9"
            result = sb_gen.generate_storyboard(
                script_id=sid, aspect_ratio=aspect,
                progress_callback=lambda txt, cur, tot: STATE.log(f"  {cur}/{tot} — {txt}"),
            )
            if result.success:
                STATE.log(f"Storyboard done — {result.total_frames} frames")
            else:
                STATE.log(f"Storyboard FAILED: {result.error}")
        except Exception as e:
            STATE.log(f"Storyboard worker crashed: {e}")
        finally:
            STATE.busy = False
            STATE.stage = "idle"

    _bg_run(_worker)
    return f"🚀 Storyboard render started for `{sid}`. Watch the gallery refresh.", [], STATE.log_text()


def do_run_video():
    sid = STATE.current_script_id
    if not sid:
        return "❌ No script.", [], STATE.log_text()
    sb_manifest = STORYBOARDS_DIR / sid / "storyboard.json"
    if not sb_manifest.exists():
        return "❌ Storyboard not generated yet.", [], STATE.log_text()
    if STATE.busy:
        return "⏳ Busy.", [], STATE.log_text()

    STATE.busy = True
    STATE.stage = "video"
    STATE.log(f"Video render started for {sid}…")

    def _worker():
        try:
            script = STATE.current_script or json.loads(
                (SCRIPTS_DIR / f"script_{sid}.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(sb_manifest.read_text(encoding="utf-8"))
            image_by_shot = {
                f.get("shot_number"): PROJECT_ROOT / f.get("image_path")
                for f in manifest.get("frames", [])
                if f.get("shot_number") and f.get("image_path")
            }
            clip_gen = cg.ClipGenerator(sync_mode=rs.get_effective_sync_mode())
            clip_gen.tts.voice = rs.get_effective_voice()
            for shot in script.get("shots", []):
                num = shot.get("shot_number")
                if num not in image_by_shot:
                    continue
                approved = pap.get_shot_prompts(sid, num) or {}
                motion = (approved.get("motion_prompt", "").strip()
                          or shot.get("motion_prompt", "").strip()
                          or shot.get("visual_description", "").strip())
                motion += " No speech. No mouth movement."
                seed = approved.get("motion_seed", -1)
                seed_arg = seed if isinstance(seed, int) and seed > 0 else None
                beat = (shot.get("beat") or "").lower()
                STATE.log(f"  Clip shot {num} starting (beat={beat}, seed={seed_arg})")
                cg_out = clip_gen.generate_clip(
                    shot_id=f"{sid}_shot{num}",
                    narration=shot.get("narration", "").strip(),
                    action_prompt=motion,
                    storyboard_image=image_by_shot[num],
                    output_filename=f"clip_{sid}_shot{num}.mp4",
                    seed=seed_arg, beat=beat,
                    voice=_vc.resolve_voice(script, shot.get("speaker")),
                )
                STATE.log(f"  Clip shot {num} done → {cg_out.name}")
            STATE.log("Video render complete")
        except Exception as e:
            STATE.log(f"Video worker crashed: {e}")
        finally:
            STATE.busy = False
            STATE.stage = "idle"

    _bg_run(_worker)
    return f"🚀 Video render started for `{sid}`.", [], STATE.log_text()


def do_assemble_final():
    sid = STATE.current_script_id
    if not sid:
        return "❌ No script.", None, None
    if STATE.busy:
        return "⏳ Busy.", None, None
    STATE.busy = True
    STATE.stage = "final"
    STATE.log(f"Final assembly running for {sid}…")
    try:
        result = asm.assemble_final(sid, None)
        STATE.log(f"Final ready — {result['shot_count']} shots, {result['total_duration_sec']:.1f}s")
        f916 = result.get("9x16")
        f169 = result.get("16x9")
        return (
            f"✅ Final ready: {result['shot_count']} shots · "
            f"{result['total_duration_sec']:.1f}s",
            str(f916) if f916 and f916.exists() else None,
            str(f169) if f169 and f169.exists() else None,
        )
    except Exception as e:
        STATE.log(f"Assembly failed: {e}")
        return f"❌ Assembly failed: `{e}`", None, None
    finally:
        STATE.busy = False
        STATE.stage = "idle"


# ==============================================================================
# AUTO-REFRESH TICKER (every 2s while pipeline running)
# ==============================================================================

def tick_refresh():
    sid = STATE.current_script_id
    return (
        STATE.log_text(),
        _approved_prompts_table(sid),
        _storyboard_gallery(sid),
        _video_gallery(sid),
        _gpu_line(),
        f"**Stage:** `{STATE.stage}` · **Busy:** {'🔴 yes' if STATE.busy else '🟢 no'}",
    )


# ==============================================================================
# UI ASSEMBLY
# ==============================================================================

def build_ui() -> gr.Blocks:
    cultures = ["(auto)", "indian", "western", "japanese", "mixed", "animal-kingdom", "fantasy"]
    styles = ["(auto)"] + sg.get_available_style_ids()

    with gr.Blocks(title="Rex VFX Bot · Dashboard") as app:

        gr.HTML('<div id="rex-header">🎬 Rex VFX Bot — Claw Dashboard ✨</div>')
        with gr.Row():
            status_bar = gr.Markdown(value="**Stage:** `idle` · **Busy:** 🟢 no")
            gpu_bar = gr.Markdown(value=_gpu_line())

        with gr.Tabs():

            # ---------------- PIPELINE TAB ----------------
            with gr.Tab("🚀 Pipeline"):

                # Step 1 — Script
                with gr.Group(elem_classes=["rex-card"]):
                    gr.Markdown("### 1️⃣ Script")
                    with gr.Row():
                        theme = gr.Textbox(label="Theme", placeholder="e.g. a tortoise teaches a hare patience",
                                           scale=4)
                        culture = gr.Dropdown(cultures, value="(auto)", label="Culture", scale=1)
                        style = gr.Dropdown(styles, value="(auto)", label="Style", scale=1)
                    with gr.Row():
                        gen_script_btn = gr.Button("✨ Generate Script", variant="primary")
                        random_theme_btn = gr.Button("🎲 Random Theme")
                    script_picker = gr.Dropdown(
                        choices=[f"{sid} — {t}" for sid, t in _list_scripts()],
                        label="Or load an existing script", interactive=True,
                    )
                    script_md = gr.Markdown(value=_script_summary(None))
                    script_status = gr.Markdown(value="")

                    with gr.Row():
                        revise_feedback = gr.Textbox(label="Revision feedback",
                                                     placeholder="e.g. make the ending happier")
                        revise_btn = gr.Button("✏️ Revise")
                    with gr.Row():
                        approve_script_btn = gr.Button(
                            "✅ Approve Script → Generate Prompts", variant="primary",
                        )

                # Step 2 — Prompts
                with gr.Group(elem_classes=["rex-card"]):
                    gr.Markdown("### 2️⃣ Prompts (image + motion per shot)")
                    prompts_status = gr.Markdown(value="")
                    prompts_table = gr.Dataframe(
                        headers=["Shot", "Beat", "✓", "Img Seed", "Mot Seed",
                                 "Image Prompt (preview)", "Motion Prompt (preview)"],
                        datatype=["number", "str", "str", "number", "number", "str", "str"],
                        value=[],
                        wrap=True,
                        interactive=False,
                        label="Approved prompts",
                    )
                    with gr.Row():
                        edit_shot_num = gr.Number(label="Shot #", value=1, precision=0, scale=1)
                        edit_kind = gr.Radio(["image", "motion"], value="image",
                                             label="Which prompt", scale=1)
                        edit_text = gr.Textbox(label="New prompt", lines=3, scale=4)
                    with gr.Row():
                        edit_btn = gr.Button("✏️ Save Edit (auto-reseed)")
                        reseed_btn = gr.Button("🎲 Reseed Only")
                        approve_shot_btn = gr.Button("✅ Approve This Shot")
                    with gr.Row():
                        approve_all_btn = gr.Button(
                            "🚀 Approve ALL & Render Storyboard", variant="primary",
                        )

                # Step 3 — Storyboard
                with gr.Group(elem_classes=["rex-card"]):
                    gr.Markdown("### 3️⃣ Storyboard")
                    storyboard_gallery = gr.Gallery(
                        value=[], label="Storyboard frames",
                        columns=4, rows=2, height=500, allow_preview=True,
                    )
                    with gr.Row():
                        approve_storyboard_btn = gr.Button(
                            "✅ Approve Storyboard → Render Video", variant="primary",
                        )

                # Step 4 — Video
                with gr.Group(elem_classes=["rex-card"]):
                    gr.Markdown("### 4️⃣ Video clips")
                    video_gallery = gr.Gallery(
                        value=[], label="Generated clips",
                        columns=2, rows=2, height=500,
                    )
                    approve_video_btn = gr.Button(
                        "✅ Approve Video → Final Assembly", variant="primary",
                    )

                # Step 5 — Final
                with gr.Group(elem_classes=["rex-card"]):
                    gr.Markdown("### 5️⃣ Final assembly")
                    final_status = gr.Markdown(value="_(not assembled yet)_")
                    with gr.Row():
                        final_916 = gr.Video(label="📱 9x16 (Shorts)", height=400)
                        final_169 = gr.Video(label="🖥️ 16x9", height=400)

            # ---------------- LOG TAB ----------------
            with gr.Tab("📜 Log"):
                log_box = gr.Code(value=STATE.log_text(), label="Recent events",
                                  language="markdown", interactive=False, lines=20)

            # ---------------- SETTINGS TAB ----------------
            with gr.Tab("⚙️ Settings"):
                gr.Markdown("### Runtime knobs (saved to runtime_settings.json)")
                with gr.Row():
                    cur_style = gr.Textbox(label="Active style", value=rs.get_effective_style() or "(default)",
                                           interactive=False)
                    cur_voice = gr.Textbox(label="Active voice", value=rs.get_effective_voice() or "(default)",
                                           interactive=False)
                    cur_aspect = gr.Textbox(label="Aspect", value=rs.get_effective_aspect_ratio() or "16:9",
                                            interactive=False)
                    cur_sync = gr.Textbox(label="Sync", value=rs.get_effective_sync_mode() or "strict",
                                          interactive=False)
                gr.Markdown(
                    "_Edit via Discord commands (`!set_style`, `!set_voice`, `!set_resolution`, …) "
                    "or directly in `05_Config/runtime_settings.json`. Reload page to refresh._"
                )

            # ---------------- STATS TAB ----------------
            with gr.Tab("📊 Stats"):
                stats_md = gr.Markdown(value="_(stats load when bot stats.json present)_")
                refresh_stats_btn = gr.Button("🔄 Refresh stats")

                def _read_stats():
                    p = PROJECT_ROOT / "05_Config" / "stats.json"
                    if not p.exists():
                        return "_(no stats yet)_"
                    try:
                        d = json.loads(p.read_text(encoding="utf-8"))
                        lines = ["### 📈 Counters"]
                        for k, v in d.items():
                            lines.append(f"- **{k}**: `{v}`")
                        lines.append(f"\n_Theme bank size: {get_theme_count()}_")
                        return "\n".join(lines)
                    except Exception as e:
                        return f"_(stats unreadable: {e})_"

                refresh_stats_btn.click(_read_stats, outputs=[stats_md])

        # ============================ WIRING ============================

        gen_script_btn.click(
            do_generate_script,
            inputs=[theme, culture, style],
            outputs=[script_status, script_md, script_picker],
        )

        random_theme_btn.click(
            lambda: get_random_theme(),
            outputs=[theme],
        )

        script_picker.change(
            do_load_script,
            inputs=[script_picker],
            outputs=[script_md, prompts_table, storyboard_gallery, video_gallery],
        )

        revise_btn.click(
            do_revise_script,
            inputs=[revise_feedback],
            outputs=[script_status, script_md],
        )

        approve_script_btn.click(
            do_approve_script_and_gen_prompts,
            outputs=[prompts_status, prompts_table],
        )

        edit_btn.click(
            do_edit_prompt_row,
            inputs=[prompts_table, edit_shot_num, edit_kind, edit_text],
            outputs=[prompts_status, prompts_table],
        )

        reseed_btn.click(
            do_reseed_prompt,
            inputs=[edit_shot_num, edit_kind],
            outputs=[prompts_status, prompts_table],
        )

        approve_shot_btn.click(
            do_approve_shot_prompt,
            inputs=[edit_shot_num],
            outputs=[prompts_status, prompts_table],
        )

        approve_all_btn.click(
            do_approve_all_and_run_storyboard,
            outputs=[prompts_status, storyboard_gallery, log_box],
        )

        approve_storyboard_btn.click(
            do_run_video,
            outputs=[prompts_status, video_gallery, log_box],
        )

        approve_video_btn.click(
            do_assemble_final,
            outputs=[final_status, final_916, final_169],
        )

        # Auto-refresh every 2 seconds
        timer = gr.Timer(2.0)
        timer.tick(
            tick_refresh,
            outputs=[log_box, prompts_table, storyboard_gallery, video_gallery, gpu_bar, status_bar],
        )

    return app


# ==============================================================================
# LAUNCHER — called from claw_bot.py on bot ready
# ==============================================================================

_LAUNCHED = False


def launch_dashboard(
    port: int = 7860,
    host: str = "127.0.0.1",
    open_browser: bool = True,
) -> threading.Thread:
    """Launch Gradio in a daemon thread so it doesn't block the asyncio loop.

    Returns the thread. Safe to call once at bot on_ready. Idempotent —
    repeat calls are no-ops.
    """
    global _LAUNCHED
    if _LAUNCHED:
        log.info("Dashboard already launched, skipping.")
        return None

    def _runner():
        try:
            app = build_ui()
            log.info(f"Launching dashboard on http://{host}:{port}")
            app.launch(
                server_name=host,
                server_port=port,
                inbrowser=open_browser,
                prevent_thread_lock=False,
                quiet=True,
                show_error=True,
                share=False,
                theme=gr.themes.Soft(primary_hue="pink", secondary_hue="purple"),
                css=CUSTOM_CSS,
            )
        except Exception as e:
            log.exception(f"Dashboard launch crashed: {e}")

    t = threading.Thread(target=_runner, daemon=True, name="ClawBotDashboard")
    t.start()
    _LAUNCHED = True
    return t


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    launch_dashboard(open_browser=True)
    # Standalone mode: keep main thread alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Dashboard stopped.")
