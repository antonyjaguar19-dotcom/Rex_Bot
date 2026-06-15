"""
Claw Bot — NiceGUI Dashboard

Replaces the Gradio dashboard with a Material-Design-y wizard built on
NiceGUI (FastAPI + Vue + Quasar under the hood). Smoother animations,
nicer cards, real toast notifications, async-native.

Launched in its own thread from claw_bot.py on bot ready.
"""

import asyncio
import json
import logging
import os
import random
import secrets as pysecrets
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from nicegui import ui, app
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import script_generator as sg
from modules import prompt_approval as pap
from modules import storyboard_generator as sb_gen
from modules import clip_generator as cg
from modules import runtime_settings as rs
from modules import voice_casting as vc
from modules import gpu_utils
from modules import assembly as asm
from modules import model_registry
from modules import pending_feedback as pf
from modules import upscaler
from modules import sync_bridge as sbr
from modules.theme_bank import get_random_theme, get_theme_count
from modules.file_utils import atomic_write_json
from modules import job_lock
from modules import config_check

# Common Kokoro voices (no list API in tts_engine; mirror control panel hints)
VOICE_CHOICES = [
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
]
try:
    from modules.music_generator import VALID_MOODS as MUSIC_MOODS
    MUSIC_MOODS = list(MUSIC_MOODS)
except Exception:
    MUSIC_MOODS = ["calm", "cheerful", "adventurous", "mysterious"]

log = logging.getLogger("claw_bot.dashboard_nicegui")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SCRIPTS_DIR = PROJECT_ROOT / "04_Outputs" / "scripts"
STORYBOARDS_DIR = PROJECT_ROOT / "04_Outputs" / "storyboards"
CLIPS_DIR = PROJECT_ROOT / "04_Outputs" / "clips"
APPROVED_DIR = PROJECT_ROOT / "04_Outputs" / "approved_scripts"


# ==============================================================================
# AUTH — password gate (REQUIRED when exposing via a public Cloudflare tunnel)
# ==============================================================================
# The dashboard controls the GPU pipeline, so a public URL MUST be protected.
# Set DASHBOARD_PASSWORD in 05_Config/secrets.env. When unset the gate is OFF
# (localhost-only dev); a loud warning is logged so you don't expose it open.
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=PROJECT_ROOT / "05_Config" / "secrets.env")
except Exception:
    pass
DASHBOARD_PASSWORD = (os.environ.get("DASHBOARD_PASSWORD") or "").strip()

# Paths that must load without a session (NiceGUI's own JS/CSS + the login page),
# otherwise the login screen itself can't render.
_AUTH_OPEN_PREFIXES = ("/_nicegui", "/login")


class _AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if DASHBOARD_PASSWORD:
            path = request.url.path
            if not path.startswith(_AUTH_OPEN_PREFIXES):
                try:
                    authed = app.storage.user.get("authenticated", False)
                except Exception:
                    authed = False
                if not authed:
                    app.storage.user["referrer_path"] = path
                    return RedirectResponse("/login")
        return await call_next(request)


app.add_middleware(_AuthMiddleware)
if not DASHBOARD_PASSWORD:
    log.warning("DASHBOARD_PASSWORD not set — dashboard has NO login gate. "
                "Do NOT expose it on a public tunnel until you set one.")

# Failed-login throttle state (shared across sessions on purpose — one GPU
# box, one dashboard; a global lockout is the simplest brute-force brake).
_LOGIN_FAILS = {"count": 0, "locked_until": 0.0}


# ==============================================================================
# SHARED STATE
# ==============================================================================

class State:
    def __init__(self):
        self.script_id: Optional[str] = None
        self.script: Optional[dict] = None
        self.stage: str = "idle"  # idle | script | prompts | storyboard | video | final
        self.busy: bool = False
        self.log_lines: list[str] = []

    def push(self, line: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_lines.append(f"[{ts}] {line}")
        self.log_lines = self.log_lines[-300:]
        log.info(line)


S = State()
STAGES = ["script", "prompts", "storyboard", "video", "final"]


# ==============================================================================
# CUSTOM CSS / HEAD
# ==============================================================================

CUSTOM_CSS = """
:root {
    --rex-pink: #ff6f91;
    --rex-purple: #6c63ff;
    --rex-mint: #43e8d8;
    --rex-night: #0e0e1f;
}
body {
    background: radial-gradient(circle at top right, #2a1f4a 0%, #0e0e1f 60%) !important;
    color: #f1edff !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}
.q-page {
    background: transparent !important;
}
.rex-hero {
    background: linear-gradient(110deg, #ff6f91 0%, #6c63ff 55%, #43e8d8 100%);
    border-radius: 16px;
    padding: 18px 26px;
    color: white;
    box-shadow: 0 6px 22px rgba(108,99,255,0.28);
}
.rex-card {
    background: rgba(255,255,255,0.035) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    box-shadow: 0 3px 14px rgba(0,0,0,0.22);
}
/* Left-nav drawer + vertical tabs */
.rex-drawer {
    background: rgba(14,14,31,0.92) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
.rex-nav .q-tab {
    justify-content: flex-start !important;
    min-height: 46px;
    border-radius: 10px;
    margin: 2px 6px;
    color: #c8c3e0 !important;
}
.rex-nav .q-tab--active {
    background: linear-gradient(120deg, rgba(255,111,145,0.22), rgba(108,99,255,0.22)) !important;
    color: #fff !important;
}
.rex-nav .q-tab__indicator { display: none; }
.rex-section-title {
    font-size: 20px; font-weight: 700; margin-bottom: 4px;
}
.rex-step {
    display: flex; align-items: center; justify-content: center;
    width: 44px; height: 44px; border-radius: 50%;
    background: rgba(255,255,255,0.08); color: #f1edff;
    font-weight: 700; font-size: 16px; border: 2px solid rgba(255,255,255,0.12);
    transition: all 0.4s ease;
}
.rex-step.active {
    background: linear-gradient(120deg, #ff6f91, #6c63ff);
    border-color: white;
    transform: scale(1.15);
    box-shadow: 0 0 22px rgba(255,111,145,0.7);
}
.rex-step.done {
    background: linear-gradient(120deg, #43e8d8, #6c63ff);
    border-color: rgba(255,255,255,0.4);
}
.rex-step-line {
    flex: 1; height: 3px; background: rgba(255,255,255,0.08);
    margin: 0 4px; border-radius: 2px;
}
.rex-step-line.done {
    background: linear-gradient(90deg, #43e8d8, #6c63ff);
    box-shadow: 0 0 10px rgba(67,232,216,0.5);
}
.rex-btn-primary {
    background: linear-gradient(120deg, #ff6f91, #6c63ff) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 14px !important;
    padding: 12px 24px !important;
    box-shadow: 0 6px 20px rgba(255,111,145,0.4) !important;
    transition: transform 0.15s ease !important;
}
.rex-btn-primary:hover {
    transform: translateY(-2px) scale(1.03) !important;
}
.rex-pulse {
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
.rex-shot-card {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    padding: 10px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    transition: all 0.2s ease;
}
.rex-shot-card.approved {
    border-color: rgba(67,232,216,0.5) !important;
    box-shadow: 0 0 18px rgba(67,232,216,0.25) !important;
}
.rex-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 14px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.rex-badge-pink { background: linear-gradient(120deg, #ff6f91, #ff3d6f); color: white; }
.rex-badge-purple { background: linear-gradient(120deg, #6c63ff, #4a3dff); color: white; }
.rex-badge-mint { background: linear-gradient(120deg, #43e8d8, #1ec5b5); color: #0e0e1f; }
.rex-log {
    background: rgba(0,0,0,0.35) !important;
    border-radius: 12px !important;
    padding: 14px !important;
    font-family: 'JetBrains Mono', 'Consolas', monospace !important;
    font-size: 12px !important;
    color: #b8b3d4 !important;
    max-height: 280px;
    overflow-y: auto;
}
"""


# ==============================================================================
# HELPERS
# ==============================================================================

def list_scripts() -> list[tuple[str, str]]:
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


def storyboard_images(script_id: Optional[str]) -> list[Path]:
    if not script_id:
        return []
    sb_dir = STORYBOARDS_DIR / script_id
    if not sb_dir.exists():
        return []
    return sorted(sb_dir.glob("shot*_first.png"),
                  key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))


def video_clips(script_id: Optional[str]) -> list[Path]:
    if not script_id or not CLIPS_DIR.exists():
        return []
    by_shot: dict[int, Path] = {}
    for f in CLIPS_DIR.glob(f"clip_{script_id}_shot*.mp4"):
        try:
            shot = int(''.join(c for c in f.stem.split("shot")[-1] if c.isdigit()))
        except Exception:
            continue
        if shot not in by_shot or "_v2" in f.name:
            by_shot[shot] = f
    return [by_shot[k] for k in sorted(by_shot.keys())]


def gpu_summary() -> str:
    try:
        s = gpu_utils.get_vram_stats() or {}
        used = s.get("used_mb", 0) / 1024
        free = s.get("free_mb", 0) / 1024
        return f"🖥️ VRAM {used:.1f} GB used · {free:.1f} GB free"
    except Exception:
        return "🖥️ VRAM (unavailable)"


def _bg(fn, *args, **kwargs):
    """Run blocking work in a daemon thread."""
    threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()


def _try_begin(label: str) -> bool:
    """Claim the dashboard busy flag + the shared GPU job lock.

    The job lock is cross-frontend: it also blocks while a Discord command
    or the daily scheduler is rendering. Returns False (and notifies) when
    anything else holds the GPU.
    """
    if S.busy:
        ui.notify("⏳ Already busy.", type="warning")
        return False
    if not job_lock.acquire(f"dashboard:{label}"):
        ui.notify(f"⏳ GPU busy: {job_lock.holder_label()}", type="warning")
        return False
    disk_ok, free_gb = config_check.check_disk_space()
    if not disk_ok:
        job_lock.release()
        ui.notify(f"⚠️ Low disk: only {free_gb} GB free — clear 04_Outputs first.",
                  type="negative")
        return False
    S.busy = True
    return True


def _end():
    """Counterpart of _try_begin — call from the worker's finally block."""
    S.busy = False
    job_lock.release()


def _stage_index(stage: str) -> int:
    return STAGES.index(stage) if stage in STAGES else -1


# ==============================================================================
# PIPELINE ACTIONS
# ==============================================================================

def generate_script_action(theme: str, culture: str, style: str, refresh_cb):
    if not theme.strip():
        ui.notify("❌ Theme required.", type="negative")
        return
    if not _try_begin("script generation"):
        return
    S.stage = "script"
    S.push(f"Generating script — theme='{theme}'")
    refresh_cb()

    def worker():
        try:
            culture_arg = None if culture == "(auto)" else culture
            style_arg = None if style == "(auto)" else style
            script = sg.generate_script(theme, style_override=style_arg, culture_override=culture_arg)
            S.script = script
            S.script_id = script.get("_id") or script.get("script_id")
            S.push(f"Script {S.script_id} ready — '{script.get('title')}'")
        except Exception as e:
            S.push(f"Script gen failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg(worker)


def revise_script_action(feedback: str, refresh_cb):
    if not S.script:
        ui.notify("❌ No script loaded.", type="negative")
        return
    if not _try_begin("script revision"):
        return
    S.push(f"Revising {S.script_id}")
    refresh_cb()

    def worker():
        try:
            new_script = sg.revise_script(S.script, feedback)
            S.script = new_script
            S.script_id = new_script.get("_id") or new_script.get("script_id")
            S.push(f"Revision {S.script_id} ready")
        except Exception as e:
            S.push(f"Revision failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg(worker)


def approve_script_gen_prompts(refresh_cb):
    if not S.script:
        ui.notify("❌ No script loaded.", type="negative")
        return
    if not _try_begin("prompt generation"):
        return
    S.stage = "prompts"
    S.push(f"Generating prompts for {S.script_id}…")
    refresh_cb()
    APPROVED_DIR.mkdir(parents=True, exist_ok=True)
    (APPROVED_DIR / f"{S.script_id}.approved").write_text(
        f"approved_by=dashboard\napproved_at={datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )

    def worker():
        try:
            state = pap.generate_all_prompts(S.script)
            pap._save(state)
            pap._ACTIVE_STATES[S.script_id] = state
            S.push(f"Prompts ready — {len(state.get('prompts', {}))} shots")
        except Exception as e:
            S.push(f"Prompt gen failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg(worker)


def approve_all_run_storyboard(refresh_cb):
    sid = S.script_id
    if not sid:
        ui.notify("❌ No script.", type="negative")
        return
    state = pap.load_approved_prompts(sid)
    if not state or not state.get("prompts"):
        ui.notify("❌ Generate prompts first.", type="negative")
        return
    for p in state["prompts"].values():
        p["approved"] = True
    pap._save(state)
    pap._ACTIVE_STATES[sid] = state
    if not _try_begin("storyboard render"):
        return
    S.stage = "storyboard"
    S.push(f"Storyboard render started for {sid}…")
    refresh_cb()

    def worker():
        try:
            aspect = rs.get_effective_aspect_ratio() or "16:9"
            result = sb_gen.generate_storyboard(
                script_id=sid, aspect_ratio=aspect,
                progress_callback=lambda txt, cur, tot: S.push(f"  {cur}/{tot} — {txt}"),
            )
            if result.success:
                S.push(f"Storyboard done — {result.total_frames} frames")
            else:
                S.push(f"Storyboard FAILED: {result.error}")
        except Exception as e:
            S.push(f"Storyboard worker crashed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg(worker)


def run_video(refresh_cb):
    sid = S.script_id
    if not sid:
        ui.notify("❌ No script.", type="negative")
        return
    sb_manifest = STORYBOARDS_DIR / sid / "storyboard.json"
    if not sb_manifest.exists():
        ui.notify("❌ Storyboard not generated yet.", type="negative")
        return
    if not _try_begin("video render"):
        return
    S.stage = "video"
    S.push(f"Video render started for {sid}…")
    refresh_cb()

    def worker():
        try:
            script = S.script or json.loads(
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
                S.push(f"  Clip shot {num} (beat={beat}, seed={seed_arg})")
                cg_out = clip_gen.generate_clip(
                    shot_id=f"{sid}_shot{num}",
                    narration=shot.get("narration", "").strip(),
                    action_prompt=motion,
                    storyboard_image=image_by_shot[num],
                    output_filename=f"clip_{sid}_shot{num}.mp4",
                    seed=seed_arg, beat=beat,
                    voice=vc.resolve_voice(script, shot.get("speaker")),
                    lipsync=vc.is_character_speaker(shot.get("speaker")),
                )
                S.push(f"  Clip shot {num} done → {cg_out.name}")
            S.push("Video render complete")
            sbr.emit("video_done", script_id=sid)
        except Exception as e:
            S.push(f"Video worker crashed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg(worker)


def assemble_final(refresh_cb):
    sid = S.script_id
    if not sid:
        ui.notify("❌ No script.", type="negative")
        return
    if not _try_begin("final assembly"):
        return
    S.stage = "final"
    S.push(f"Final assembly for {sid}…")
    refresh_cb()

    def worker():
        try:
            result = asm.assemble_final(sid, None)
            S.push(f"Final ready — {result['shot_count']} shots, "
                   f"{result['total_duration_sec']:.1f}s")
            sbr.emit("final_done", script_id=sid)
        except Exception as e:
            S.push(f"Assembly failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg(worker)


def update_shot_prompt(shot_num: int, kind: str, new_text: str, reseed: bool = True):
    sid = S.script_id
    if not sid:
        return
    state = pap.load_approved_prompts(sid) or {"script_id": sid, "prompts": {}}
    entry = state.setdefault("prompts", {}).setdefault(str(shot_num), {})
    key = f"{kind}_prompt"
    seed_key = f"{kind}_seed"
    if new_text:
        entry[key] = new_text.strip()
    if reseed:
        entry[seed_key] = random.randint(1, 2_147_483_647)
    pap._save(state)
    pap._ACTIVE_STATES[sid] = state
    S.push(f"Shot {shot_num} {kind} updated")


def update_shot_narration(shot_num: int, new_text: str) -> bool:
    """Edit/rewrite the spoken narration for one shot. Writes the script JSON
    on disk (the render's source of truth) so the next video render uses it.
    Reads from disk first so a Discord-side edit isn't clobbered."""
    sid = S.script_id
    if not sid:
        return False
    path = SCRIPTS_DIR / f"script_{sid}.json"
    if not path.exists():
        return False
    try:
        script = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        S.push(f"Narration save failed (read): {e}")
        return False
    found = False
    for sh in script.get("shots", []):
        if sh.get("shot_number") == shot_num:
            sh["narration"] = (new_text or "").strip()
            found = True
            break
    if not found:
        return False
    try:
        atomic_write_json(path, script)
    except Exception as e:
        S.push(f"Narration save failed (write): {e}")
        return False
    S.script = script
    S.push(f"Shot {shot_num} narration updated")
    return True


def ai_rewrite_narration(shot_num: int, instruction: str, refresh_cb):
    """AI-rewrite one shot's narration (Qwen via script_generator), then save."""
    sid = S.script_id
    if not sid:
        ui.notify("❌ No script.", type="negative"); return
    if not _try_begin("narration rewrite"):
        return
    S.push(f"AI rewriting narration — shot {shot_num}…")
    refresh_cb()

    def worker():
        try:
            sp = SCRIPTS_DIR / f"script_{sid}.json"
            script = json.loads(sp.read_text(encoding="utf-8"))
            shot = next((s for s in script.get("shots", [])
                         if s.get("shot_number") == shot_num), None)
            if not shot:
                S.push(f"Shot {shot_num} not found"); return
            new = sg.rewrite_narration(
                shot.get("narration", ""), instruction or "",
                shot=shot, title=script.get("title", ""),
                moral=script.get("moral", ""),
            )
            update_shot_narration(shot_num, new)
            S.push(f"Shot {shot_num} narration rewritten → {new[:70]}")
        except Exception as e:
            S.push(f"AI rewrite failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg(worker)


def regen_storyboard_shot(shot_num: int, refresh_cb):
    """Delete one shot's frame so generate_storyboard re-renders only it."""
    sid = S.script_id
    if not sid:
        ui.notify("❌ No script.", type="negative"); return
    if not _try_begin(f"storyboard regen shot {shot_num}"):
        return
    S.push(f"Regen storyboard shot {shot_num}…")
    refresh_cb()

    def worker():
        try:
            sb_dir = STORYBOARDS_DIR / sid
            for f in sb_dir.glob(f"shot{shot_num}_*.png"):
                f.unlink(missing_ok=True)
            aspect = rs.get_effective_aspect_ratio() or "16:9"
            sb_gen.generate_storyboard(
                script_id=sid, aspect_ratio=aspect,
                progress_callback=lambda t, c, tot: S.push(f"  {c}/{tot} — {t}"),
            )
            S.push(f"Shot {shot_num} storyboard regenerated")
            # Mirror to Discord (bot poller re-posts the updated frame).
            new_frame = STORYBOARDS_DIR / sid / f"shot{shot_num}_first.png"
            if new_frame.exists():
                sbr.emit("storyboard_regen", script_id=sid, shot=shot_num,
                         image_path=str(new_frame))
        except Exception as e:
            S.push(f"Storyboard regen failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg(worker)


def add_shot_action(position: int, brief: str, refresh_cb):
    """Insert a new LLM-written shot at `position` (1-based). Existing shots
    renumber and their frames/clips follow; only the new shot needs rendering."""
    sid = S.script_id
    if not sid:
        ui.notify("❌ No script.", type="negative"); return
    if not _try_begin(f"add shot {position}"):
        return
    S.push(f"Writing new shot {position}…")
    refresh_cb()

    def worker():
        try:
            from modules import shot_editor as se
            result = se.insert_shot(sid, position, brief or "")
            ns = result["new_shot"]
            S.push(
                f"New shot {position} inserted ({ns['beat']}/{ns['shot_type']}): "
                f"{ns['narration'][:80]}"
            )
            S.push("Next: regen this shot's storyboard + video, then re-assemble.")
            try:
                S.script = json.loads(
                    (SCRIPTS_DIR / f"script_{sid}.json").read_text(encoding="utf-8"))
            except Exception:
                pass
        except Exception as e:
            S.push(f"Add shot failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg(worker)


def add_breathing_shot_action(position: int, image_prompt: str,
                              motion_prompt: str, narration: str,
                              shot_type: str, refresh_cb):
    """Insert a manually-prompted breathing shot (quiet atmospheric beat) at
    `position` (1-based). No LLM — the typed prompts are used verbatim."""
    sid = S.script_id
    if not sid:
        ui.notify("❌ No script.", type="negative"); return
    if not (image_prompt or "").strip():
        ui.notify("❌ Image prompt required for a breathing shot.", type="negative")
        return
    if not _try_begin(f"breathing shot {position}"):
        return
    S.push(f"Inserting breathing shot {position}…")
    refresh_cb()

    def worker():
        try:
            from modules import shot_editor as se
            result = se.insert_breathing_shot(
                sid, position,
                image_prompt=image_prompt, motion_prompt=motion_prompt,
                narration=narration, shot_type=shot_type)
            ns = result["new_shot"]
            S.push(
                f"Breathing shot {position} inserted ({ns['shot_type']}): "
                f"{ns['visual_description'][:80]}"
            )
            S.push("Next: regen this shot's storyboard + video, then re-assemble.")
            try:
                S.script = json.loads(
                    (SCRIPTS_DIR / f"script_{sid}.json").read_text(encoding="utf-8"))
            except Exception:
                pass
        except Exception as e:
            S.push(f"Breathing shot failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg(worker)


def regen_video_shot(shot_num: int, refresh_cb):
    """Re-render a single shot's clip (reuses the run_video per-shot logic)."""
    sid = S.script_id
    if not sid:
        ui.notify("❌ No script.", type="negative"); return
    sb_manifest = STORYBOARDS_DIR / sid / "storyboard.json"
    if not sb_manifest.exists():
        ui.notify("❌ Storyboard not generated yet.", type="negative"); return
    if not _try_begin(f"video regen shot {shot_num}"):
        return
    S.push(f"Regen video shot {shot_num}…")
    refresh_cb()

    def worker():
        try:
            script = S.script or json.loads(
                (SCRIPTS_DIR / f"script_{sid}.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(sb_manifest.read_text(encoding="utf-8"))
            image_by_shot = {
                f.get("shot_number"): PROJECT_ROOT / f.get("image_path")
                for f in manifest.get("frames", [])
                if f.get("shot_number") and f.get("image_path")
            }
            shot = next((s for s in script.get("shots", [])
                         if s.get("shot_number") == shot_num), None)
            if not shot or shot_num not in image_by_shot:
                S.push(f"Shot {shot_num} not found in script/manifest"); return
            clip_gen = cg.ClipGenerator(sync_mode=rs.get_effective_sync_mode())
            clip_gen.tts.voice = rs.get_effective_voice()
            approved = pap.get_shot_prompts(sid, shot_num) or {}
            motion = (approved.get("motion_prompt", "").strip()
                      or shot.get("motion_prompt", "").strip()
                      or shot.get("visual_description", "").strip())
            motion += " No speech. No mouth movement."
            seed = approved.get("motion_seed", -1)
            seed_arg = seed if isinstance(seed, int) and seed > 0 else None
            beat = (shot.get("beat") or "").lower()
            out = clip_gen.generate_clip(
                shot_id=f"{sid}_shot{shot_num}",
                narration=shot.get("narration", "").strip(),
                action_prompt=motion,
                storyboard_image=image_by_shot[shot_num],
                output_filename=f"clip_{sid}_shot{shot_num}.mp4",
                seed=seed_arg, beat=beat,
                voice=vc.resolve_voice(script, shot.get("speaker")),
                lipsync=vc.is_character_speaker(shot.get("speaker")),
            )
            S.push(f"Shot {shot_num} clip regenerated → {out.name}")
            # Mirror to Discord (bot poller re-posts the updated clip).
            sbr.emit("video_regen", script_id=sid, shot=shot_num,
                     clip_path=str(out))
        except Exception as e:
            S.push(f"Video regen failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg(worker)


def run_upscale_action(refresh_cb):
    sid = S.script_id
    if not sid:
        ui.notify("❌ No script.", type="negative"); return
    if not _try_begin("upscale"):
        return
    S.push(f"Upscaling clips for {sid}…")
    refresh_cb()

    def worker():
        try:
            result = upscaler.upscale_storyboard_videos(
                sid, progress_cb=lambda m: S.push(f"  {m}"))
            done = result.get("upscaled", "?") if isinstance(result, dict) else "?"
            S.push(f"Upscale complete ({done} shots)")
        except Exception as e:
            S.push(f"Upscale failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg(worker)


# ==============================================================================
# UI BUILDERS
# ==============================================================================

@ui.page("/login")
def login_page():
    ui.add_head_html(f"<style>{CUSTOM_CSS}</style>")
    # Already logged in → bounce to dashboard.
    if not DASHBOARD_PASSWORD or app.storage.user.get("authenticated", False):
        ui.navigate.to("/")
        return

    def _try_login():
        # Brute-force throttle: the login page is reachable from the public
        # tunnel URL, so 5 wrong guesses lock the form for 60 seconds.
        now = time.time()
        if now < _LOGIN_FAILS["locked_until"]:
            wait = int(_LOGIN_FAILS["locked_until"] - now) + 1
            ui.notify(f"🚫 Too many attempts — wait {wait}s.", type="negative")
            return
        if pysecrets.compare_digest(str(pw.value or ""), DASHBOARD_PASSWORD):
            _LOGIN_FAILS["count"] = 0
            app.storage.user.update({"authenticated": True})
            ui.navigate.to(app.storage.user.get("referrer_path", "/") or "/")
        else:
            _LOGIN_FAILS["count"] += 1
            if _LOGIN_FAILS["count"] >= 5:
                _LOGIN_FAILS["count"] = 0
                _LOGIN_FAILS["locked_until"] = now + 60.0
                log.warning("Dashboard login locked for 60s after 5 failures")
                ui.notify("🚫 Too many wrong attempts — locked for 60s.",
                          type="negative")
            else:
                ui.notify("❌ Wrong password", type="negative")

    with ui.column().classes("w-full items-center").style("margin-top: 14vh;"):
        with ui.card().classes("rex-card").style("max-width: 360px; padding: 26px;"):
            ui.label("🔒 Rex VFX Bot").classes("text-xl font-bold")
            ui.label("Dashboard login").classes("text-xs opacity-70")
            pw = ui.input("Password", password=True, password_toggle_button=True) \
                .props("outlined dark dense").classes("w-full") \
                .style("margin-top: 10px;")
            pw.on("keydown.enter", lambda e: _try_login())
            ui.button("Login", on_click=_try_login) \
                .classes("rex-btn-primary w-full").style("margin-top: 12px;")


@ui.page("/")
def main_page():
    ui.add_head_html(f"<style>{CUSTOM_CSS}</style>")
    ui.add_head_html(
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&'
        'family=JetBrains+Mono&display=swap" rel="stylesheet">'
    )

    # ============== HEADER (hero) ==============
    with ui.header().classes("w-full") \
            .style("background: transparent; box-shadow: none; padding: 10px 14px;"):
        with ui.element("div").classes("rex-hero w-full"):
            with ui.row().classes("items-center justify-between w-full"):
                with ui.row().classes("items-center gap-2"):
                    # ☰ toggles the nav drawer — essential on mobile where the
                    # drawer is hidden off-canvas.
                    ui.button(icon="menu", on_click=lambda: nav_drawer.toggle()) \
                        .props("flat round color=white").tooltip("Menu")
                    with ui.column().classes("gap-0"):
                        ui.label("🎬 Rex VFX Bot").classes("text-2xl font-bold")
                        ui.label("Claw Dashboard — script to short film, all in browser.") \
                            .classes("text-xs opacity-90")
                with ui.row().classes("items-center gap-2"):
                    gpu_label = ui.label(gpu_summary()).classes("text-sm font-bold").style(
                        "background: rgba(0,0,0,0.25); padding: 8px 14px; border-radius: 12px;"
                    )
                    if DASHBOARD_PASSWORD:
                        def _logout():
                            app.storage.user.update({"authenticated": False})
                            ui.navigate.to("/login")
                        ui.button(icon="logout", on_click=_logout) \
                            .props("flat round color=white").tooltip("Log out")

    # ============== LEFT NAV (vertical tabs) ==============
    # On phones Quasar hides the drawer off-canvas; the header ☰ button toggles
    # it. On desktop it stays open (value=True). On mobile the drawer opens as an
    # overlay; tapping the page (outside it) closes it.
    with ui.left_drawer(value=True, bordered=False) \
            .classes("rex-drawer").props("width=185") as nav_drawer:
        with ui.tabs().props("vertical").classes("rex-nav w-full") as nav_tabs:
            tab_pipeline = ui.tab("Pipeline", icon="movie")
            tab_settings = ui.tab("Settings", icon="tune")
            tab_models = ui.tab("Models", icon="swap_horiz")
            tab_queue = ui.tab("Queue", icon="pause_circle")
            tab_tools = ui.tab("Tools", icon="build")

    # Empty panels — section cards are built below at page level and then
    # .move()'d into the right panel (avoids re-indenting the whole UI; a single
    # card can't straddle two panels anyway).
    panels = ui.tab_panels(nav_tabs, value=tab_pipeline) \
        .props("animated").classes("w-full").style("background: transparent;")
    with panels:
        pipeline_panel = ui.tab_panel(tab_pipeline).classes("w-full")
        settings_panel = ui.tab_panel(tab_settings).classes("w-full")
        models_panel = ui.tab_panel(tab_models).classes("w-full")
        queue_panel = ui.tab_panel(tab_queue).classes("w-full")
        tools_panel = ui.tab_panel(tab_tools).classes("w-full")

    # ============== STAGE STEPPER ==============
    stepper_row = ui.row().classes("w-full items-center justify-between") \
        .style("margin: 8px 0 24px 0; padding: 0 24px;")
    step_elements: list = []
    step_lines: list = []
    with stepper_row:
        for idx, stage_name in enumerate(STAGES):
            with ui.column().classes("items-center gap-1"):
                circle = ui.label(str(idx + 1)).classes("rex-step")
                ui.label(stage_name.title()).classes("text-xs opacity-75")
                step_elements.append(circle)
            if idx < len(STAGES) - 1:
                line = ui.element("div").classes("rex-step-line").style("flex: 1; max-width: 80px;")
                step_lines.append(line)

    def refresh_stepper():
        active_idx = _stage_index(S.stage) if S.stage != "idle" else -1
        for i, el in enumerate(step_elements):
            cls = "rex-step"
            if i < active_idx:
                cls += " done"
            elif i == active_idx:
                cls += " active"
            el.classes(replace=cls)
        for i, line in enumerate(step_lines):
            if i < active_idx:
                line.classes(replace="rex-step-line done")
            else:
                line.classes(replace="rex-step-line")

    # ============== STAGE 1 — SCRIPT ==============
    with ui.card().classes("rex-card w-full") as card_script:
        with ui.row().classes("items-center w-full"):
            ui.label("1. Script").classes("text-xl font-bold")
            ui.element("span").classes("rex-badge rex-badge-pink").style("margin-left: 8px;") \
                ._props["innerHTML"] = "WRITE"

        with ui.row().classes("w-full gap-2 items-end"):
            theme_input = ui.input(label="Theme",
                                   placeholder="e.g. a tortoise teaches a hare patience") \
                .classes("flex-1").props("outlined dark dense")
            culture_select = ui.select(
                ["(auto)", "indian", "western", "japanese", "mixed", "animal-kingdom", "fantasy"],
                value="(auto)", label="Culture",
            ).props("outlined dark dense").style("min-width: 150px;")
            style_choices = ["(auto)"] + sg.get_available_style_ids()
            style_select = ui.select(style_choices, value="(auto)", label="Style") \
                .props("outlined dark dense").style("min-width: 150px;")

        with ui.row().classes("gap-2"):
            def _gen():
                generate_script_action(theme_input.value, culture_select.value,
                                       style_select.value, full_refresh)
            def _rand():
                theme_input.value = get_random_theme()
            ui.button("✨ Generate Script", on_click=_gen).classes("rex-btn-primary")
            ui.button("🎲 Random Theme", on_click=_rand).props("flat color=accent")

        script_picker = ui.select(
            options={sid: f"{sid} — {t}" for sid, t in list_scripts()},
            label="Or load an existing script",
            with_input=True,
        ).props("outlined dark dense").classes("w-full").style("margin-top: 8px;")

        def _load_picked():
            sid = script_picker.value
            if not sid:
                return
            path = SCRIPTS_DIR / f"script_{sid}.json"
            if path.exists():
                S.script = json.loads(path.read_text(encoding="utf-8"))
                S.script_id = sid
                S.push(f"Loaded {sid}")
                full_refresh()
        script_picker.on("update:model-value", lambda e: _load_picked())

        script_view = ui.markdown("_(no script loaded)_").classes("rex-log").style(
            "max-height: 320px; margin-top: 12px;"
        )

        with ui.row().classes("w-full gap-2 items-end").style("margin-top: 10px;"):
            revise_input = ui.input(label="Revision feedback",
                                    placeholder="e.g. make the ending happier") \
                .classes("flex-1").props("outlined dark dense")
            def _rev():
                revise_script_action(revise_input.value, full_refresh)
                revise_input.value = ""
            ui.button("✏️ Revise", on_click=_rev).props("flat color=accent")

        ui.button("✅ Approve Script → Generate Prompts",
                  on_click=lambda: approve_script_gen_prompts(full_refresh)) \
            .classes("rex-btn-primary").style("margin-top: 12px;")

    # ============== STAGE 2 — PROMPTS ==============
    with ui.card().classes("rex-card w-full") as card_prompts:
        with ui.row().classes("items-center"):
            ui.label("2. Prompts").classes("text-xl font-bold")
            ui.element("span").classes("rex-badge rex-badge-purple") \
                .style("margin-left: 8px;")._props["innerHTML"] = "TWEAK"

        prompts_container = ui.column().classes("w-full gap-2")
        ui.button("🚀 Approve ALL & Render Storyboard",
                  on_click=lambda: approve_all_run_storyboard(full_refresh)) \
            .classes("rex-btn-primary").style("margin-top: 10px;")

    # ============== STAGE 3 — STORYBOARD ==============
    with ui.card().classes("rex-card w-full") as card_storyboard:
        with ui.row().classes("items-center"):
            ui.label("3. Storyboard").classes("text-xl font-bold")
            ui.element("span").classes("rex-badge rex-badge-mint") \
                .style("margin-left: 8px;")._props["innerHTML"] = "DRAW"

        storyboard_container = ui.row().classes("w-full flex-wrap gap-3")
        ui.button("✅ Approve Storyboard → Render Video",
                  on_click=lambda: run_video(full_refresh)) \
            .classes("rex-btn-primary").style("margin-top: 12px;")

    # ============== STAGE 4 — VIDEO ==============
    with ui.card().classes("rex-card w-full") as card_video:
        with ui.row().classes("items-center"):
            ui.label("4. Video clips").classes("text-xl font-bold")
            ui.element("span").classes("rex-badge rex-badge-pink") \
                .style("margin-left: 8px;")._props["innerHTML"] = "ANIMATE"

        video_container = ui.row().classes("w-full flex-wrap gap-3")
        ui.button("✅ Approve Video → Final Assembly",
                  on_click=lambda: assemble_final(full_refresh)) \
            .classes("rex-btn-primary").style("margin-top: 12px;")

    # ============== STAGE 5 — FINAL ==============
    with ui.card().classes("rex-card w-full") as card_final:
        with ui.row().classes("items-center"):
            ui.label("5. Final assembly").classes("text-xl font-bold")
            ui.element("span").classes("rex-badge rex-badge-purple") \
                .style("margin-left: 8px;")._props["innerHTML"] = "SHIP"

        final_container = ui.row().classes("w-full flex-wrap gap-3")

    # ============== SETTINGS ==============
    with ui.card().classes("rex-card w-full") as card_settings:
        with ui.row().classes("items-center"):
            ui.label("⚙️ Settings").classes("text-xl font-bold")
            ui.element("span").classes("rex-badge rex-badge-purple") \
                .style("margin-left: 8px;")._props["innerHTML"] = "TUNE"

        def _notify_set(label, val):
            ui.notify(f"{label} → {val}", type="positive")

        with ui.row().classes("w-full gap-3 flex-wrap items-end"):
            style_set = ui.select(
                ["(auto)"] + sg.get_available_style_ids(),
                value=(rs.get_style_override() or "(auto)"), label="Style",
            ).props("outlined dark dense").style("min-width: 160px;")
            def _set_style(e):
                v = style_set.value
                if v == "(auto)":
                    rs.clear_style_override()
                else:
                    rs.set_style_override(v)
                _notify_set("Style", v)
            style_set.on("update:model-value", _set_style)

            aspect_set = ui.select(
                ["16:9", "9:16", "1:1"],
                value=(rs.get_resolution_override() or "16:9"), label="Aspect",
            ).props("outlined dark dense").style("min-width: 120px;")
            aspect_set.on("update:model-value",
                          lambda e: (rs.set_resolution_override(aspect_set.value),
                                     _notify_set("Aspect", aspect_set.value)))

            voice_set = ui.select(
                VOICE_CHOICES, value=rs.get_effective_voice(), label="Voice",
                with_input=True,
            ).props("outlined dark dense").style("min-width: 150px;")
            voice_set.on("update:model-value",
                         lambda e: (rs.set_voice_override(voice_set.value),
                                    _notify_set("Voice", voice_set.value)))

            music_set = ui.select(
                MUSIC_MOODS, value=(rs.get_music_mood_override() or MUSIC_MOODS[0]),
                label="Music mood",
            ).props("outlined dark dense").style("min-width: 150px;")
            music_set.on("update:model-value",
                         lambda e: (rs.set_music_mood_override(music_set.value),
                                    _notify_set("Music", music_set.value)))

        with ui.row().classes("w-full gap-3 flex-wrap items-end"):
            sync_set = ui.select(
                ["strict", "loose"], value=rs.get_effective_sync_mode(), label="Sync mode",
            ).props("outlined dark dense").style("min-width: 130px;")
            sync_set.on("update:model-value",
                        lambda e: (rs.set_sync_mode_override(sync_set.value),
                                   _notify_set("Sync", sync_set.value)))

            trans_set = ui.select(
                ["crossfade", "cut"], value=rs.get_effective_transition_mode(),
                label="Transition",
            ).props("outlined dark dense").style("min-width: 140px;")
            trans_set.on("update:model-value",
                         lambda e: (rs.set_transition_mode_override(trans_set.value),
                                    _notify_set("Transition", trans_set.value)))

            steps_set = ui.number(label="Steps", value=rs.get_steps_override() or 0,
                                  min=0, max=60, format="%d") \
                .props("outlined dark dense").style("width: 110px;")
            steps_set.on("blur", lambda e: (
                rs.set_steps_override(int(steps_set.value)) if steps_set.value
                else rs.clear_steps_override(),
                _notify_set("Steps", int(steps_set.value or 0))))

            cfg_set = ui.number(label="CFG", value=rs.get_cfg_override() or 0,
                                min=0, max=15, step=0.5, format="%.1f") \
                .props("outlined dark dense").style("width: 110px;")
            cfg_set.on("blur", lambda e: (
                rs.set_cfg_override(float(cfg_set.value)) if cfg_set.value
                else rs.clear_cfg_override(),
                _notify_set("CFG", cfg_set.value)))

        with ui.row().classes("w-full gap-4 items-center").style("margin-top: 6px;"):
            up_sw = ui.switch("Upscale (4×)", value=rs.get_upscale_enabled())
            up_sw.on("update:model-value",
                     lambda e: (rs.set_upscale_enabled(bool(up_sw.value)),
                                _notify_set("Upscale", up_sw.value)))
            ref_sw = ui.switch("Reference mode", value=rs.get_reference_mode_enabled())
            ref_sw.on("update:model-value",
                      lambda e: (rs.set_reference_mode_enabled(bool(ref_sw.value)),
                                 _notify_set("Reference", ref_sw.value)))

            def _show_current():
                ui.notify(json.dumps(rs.get_all_overrides(), indent=2) or "(none)",
                          multiline=True, close_button="OK", timeout=0)
            def _reset_all():
                rs.clear_all_overrides()
                ui.notify("All overrides reset.", type="warning")
            ui.button("📋 Show current", on_click=_show_current).props("flat dense")
            ui.button("♻️ Reset all", on_click=_reset_all).props("flat dense color=red")

    # ============== MODELS ==============
    with ui.card().classes("rex-card w-full") as card_models:
        with ui.row().classes("items-center"):
            ui.label("🔄 Models").classes("text-xl font-bold")
            ui.element("span").classes("rex-badge rex-badge-mint") \
                .style("margin-left: 8px;")._props["innerHTML"] = "SWAP"

        def _model_opts(btype):
            try:
                avail = model_registry.list_available(btype) or {}
                cur = model_registry.get_active(btype) or {}
                return list(avail.keys()), cur.get("_id")
            except Exception:
                return [], None

        with ui.row().classes("w-full gap-3 flex-wrap items-end"):
            img_opts, img_cur = _model_opts("image_backend")
            img_sel = ui.select(img_opts or ["(none)"], value=img_cur,
                                label="🖼️ Image backend") \
                .props("outlined dark dense").style("min-width: 240px;")
            img_sel.on("update:model-value", lambda e: (
                model_registry.set_active("image_backend", img_sel.value),
                rs.reset_overrides_for_model_switch(),
                ui.notify(f"Image backend → {img_sel.value} (settings reset to model defaults)",
                          type="positive")))

            vid_opts, vid_cur = _model_opts("video_backend")
            vid_sel = ui.select(vid_opts or ["(none)"], value=vid_cur,
                                label="🎥 Video backend") \
                .props("outlined dark dense").style("min-width: 240px;")
            vid_sel.on("update:model-value", lambda e: (
                model_registry.set_active("video_backend", vid_sel.value),
                rs.reset_overrides_for_model_switch(),
                ui.notify(f"Video backend → {vid_sel.value} (settings reset to model defaults)",
                          type="positive")))

    # ============== QUEUE (paused feedback) ==============
    with ui.card().classes("rex-card w-full") as card_queue:
        with ui.row().classes("items-center"):
            ui.label("⏸️ Queue").classes("text-xl font-bold")
            ui.element("span").classes("rex-badge rex-badge-pink") \
                .style("margin-left: 8px;")._props["innerHTML"] = "PAUSED"
        queue_container = ui.column().classes("w-full gap-2")

    # ============== TOOLS ==============
    with ui.card().classes("rex-card w-full") as card_tools:
        with ui.row().classes("items-center"):
            ui.label("🛠️ Tools").classes("text-xl font-bold")
            ui.element("span").classes("rex-badge rex-badge-purple") \
                .style("margin-left: 8px;")._props["innerHTML"] = "RUN"
        with ui.row().classes("gap-2 flex-wrap"):
            ui.button("✨ Upscale clips (4×)",
                      on_click=lambda: run_upscale_action(full_refresh)) \
                .props("flat color=accent")
            ui.button("🎬 Re-assemble final",
                      on_click=lambda: assemble_final(full_refresh)) \
                .props("flat color=accent")
            def _suggest():
                theme_input.value = get_random_theme()
                ui.notify("Theme suggested into Script box.", type="info")
            ui.button("💡 Suggest theme", on_click=_suggest).props("flat")

    # ============== LIVE LOG (collapsible footer) ==============
    with ui.footer().classes("p-0").style("background: rgba(8,8,18,0.96);"):
        with ui.expansion("📜 Live log", icon="terminal").props("dense") \
                .classes("w-full"):
            log_box = ui.html("<pre class='rex-log'>(idle)</pre>")

    # ---- Place each section into its tab panel ----
    # (Cards were built at page level above; move them now so the long single
    #  scroll becomes a left-nav tabbed layout. Child widget handles created
    #  inside each card stay valid after the parent moves.)
    stepper_row.move(pipeline_panel)
    card_script.move(pipeline_panel)
    card_prompts.move(pipeline_panel)
    card_storyboard.move(pipeline_panel)
    card_video.move(pipeline_panel)
    card_final.move(pipeline_panel)
    card_settings.move(settings_panel)
    card_models.move(models_panel)
    card_queue.move(queue_panel)
    card_tools.move(tools_panel)

    # =================== REFRESHERS ===================
    # Signature cache — skip rebuilding media/prompt containers when their
    # underlying content is unchanged. Without this the 1.5s timer destroys +
    # recreates every ui.image/ui.video/textarea each tick → visible flicker
    # (and wipes textareas mid-typing).
    _rcache: dict = {}

    def _changed(key: str, sig) -> bool:
        if _rcache.get(key) == sig:
            return False
        _rcache[key] = sig
        return True

    def render_script():
        if not S.script:
            script_view.set_content("_(no script loaded)_")
            return
        s = S.script
        md = [
            f"### 🎬 {s.get('title', 'Untitled')}",
            f"**ID:** `{s.get('_id', '?')}` · "
            f"**Theme:** _{s.get('theme', '?')}_ · "
            f"**Style:** `{s.get('style', '?')}` · "
            f"**Culture:** `{s.get('culture', '?')}`",
            "",
            f"**Setting:** {s.get('setting', '')}",
            f"**Moral:** {s.get('moral', '')}",
            "",
            "**Characters:**",
        ]
        for c in s.get("characters", []) or []:
            if isinstance(c, dict):
                voice = c.get("voice")
                vtag = f" · 🎙️ `{voice}`" if voice else ""
                md.append(f"- **{c.get('name', '?')}**{vtag} — "
                          f"{c.get('locked_visual_token', c.get('appearance', ''))[:140]}")
        md.append("")
        md.append("**Shots:**")
        for sh in s.get("shots", []) or []:
            beat = (sh.get("beat") or "").lower()
            spk = (sh.get("speaker") or "narrator")
            who = "🎬" if spk.lower() == "narrator" else f"💬 {spk}"
            md.append(f"- _Shot {sh.get('shot_number')}_ ({beat}, {who}) — "
                      f"{sh.get('narration', '')[:120]}")
        script_view.set_content("\n".join(md))

    def render_prompts():
        sid = S.script_id
        state = (pap.load_approved_prompts(sid) or {}) if sid else {}
        prompts = state.get("prompts", {})
        # Old script: storyboard rendered but no editable prompts JSON (e.g. made
        # before the approval step, or after a restart). Rebuild it from disk
        # (storyboard manifest + script) — no LLM — so shots become editable.
        if sid and not prompts:
            try:
                bf = pap.backfill_from_disk(sid, S.script)
                if bf and bf.get("prompts"):
                    state, prompts = bf, bf["prompts"]
            except Exception as e:
                log.warning(f"prompt backfill failed: {e}")
        # Narration source of truth = script JSON (shots[].narration), separate
        # from the image/motion prompts. Load it from disk so a Discord-side edit
        # shows here too and the textarea seeds with the current line.
        narr_by_shot: dict[int, str] = {}
        if sid:
            sp = SCRIPTS_DIR / f"script_{sid}.json"
            try:
                sd = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
                for sh in sd.get("shots", []) or []:
                    narr_by_shot[int(sh.get("shot_number", 0))] = sh.get("narration", "")
            except Exception:
                pass
        # Signature: rebuild when prompts OR narration change. Guards against
        # wiping textareas the user is mid-edit on each timer tick.
        sig = (sid, json.dumps(prompts, sort_keys=True),
               json.dumps(narr_by_shot, sort_keys=True))
        if not _changed("prompts", sig):
            return
        prompts_container.clear()
        if not sid:
            with prompts_container:
                ui.label("_(generate prompts to edit them)_").classes("opacity-60")
            return
        if not prompts:
            with prompts_container:
                ui.label("_(no prompts yet for this script)_").classes("opacity-60")
                # Old script loaded with no saved prompts → let user generate them
                # here (without scrolling back to Stage 1) so shots become editable.
                ui.button("✨ Generate prompts for this script",
                          on_click=lambda: approve_script_gen_prompts(full_refresh)) \
                    .classes("rex-btn-primary")
            return

        with prompts_container:
            for k in sorted(prompts.keys(), key=lambda x: int(x) if x.isdigit() else 0):
                p = prompts[k]
                shot_n = int(k)
                approved = p.get("approved", False)
                card_cls = "rex-shot-card w-full"
                if approved:
                    card_cls += " approved"
                with ui.element("div").classes(card_cls):
                    with ui.row().classes("items-center w-full"):
                        ui.label(f"Shot {shot_n}").classes("text-base font-bold")
                        ui.element("span") \
                            .classes("rex-badge rex-badge-mint") \
                            .style("margin-left: 8px;") \
                            ._props["innerHTML"] = p.get("beat", "?").upper()
                        if approved:
                            ui.label("✅ approved").classes("text-xs opacity-80") \
                                .style("margin-left: auto;")

                    # --- Narration (the spoken line) — edit or rewrite freely ---
                    ui.label("🎙️ Narration (spoken line)").classes("text-xs opacity-75")
                    narr_box = ui.textarea(value=narr_by_shot.get(shot_n, "")) \
                        .props("outlined dark dense autogrow") \
                        .classes("w-full")
                    with ui.row().classes("gap-2"):
                        def _save_narr(s=shot_n, box=narr_box):
                            if update_shot_narration(s, box.value):
                                ui.notify(f"✏️ Shot {s} narration saved", type="positive")
                            else:
                                ui.notify(f"❌ Shot {s} narration save failed",
                                          type="negative")
                            full_refresh()
                        def _clear_narr(s=shot_n, box=narr_box):
                            box.value = ""
                        def _ai_rewrite(s=shot_n):
                            with ui.dialog() as dlg, ui.card().classes("rex-card"):
                                ui.label(f"🪄 AI Rewrite — Shot {s} narration") \
                                    .classes("font-bold")
                                instr = ui.input(
                                    label="Instruction (optional)",
                                    placeholder="e.g. make it funnier / shorter / calmer",
                                ).props("outlined dark dense").classes("w-full") \
                                    .style("min-width: 320px;")
                                with ui.row().classes("gap-2 justify-end w-full"):
                                    ui.button("Cancel", on_click=dlg.close).props("flat dense")
                                    ui.button(
                                        "🪄 Rewrite",
                                        on_click=lambda: (dlg.close(),
                                                          ai_rewrite_narration(
                                                              s, instr.value, full_refresh)),
                                    ).props("color=purple dense")
                            dlg.open()
                        ui.button("💾 Save Narration", on_click=_save_narr) \
                            .props("flat dense color=teal")
                        ui.button("🪄 AI Rewrite", on_click=_ai_rewrite) \
                            .props("flat dense color=purple") \
                            .tooltip("Let the AI rewrite this line")
                        ui.button("🧹 Clear", on_click=_clear_narr).props("flat dense") \
                            .tooltip("Clear text to rewrite from scratch")

                    with ui.row().classes("w-full gap-2 items-start"):
                        with ui.column().classes("flex-1"):
                            ui.label(f"🖼️ Image · seed {p.get('image_seed', -1)}") \
                                .classes("text-xs opacity-75")
                            img_box = ui.textarea(value=p.get("image_prompt", "")) \
                                .props("outlined dark dense autogrow") \
                                .classes("w-full")
                        with ui.column().classes("flex-1"):
                            ui.label(f"🎥 Motion · seed {p.get('motion_seed', -1)}") \
                                .classes("text-xs opacity-75")
                            mot_box = ui.textarea(value=p.get("motion_prompt", "")) \
                                .props("outlined dark dense autogrow") \
                                .classes("w-full")

                    with ui.row().classes("gap-2"):
                        def _save_img(s=shot_n, box=img_box):
                            update_shot_prompt(s, "image", box.value, reseed=True)
                            ui.notify(f"✏️ Shot {s} image saved + reseeded", type="positive")
                            full_refresh()
                        def _reseed_img(s=shot_n):
                            update_shot_prompt(s, "image", "", reseed=True)
                            ui.notify(f"🎲 Shot {s} image reseeded", type="info")
                            full_refresh()
                        def _save_mot(s=shot_n, box=mot_box):
                            update_shot_prompt(s, "motion", box.value, reseed=True)
                            ui.notify(f"✏️ Shot {s} motion saved + reseeded", type="positive")
                            full_refresh()
                        def _reseed_mot(s=shot_n):
                            update_shot_prompt(s, "motion", "", reseed=True)
                            ui.notify(f"🎲 Shot {s} motion reseeded", type="info")
                            full_refresh()
                        def _approve_shot(s=shot_n):
                            state = pap.load_approved_prompts(sid) or {"prompts": {}}
                            state.setdefault("prompts", {}) \
                                .setdefault(str(s), {})["approved"] = True
                            pap._save(state)
                            pap._ACTIVE_STATES[sid] = state
                            ui.notify(f"✅ Shot {s} locked", type="positive")
                            full_refresh()
                        ui.button("💾 Save Image", on_click=_save_img).props("flat dense")
                        ui.button("🎲", on_click=_reseed_img).props("flat dense").tooltip("Reseed image")
                        ui.button("💾 Save Motion", on_click=_save_mot).props("flat dense")
                        ui.button("🎲", on_click=_reseed_mot).props("flat dense").tooltip("Reseed motion")
                        ui.button("✅ Approve Shot", on_click=_approve_shot) \
                            .props("color=teal dense")

    def render_storyboard():
        imgs = storyboard_images(S.script_id)
        sig = tuple((str(p), p.stat().st_mtime_ns) for p in imgs)
        if not _changed("storyboard", sig):
            return
        storyboard_container.clear()
        if not imgs:
            with storyboard_container:
                ui.label("_(no storyboard yet)_").classes("opacity-60")
            return
        with storyboard_container:
            for p in imgs:
                try:
                    shot_n = int(''.join(c for c in p.stem.split("shot")[-1] if c.isdigit()))
                except Exception:
                    shot_n = 0
                with ui.element("div").classes("rex-shot-card") \
                        .style("width: 220px;"):
                    ui.image(str(p)).style("border-radius: 8px; width: 100%;")
                    ui.label(p.stem).classes("text-xs opacity-75")
                    ui.button("🔁 Regen shot",
                              on_click=lambda s=shot_n: regen_storyboard_shot(s, full_refresh)) \
                        .props("flat dense").classes("w-full")

                    def _open_add_shot(s=shot_n):
                        with ui.dialog() as dlg, ui.card().classes("w-96"):
                            ui.label(f"➕ Insert a new shot next to shot {s}") \
                                .classes("text-lg font-bold")
                            where = ui.radio(["before", "after"], value="after") \
                                .props("inline")
                            brief = ui.textarea(
                                "What should happen? (optional — AI invents "
                                "a dramatic beat if left empty)"
                            ).classes("w-full")
                            with ui.row():
                                def _go():
                                    pos = s if where.value == "before" else s + 1
                                    dlg.close()
                                    add_shot_action(pos, brief.value or "",
                                                    full_refresh)
                                ui.button("➕ Insert", on_click=_go) \
                                    .props("color=teal")
                                ui.button("Cancel", on_click=dlg.close) \
                                    .props("flat")
                        dlg.open()

                    ui.button("➕ Add shot", on_click=_open_add_shot) \
                        .props("flat dense").classes("w-full") \
                        .tooltip("Insert a new shot before/after this one")

                    def _open_breathing(s=shot_n):
                        with ui.dialog() as dlg, ui.card().classes("w-96"):
                            ui.label(f"🌬️ Breathing shot next to shot {s}") \
                                .classes("text-lg font-bold")
                            ui.label("Quiet atmospheric beat — type the prompts "
                                     "yourself (no AI).").classes("text-xs opacity-75")
                            where = ui.radio(["before", "after"], value="after") \
                                .props("inline")
                            stype = ui.select(
                                ["wide", "medium", "closeup", "insert"],
                                value="wide", label="Shot type").classes("w-full")
                            img = ui.textarea(
                                "Image prompt (what's on screen) — required"
                            ).classes("w-full")
                            mot = ui.textarea(
                                "Motion prompt (camera/ambient move) — optional"
                            ).classes("w-full")
                            narr = ui.textarea(
                                "Narration (optional — leave empty for a silent hold)"
                            ).classes("w-full")
                            with ui.row():
                                def _go():
                                    pos = s if where.value == "before" else s + 1
                                    dlg.close()
                                    add_breathing_shot_action(
                                        pos, img.value or "", mot.value or "",
                                        narr.value or "", stype.value or "wide",
                                        full_refresh)
                                ui.button("🌬️ Insert", on_click=_go) \
                                    .props("color=teal")
                                ui.button("Cancel", on_click=dlg.close) \
                                    .props("flat")
                        dlg.open()

                    ui.button("🌬️ Breathing shot", on_click=_open_breathing) \
                        .props("flat dense").classes("w-full") \
                        .tooltip("Insert a manually-prompted quiet beat before/after")

    def render_video():
        clips = video_clips(S.script_id)
        sig = tuple((str(p), p.stat().st_mtime_ns) for p in clips)
        if not _changed("video", sig):
            return
        video_container.clear()
        if not clips:
            with video_container:
                ui.label("_(no clips yet)_").classes("opacity-60")
            return
        with video_container:
            for p in clips:
                try:
                    shot_n = int(''.join(c for c in p.stem.split("shot")[-1] if c.isdigit()))
                except Exception:
                    shot_n = 0
                with ui.element("div").classes("rex-shot-card") \
                        .style("width: 320px;"):
                    ui.video(str(p)).style("border-radius: 8px; width: 100%;")
                    ui.label(p.name).classes("text-xs opacity-75")
                    ui.button("🔁 Regen shot",
                              on_click=lambda s=shot_n: regen_video_shot(s, full_refresh)) \
                        .props("flat dense").classes("w-full")

    def render_final():
        sid = S.script_id
        final_dir = PROJECT_ROOT / "04_Outputs" / "final"
        finals = (sorted(final_dir.glob(f"final_{sid}_*.mp4"))
                  if sid and final_dir.exists() else [])
        placeholder = ("_(no final yet)_" if not sid else "_(not assembled yet)_")
        sig = tuple((str(p), p.stat().st_mtime_ns) for p in finals) or (placeholder,)
        if not _changed("final", sig):
            return
        final_container.clear()
        if not finals:
            with final_container:
                ui.label(placeholder).classes("opacity-60")
            return
        with final_container:
            for p in finals:
                with ui.element("div").classes("rex-shot-card") \
                        .style("width: 360px;"):
                    ui.video(str(p)).style("border-radius: 8px; width: 100%;")
                    ui.label(p.name).classes("text-xs opacity-75")

    def render_queue():
        try:
            items = pf.list_all()
        except Exception:
            items = []
        sig = tuple((i.get("script_id"), i.get("reason", "")) for i in items)
        if not _changed("queue", sig):
            return
        queue_container.clear()
        if not items:
            with queue_container:
                ui.label("_(no paused scripts)_").classes("opacity-60")
            return
        with queue_container:
            for it in items:
                qsid = it.get("script_id", "?")
                reason = it.get("reason", "")
                with ui.element("div").classes("rex-shot-card w-full"):
                    with ui.row().classes("items-center w-full gap-2"):
                        ui.label(f"⏸️ {qsid}").classes("font-bold")
                        ui.label(reason[:80]).classes("text-xs opacity-70")
                        def _load(s=qsid):
                            path = SCRIPTS_DIR / f"script_{s}.json"
                            if path.exists():
                                S.script = json.loads(path.read_text(encoding="utf-8"))
                                S.script_id = s
                                S.push(f"Loaded paused {s} into editor")
                                full_refresh()
                            else:
                                ui.notify(f"Script {s} file missing", type="negative")
                        def _drop(s=qsid):
                            pf.remove(s)
                            ui.notify(f"Dropped {s} from queue", type="warning")
                            full_refresh()
                        ui.button("📂 Load", on_click=_load).props("flat dense") \
                            .style("margin-left: auto;")
                        ui.button("🗑️ Drop", on_click=_drop).props("flat dense color=red")

    def render_log():
        text = "\n".join(S.log_lines[-30:]) or "(idle)"
        # escape angle brackets
        safe = text.replace("<", "&lt;").replace(">", "&gt;")
        cls = "rex-log"
        if S.busy:
            cls += " rex-pulse"
        log_box.content = f"<pre class='{cls}'>{safe}</pre>"

    def full_refresh():
        try:
            gpu_label.text = gpu_summary()
            refresh_stepper()
            render_script()
            render_prompts()
            render_storyboard()
            render_video()
            render_final()
            render_queue()
            render_log()
        except Exception as e:
            log.warning(f"refresh err: {e}")

    full_refresh()
    # Auto-refresh every 1.5s
    ui.timer(1.5, full_refresh)


# ==============================================================================
# LAUNCHER
# ==============================================================================

_LAUNCHED = False


def launch_dashboard(
    port: int = 7860,
    host: str = "127.0.0.1",
    open_browser: bool = True,
) -> Optional[threading.Thread]:
    """Spawn NiceGUI uvicorn in its own thread.

    NiceGUI's ui.run() expects to own the main thread by default; we spawn it
    in a daemon thread so it doesn't conflict with the Discord asyncio loop.
    """
    global _LAUNCHED
    if _LAUNCHED:
        log.info("Dashboard already launched.")
        return None

    def _runner():
        try:
            log.info(f"Launching NiceGUI dashboard on http://{host}:{port}")
            ui.run(
                host=host,
                port=port,
                show=open_browser,
                reload=False,
                title="Rex VFX Bot — Dashboard",
                favicon="🎬",
                dark=True,
                storage_secret="rex-vfx-claw-bot-local",
            )
        except Exception as e:
            log.exception(f"NiceGUI launch crashed: {e}")

    t = threading.Thread(target=_runner, daemon=True, name="ClawBotDashboardNiceGUI")
    t.start()
    _LAUNCHED = True
    return t


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Standalone: run on main thread (no Discord conflict)
    ui.run(host="127.0.0.1", port=7860, show=True, reload=False,
           title="Rex VFX Bot — Dashboard", favicon="🎬",
           dark=True, storage_secret="rex-vfx-claw-bot-local")
