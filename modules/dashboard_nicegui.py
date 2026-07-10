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
import re
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
from modules import job_recovery as jr
from modules import config_check
from modules import manual_mode as mm
from modules import voices

# How often a running job rewrites its stage into the recovery register.
CHECKPOINT_EVERY_SEC = 10.0

# Voice ids come from ONE place now (modules/voices.py). Hardcoding them here is
# how "Af_bella" got saved and then crashed the page build with "Invalid value".
VOICE_CHOICES = list(voices.KOKORO_VOICES)
FACTS_VOICE_CHOICES = list(voices.FACTS_VOICES)

# Option maps reused for both the widget and its _sel() clamp.
_FACTS_PACE = {0.92: "calm", 1.06: "lively", 1.20: "excited"}
_FACTS_VIDEO = {"kenburns": "Ken Burns (fast)", "wan": "Animate (slow)"}
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
_AUTH_OPEN_PREFIXES = ("/_nicegui", "/login", "/sb_static")


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

# Serve storyboards as static files so the browser can fetch each frame by URL.
# ui.image(local_path) gives a path-based URL that the browser caches across
# regenerations (same filename) — showing stale frames. Referencing the static
# URL with a ?v=<mtime> cache-buster forces a refetch when a frame changes.
try:
    app.add_static_files("/sb_static", str(STORYBOARDS_DIR))
except Exception as _e:
    logging.getLogger("claw_bot.dashboard").warning(f"sb_static mount failed: {_e}")

# Serve ALL of 04_Outputs as media so ui.video()/downloads work. Passing a raw
# absolute path to ui.video() makes the browser treat it as a URL (it can't fetch
# "E:\..."), so videos never played — this route fixes previews everywhere.
OUTPUTS_DIR = PROJECT_ROOT / "04_Outputs"
try:
    app.add_media_files("/media", str(OUTPUTS_DIR))
except Exception as _e:
    logging.getLogger("claw_bot.dashboard").warning(f"media mount failed: {_e}")


def _media_url(p) -> str:
    """Map a file under 04_Outputs to its /media URL (with an mtime cache-buster)."""
    p = Path(p)
    try:
        rel = p.resolve().relative_to(OUTPUTS_DIR).as_posix()
    except Exception:
        return str(p)
    mt = int(p.stat().st_mtime) if p.exists() else 0
    return f"/media/{rel}?v={mt}"
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
        self.current_action: str = ""   # label of the running job (for the status strip)
        self.log_lines: list[str] = []
        self.song: Optional[dict] = None       # music-video pipeline: current song
        self.horror: Optional[dict] = None     # horror pipeline: current story
        self.facts: Optional[dict] = None      # facts-shorts pipeline: current reel
        self.facts_stage: str = "idle"         # facts stepper: write|voice|images|assemble|done
        self.music_stage: str = "idle"         # music stepper: lyrics|song|visuals|assemble|done
        # Set by _bg_gpu while a job runs: {"id", "mode", "checkpointed"}.
        self.active_job: Optional[dict] = None

    def push(self, line: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_lines.append(f"[{ts}] {line}")
        self.log_lines = self.log_lines[-300:]
        log.info(line)
        self._checkpoint()

    def _checkpoint(self):
        """Persist the running job's stage as it progresses, so a resume after a
        crash re-enters near where it died. Throttled — every progress line
        would otherwise rewrite the register."""
        job = self.active_job
        if not job or not job.get("id"):
            return
        now = time.time()
        if now - job.get("checkpointed", 0.0) < CHECKPOINT_EVERY_SEC:
            return
        job["checkpointed"] = now
        mode = job["mode"]
        stage = {"facts": self.facts_stage, "music": self.music_stage}.get(mode, self.stage)
        try:
            jr.checkpoint(job["id"], stage=stage, **_job_context_for(mode, self))
        except Exception as e:
            log.debug(f"checkpoint failed: {e}")


S = State()
STAGES = ["script", "prompts", "storyboard", "video", "final"]


# ==============================================================================
# CUSTOM CSS / HEAD
# ==============================================================================

CUSTOM_CSS = """
:root{
  --rex-orange:#ff6a2b; --rex-orange-2:#ff4d0f;
  --rex-steel:#c9d1dc; --rex-panel:rgba(255,255,255,0.045);
  --rex-line:rgba(255,255,255,0.08);
}
body{
  background:
    radial-gradient(1100px 560px at 88% -12%, rgba(255,106,43,0.16), transparent 60%),
    radial-gradient(820px 480px at -5% 108%, rgba(70,110,200,0.10), transparent 55%),
    #0b0d12 !important;
  color:#e9edf5 !important;
  font-family:'Inter','Segoe UI',system-ui,sans-serif !important;
}
.q-page{background:transparent !important;}
::-webkit-scrollbar{width:10px;height:10px;}
::-webkit-scrollbar-thumb{background:rgba(255,106,43,0.35);border-radius:6px;}
::-webkit-scrollbar-thumb:hover{background:rgba(255,106,43,0.55);}
::-webkit-scrollbar-track{background:transparent;}

/* HERO with animated sheen */
.rex-hero{
  position:relative;overflow:hidden;
  background:linear-gradient(120deg,#1b1e28 0%,#111319 62%);
  border:1px solid var(--rex-line);border-radius:18px;padding:18px 26px;color:#fff;
  box-shadow:0 12px 40px rgba(0,0,0,0.45);
}
.rex-hero:before{
  content:"";position:absolute;top:0;bottom:0;left:0;width:40%;
  background:linear-gradient(90deg,transparent,rgba(255,106,43,0.10),transparent);
  animation:sheen 6.5s linear infinite;
}
@keyframes sheen{0%{transform:translateX(-120%);}100%{transform:translateX(360%);}}

/* GLASS CARDS */
.rex-card{
  background:var(--rex-panel) !important;border:1px solid var(--rex-line) !important;
  border-radius:16px !important;box-shadow:0 6px 24px rgba(0,0,0,0.30);
  backdrop-filter:blur(6px);margin-bottom:14px;
  transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease;
}
.rex-card:hover{
  transform:translateY(-2px);border-color:rgba(255,106,43,0.35) !important;
  box-shadow:0 12px 32px rgba(255,106,43,0.12);
}

/* DRAWER + NAV */
.rex-drawer{
  background:linear-gradient(180deg,#0f1218,#0b0d12) !important;
  border-right:1px solid var(--rex-line) !important;
}
.rex-nav .q-tab{
  justify-content:flex-start !important;min-height:48px;border-radius:12px;
  margin:3px 8px;color:#aab2c2 !important;font-weight:600;transition:all .15s ease;
}
.rex-nav .q-tab:hover{background:rgba(255,255,255,0.05) !important;color:#fff !important;}
.rex-nav .q-tab--active{
  background:linear-gradient(120deg,rgba(255,106,43,0.24),rgba(255,77,15,0.10)) !important;
  color:#fff !important;box-shadow:inset 3px 0 0 var(--rex-orange);
}
.rex-nav .q-tab__indicator{display:none;}

.rex-section-title{font-size:20px;font-weight:800;margin-bottom:4px;letter-spacing:.2px;}

/* STEPPER */
.rex-step{display:flex;align-items:center;justify-content:center;width:44px;height:44px;
  border-radius:50%;background:rgba(255,255,255,0.06);color:#e9edf5;font-weight:800;
  border:2px solid var(--rex-line);transition:all .4s ease;}
.rex-step.active{background:linear-gradient(120deg,var(--rex-orange),var(--rex-orange-2));
  border-color:#fff;transform:scale(1.15);box-shadow:0 0 22px rgba(255,106,43,0.7);}
.rex-step.done{background:linear-gradient(120deg,#3ad29c,#1ea97a);border-color:rgba(255,255,255,0.4);}
.rex-step-line{flex:1;height:3px;background:rgba(255,255,255,0.08);margin:0 4px;border-radius:2px;}
.rex-step-line.done{background:linear-gradient(90deg,#3ad29c,var(--rex-orange));box-shadow:0 0 10px rgba(255,106,43,0.4);}

/* BUTTONS */
.rex-btn-primary{
  background:linear-gradient(120deg,var(--rex-orange),var(--rex-orange-2)) !important;
  color:#fff !important;font-weight:700 !important;border-radius:14px !important;
  padding:12px 24px !important;box-shadow:0 8px 24px rgba(255,106,43,0.35) !important;
  transition:transform .15s ease,box-shadow .15s ease !important;letter-spacing:.3px;
}
.rex-btn-primary:hover{transform:translateY(-2px) scale(1.02) !important;
  box-shadow:0 12px 30px rgba(255,106,43,0.5) !important;}

.rex-pulse{animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.45;}}

/* MEDIA CARDS */
.rex-shot-card{background:rgba(255,255,255,0.05) !important;border-radius:14px !important;
  padding:10px !important;border:1px solid var(--rex-line) !important;transition:all .2s ease;}
.rex-shot-card:hover{border-color:rgba(255,106,43,0.4) !important;transform:translateY(-1px);}
.rex-shot-card.approved{border-color:rgba(58,210,156,0.6) !important;box-shadow:0 0 18px rgba(58,210,156,0.22) !important;}

/* BADGES */
.rex-badge{display:inline-block;padding:4px 12px;border-radius:14px;font-size:11px;
  font-weight:800;letter-spacing:.6px;text-transform:uppercase;}
.rex-badge-pink{background:linear-gradient(120deg,var(--rex-orange),var(--rex-orange-2));color:#fff;}
.rex-badge-purple{background:linear-gradient(120deg,#5b6cff,#3a3dff);color:#fff;}
.rex-badge-mint{background:linear-gradient(120deg,#3ad29c,#1ea97a);color:#08130f;}

/* LOG */
.rex-log{background:rgba(0,0,0,0.4) !important;border:1px solid var(--rex-line) !important;
  border-radius:12px !important;padding:14px !important;
  font-family:'JetBrains Mono','Consolas',monospace !important;font-size:12px !important;
  color:#9fb0c8 !important;max-height:280px;overflow-y:auto;}

/* nicer inputs */
.q-field--outlined .q-field__control{border-radius:12px !important;}
.q-field--outlined .q-field__control:before{border-color:var(--rex-line) !important;}
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
    """Admission check before queueing a GPU job.

    No longer claims the lock — jobs QUEUE now (see _bg_gpu). This runs on the
    UI thread, so it must never block: the actual wait happens in the worker.
    Returns False only when the job can't be accepted at all (low disk).
    """
    disk_ok, free_gb = config_check.check_disk_space()
    if not disk_ok:
        ui.notify(f"⚠️ Low disk: only {free_gb} GB free — clear 04_Outputs first.",
                  type="negative")
        return False
    depth = job_lock.queue_depth()
    if S.busy or depth:
        who = job_lock.holder_label()
        ui.notify(f"🧾 Queued behind {who}"
                  + (f" (+{depth} waiting)" if depth else "") + ".", type="info")
    return True


def _mode_for_label(label: str) -> str:
    """Which tab owns this job. Labels are distinctive enough to route on."""
    l = label.lower()
    if l.startswith("manual"):
        return "manual"
    if "facts" in l:
        return "facts"
    if "horror" in l:
        return "horror"
    if any(k in l for k in ("song", "music video", "lyrics")):
        return "music"
    return "story"


def _job_context_for(mode: str, state) -> dict:
    """Enough on-disk identity to reload this job's state after a crash.

    Each mode already persists its own JSON (script / facts / song / horror) and
    manual keeps a project manifest — so only the id needs recording.
    """
    if mode == "story" and state.script_id:
        return {"script_id": state.script_id}
    if mode == "facts" and state.facts:
        return {"facts_id": state.facts.get("_id") or state.facts.get("facts_id")}
    if mode == "music" and state.song:
        return {"song_id": state.song.get("_id") or state.song.get("song_id")}
    if mode == "horror" and state.horror:
        return {"horror_id": state.horror.get("_id") or state.horror.get("horror_id")}
    if mode == "manual":
        pid = mm.current_project_id()
        return {"project_id": pid} if pid else {}
    return {}


def _job_context(mode: str) -> dict:
    return _job_context_for(mode, S)


def _current_stage(mode: str) -> str:
    return {"facts": S.facts_stage, "music": S.music_stage}.get(mode, S.stage)


def _bg_gpu(label: str, worker) -> None:
    """Queue a GPU job and run it when its turn comes, first-come-first-served.

    The wait happens on the worker thread — blocking the UI thread would freeze
    the browser session. `worker` keeps its own `finally: _end()`, which is what
    releases the lock (and may run on this same thread).

    The job is also registered with job_recovery for the whole run, so a power
    cut or a killed bot leaves a record its tab can offer to resume. finish()
    runs even when the worker raises — a failed render is not an interrupted one.
    """
    mode = _mode_for_label(label)

    def runner():
        try:
            job_lock.acquire_blocking(
                f"dashboard:{label}",
                on_queued=lambda pos: S.push(f"🧾 Queued '{label}' — #{pos} in line "
                                             f"(running: {job_lock.holder_label()})"),
            )
        except Exception as e:                     # QueueTimeout or worse
            S.push(f"Could not start '{label}': {e}")
            return
        S.busy = True
        S.current_action = label
        job_id = ""
        try:
            job_id = jr.begin(mode, label, stage=_current_stage(mode),
                              context=_job_context(mode))
        except Exception as e:
            log.warning(f"recovery register failed: {e}")
        S.active_job = {"id": job_id, "mode": mode, "checkpointed": 0.0}
        try:
            worker()
        finally:
            S.active_job = None
            if job_id:
                jr.finish(job_id)
    threading.Thread(target=runner, daemon=True).start()


def _end():
    """Counterpart of _bg_gpu — call from the worker's finally block."""
    S.busy = False
    S.current_action = ""
    job_lock.release()


def _sel(value, options, default=None):
    """Clamp a stored setting to a value the select actually offers.

    NiceGUI raises "Invalid value: X" while BUILDING the page when a select's
    initial value isn't among its options — the whole dashboard then returns
    HTTP 500. A stale or hand-typed setting must never be able to do that.
    `options` may be a list or an {value: label} dict.
    """
    valid = list(options.keys()) if isinstance(options, dict) else list(options)
    if value in valid:
        return value
    if value is not None:
        log.warning(f"Setting {value!r} not in {valid[:6]}… — falling back")
    return default if default is not None else (valid[0] if valid else None)


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
            # Audio-first: voiceover first, its pauses dictate the cuts. Returns a
            # standard script JSON so the rest of the dashboard flow is unchanged.
            from modules.audio_first_pipeline import generate_script_audio_first
            script = generate_script_audio_first(
                theme, style_override=style_arg, culture_override=culture_arg,
                progress_cb=lambda m: S.push(f"· {m}"),
            )
            S.script = script
            S.script_id = script.get("_id") or script.get("script_id")
            S.push(f"Script {S.script_id} ready — '{script.get('title')}'")
        except Exception as e:
            S.push(f"Script gen failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg_gpu("script generation", worker)


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
    _bg_gpu("script revision", worker)


# ----------------------------------------------------------------------------
# MUSIC VIDEO PIPELINE (web parity)
# ----------------------------------------------------------------------------

def generate_song_action(theme: str, refresh_cb):
    """Write a song from a theme (music-video pipeline). Stores S.song."""
    if not theme.strip():
        ui.notify("❌ Theme required.", type="negative")
        return
    if not _try_begin("song generation"):
        return
    S.music_stage = "lyrics"
    S.push(f"Writing song — theme='{theme}'")
    refresh_cb()

    def worker():
        try:
            from modules import song_generator as sgn
            song = sgn.generate_song(theme, None)
            S.song = song
            S.push(f"Song ready — '{song.get('title')}' "
                   f"({song.get('song_style')}/{song.get('vocal_type')}, "
                   f"{len(song.get('scenes', []))} scenes). Review lyrics, then Render.")
        except Exception as e:
            S.push(f"Song gen failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg_gpu("song generation", worker)


def rewrite_song_lyrics_action(instruction: str, refresh_cb):
    if not S.song:
        ui.notify("❌ No song loaded.", type="negative")
        return
    # Queued like any other GPU job: this loads Qwen (~12.6 GB) into VRAM, and
    # doing that beside a running Wan render is what turned a 90s clip into 16 min.
    if not _try_begin("lyrics rewrite"):
        return

    def worker():
        try:
            from modules import song_generator as sgn
            S.song = sgn.rewrite_lyrics(S.song, instruction or "")
            S.push("Lyrics rewritten.")
        except Exception as e:
            S.push(f"Lyrics rewrite failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg_gpu("lyrics rewrite", worker)


def render_musicvideo_action(refresh_cb):
    """Render the approved song into music videos (audio + Ken Burns + assemble)."""
    if not S.song:
        ui.notify("❌ Generate a song first.", type="negative")
        return
    if not _try_begin("music video render"):
        return
    S.music_stage = "song"
    S.push("Rendering music video...")
    refresh_cb()

    def _pm(m):
        S.push(f"· {m}")
        ml = m.lower()
        if "ace-step" in ml or "rendering song" in ml or "song ready" in ml:
            S.music_stage = "song"
        elif "scene" in ml or "still" in ml:
            S.music_stage = "visuals"
        elif "assembl" in ml:
            S.music_stage = "assemble"
        elif "complete" in ml:
            S.music_stage = "done"

    def worker():
        try:
            from modules import musicvideo_pipeline as mvp
            outputs = mvp.render_musicvideo(S.song, progress_cb=_pm)
            S.music_stage = "done"
            done = [k for k in ("9x16", "16x9", "1x1") if outputs.get(k)]
            S.push(f"✅ Music video ready: {', '.join(done)} in 04_Outputs/final")
        except Exception as e:
            S.push(f"Music video render failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg_gpu("music video render", worker)


# ----------------------------------------------------------------------------
# HORROR STORY PIPELINE (web parity)
# ----------------------------------------------------------------------------

def generate_horror_action(theme: str, refresh_cb):
    """Write a long-form horror story (chunked) from a theme. Stores S.horror."""
    if not theme.strip():
        ui.notify("❌ Theme required.", type="negative")
        return
    if not _try_begin("horror writing"):
        return
    S.stage = "script"
    S.push(f"Writing horror story — theme='{theme}'")
    refresh_cb()

    def worker():
        try:
            from modules import horror_writer as hw
            story = hw.generate_horror_story(theme, 30, progress_cb=lambda m: S.push(f"· {m}"))
            S.horror = story
            words = sum(len(b.get("narration", "").split()) for b in story.get("beats", []))
            S.push(f"Horror ready — '{story.get('title')}' "
                   f"({len(story.get('beats', []))} beats, ~{words} words). Review, then Render.")
        except Exception as e:
            S.push(f"Horror writing failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg_gpu("horror writing", worker)


def render_horror_action(refresh_cb):
    """Render the approved horror story → 16x9 video (multi-hour)."""
    if not S.horror:
        ui.notify("❌ Generate a horror story first.", type="negative")
        return
    if not _try_begin("horror render"):
        return
    S.stage = "final"
    S.push("Rendering horror video (deep voice + photoreal stills)...")
    refresh_cb()

    def worker():
        try:
            from modules import horrorstory_pipeline as hsp
            out = hsp.render_horror(S.horror, progress_cb=lambda m: S.push(f"· {m}"))
            S.push(f"✅ Horror video ready: {out.get('16x9')}")
        except Exception as e:
            S.push(f"Horror render failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg_gpu("horror render", worker)


def generate_facts_action(topic: str, refresh_cb):
    """One-shot Facts Shorts: write true facts about a topic, then render the
    9x16 reel (energetic voice + creative images + read-along subs)."""
    if not (topic or "").strip():
        ui.notify("❌ Topic required.", type="negative")
        return
    if not _try_begin("facts reel"):
        return
    S.facts_stage = "write"
    S.push(f"Facts reel — topic='{topic}'")
    refresh_cb()

    def _p(m):
        S.push(f"· {m}")
        ml = m.lower()
        if "voicing" in ml or "narration" in ml:
            S.facts_stage = "voice"
        elif "backdrop" in ml or "building" in ml:
            S.facts_stage = "images"
        elif any(k in ml for k in ("assembl", "ken burns", "animating", "muxing")):
            S.facts_stage = "assemble"
        elif "complete" in ml:
            S.facts_stage = "done"

    def worker():
        try:
            from modules import facts_writer as fw
            from modules import facts_pipeline as fp
            story = fw.generate_facts_short(topic, 6, progress_cb=_p)
            S.facts = story
            S.facts_stage = "voice"
            n = len([b for b in story.get("beats", []) if b.get("kind") == "fact"])
            S.push(f"Facts written — '{story.get('title')}' ({n} facts). Rendering reel…")
            out = fp.render_facts(story, progress_cb=_p)
            S.facts_stage = "done"
            S.push(f"✅ Facts reel ready: {Path(out.get('9x16')).name}")
        except Exception as e:
            S.push(f"Facts reel failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg_gpu("facts reel", worker)


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
    _bg_gpu("prompt generation", worker)


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
    _bg_gpu("storyboard render", worker)


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
                # Audio-first: silent clip sized to the segment window; the master
                # VO is laid over at assembly (no per-shot TTS).
                if script.get("_audio_first") and float(shot.get("win_dur") or 0) > 0:
                    cg_out = clip_gen.generate_silent_clip(
                        shot_id=f"{sid}_shot{num}",
                        motion_prompt=motion,
                        storyboard_image=image_by_shot[num],
                        target_dur=float(shot["win_dur"]),
                        output_filename=f"clip_{sid}_shot{num}.mp4",
                        seed=seed_arg, beat=beat,
                    )
                else:
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
    _bg_gpu("video render", worker)


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
    _bg_gpu("final assembly", worker)


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
    # Audio-first: the master voiceover is already rendered. Editing the text
    # here would only change the caption, desyncing it from the spoken audio.
    # Block it — regenerate the script to change the wording.
    if script.get("_audio_first"):
        S.push("🎙️ Audio-first: narration is locked to the voiced track. "
               "Regenerate the script to change wording.")
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
    if (S.script or {}).get("_audio_first"):
        ui.notify("🎙️ Audio-first: narration is locked to the voiced track. "
                  "Regenerate the script to change wording.", type="warning")
        return
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
    _bg_gpu("narration rewrite", worker)


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
            aspect = rs.get_effective_aspect_ratio() or "16:9"
            # Render ONLY this shot (regenerate_shot), not the whole storyboard.
            sb_gen.regenerate_shot(
                script_id=sid, shot_number=shot_num,
                which_frame="first", aspect_ratio=aspect,
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
    _bg_gpu(f"storyboard regen shot {shot_num}", worker)


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
    _bg_gpu(f"add shot {position}", worker)


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
    _bg_gpu(f"breathing shot {position}", worker)


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
            # Audio-first: silent clip sized to the segment window (master VO laid
            # over at assembly); classic path does per-shot TTS.
            if script.get("_audio_first") and float(shot.get("win_dur") or 0) > 0:
                out = clip_gen.generate_silent_clip(
                    shot_id=f"{sid}_shot{shot_num}",
                    motion_prompt=motion,
                    storyboard_image=image_by_shot[shot_num],
                    target_dur=float(shot["win_dur"]),
                    output_filename=f"clip_{sid}_shot{shot_num}.mp4",
                    seed=seed_arg, beat=beat,
                )
            else:
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
    _bg_gpu(f"video regen shot {shot_num}", worker)


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
    _bg_gpu("upscale", worker)


# ==============================================================================
# INTERRUPTED-JOB RECOVERY (Resume / Discard)
# ==============================================================================
# A power cut or a killed bot leaves a record in job_recovery. Each mode's tab
# offers to resume it. Resume RESTARTS THE RECORDED STAGE — the artifacts of
# finished stages (script JSON, storyboard frames, rendered clips) are on disk
# and reused; the stage that died is re-run. It does not resume mid-frame.

def resume_facts_render_action(story: dict, refresh_cb):
    """Re-render a facts reel from its saved story JSON (skips the writing)."""
    if not _try_begin("facts reel"):
        return
    S.facts = story
    S.facts_stage = "voice"
    S.push(f"Resuming facts reel — '{story.get('title')}'")
    refresh_cb()

    def worker():
        try:
            from modules import facts_pipeline as fp
            out = fp.render_facts(story, progress_cb=lambda m: S.push(f"· {m}"))
            S.facts_stage = "done"
            S.push(f"✅ Facts reel ready: {Path(out.get('9x16')).name}")
        except Exception as e:
            S.push(f"Facts resume failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg_gpu("facts reel", worker)


def _resume_story(rec: dict, refresh_cb) -> Optional[str]:
    sid = (rec.get("context") or {}).get("script_id")
    if not sid:
        return "no script id was recorded"
    path = SCRIPTS_DIR / f"script_{sid}.json"
    if not path.exists():
        return f"script {sid} is gone from disk"
    S.script = json.loads(path.read_text(encoding="utf-8"))
    S.script_id = sid
    stage = rec.get("stage") or "script"
    # Re-enter at the stage that died. Earlier stages already left artifacts.
    if stage in ("idle", "script", "prompts"):
        approve_script_gen_prompts(refresh_cb)
    elif stage == "storyboard":
        approve_all_run_storyboard(refresh_cb)
    elif stage == "video":
        run_video(refresh_cb)
    elif stage == "final":
        assemble_final(refresh_cb)
    else:
        return f"unknown stage '{stage}'"
    return None


def _resume_facts(rec: dict, refresh_cb) -> Optional[str]:
    fid = (rec.get("context") or {}).get("facts_id")
    if not fid:
        return "the reel died before its facts were written — start a new one"
    from modules import facts_writer as fw
    story = fw.load_facts(fid)
    if not story:
        return f"facts {fid} is gone from disk"
    resume_facts_render_action(story, refresh_cb)
    return None


def _resume_music(rec: dict, refresh_cb) -> Optional[str]:
    sid = (rec.get("context") or {}).get("song_id")
    if not sid:
        return "the song was never saved — start a new one"
    from modules import song_generator as sgn
    song = sgn.load_song(sid)
    if not song:
        return f"song {sid} is gone from disk"
    S.song = song
    render_musicvideo_action(refresh_cb)
    return None


def _resume_horror(rec: dict, refresh_cb) -> Optional[str]:
    hid = (rec.get("context") or {}).get("horror_id")
    if not hid:
        return "the story was never saved — start a new one"
    from modules import horror_writer as hw
    story = hw.load_horror(hid)
    if not story:
        return f"horror story {hid} is gone from disk"
    S.horror = story
    render_horror_action(refresh_cb)
    return None


def _resume_manual(rec: dict, refresh_cb) -> Optional[str]:
    ctx = rec.get("context") or {}
    pid = ctx.get("project_id")
    if not pid or not mm.load_project(pid):
        return "the manual project is gone"
    mm.set_current(pid)
    label = (rec.get("label") or "").lower()
    proj = mm.load_project(pid)
    # Re-run the exact sub-job that died, when we can name it.
    m = re.search(r"shot (\d+)", label)
    if "animate" in label and m:
        manual_animate_action(int(m.group(1)), refresh_cb)
    elif "narration" in label and m:
        manual_narrate_action(int(m.group(1)), None, refresh_cb)
    elif "assembly" in label:
        manual_assemble_action([proj.get("aspect_ratio", "16:9")], refresh_cb)
    elif "music" in label:
        tags = (proj.get("music") or {}).get("tags", "")
        if not tags:
            return "no music tags recorded — use the Music box"
        manual_music_action(tags, "", refresh_cb)
    else:
        # image gen can't be replayed without the prompt; just reopen the board
        S.push(f"Reopened manual project {pid} — the board is intact.")
        refresh_cb()
    return None


_RESUMERS = {
    "story": _resume_story,
    "facts": _resume_facts,
    "music": _resume_music,
    "horror": _resume_horror,
    "manual": _resume_manual,
}


def resume_job_action(job_id: str, refresh_cb):
    rec = jr.get(job_id)
    if not rec:
        ui.notify("That job record is gone.", type="warning")
        refresh_cb()
        return
    if not jr.is_interrupted(rec):
        ui.notify("That job is running right now — nothing to resume.", type="warning")
        return
    resumer = _RESUMERS.get(rec.get("mode"))
    if resumer is None:
        ui.notify(f"No resume path for mode '{rec.get('mode')}'.", type="negative")
        return
    try:
        problem = resumer(rec, refresh_cb)
    except Exception as e:
        log.exception("resume failed")
        ui.notify(f"Resume failed: {e}", type="negative")
        return
    if problem:
        ui.notify(f"Can't resume: {problem}", type="negative")
        return
    # Only clear the record once the replacement job is actually under way.
    jr.discard(job_id)
    S.push(f"Resumed interrupted job: {rec.get('label')}")
    refresh_cb()


def discard_job_action(job_id: str, refresh_cb):
    rec = jr.get(job_id)
    jr.discard(job_id)
    S.push(f"Discarded interrupted job: {rec.get('label') if rec else job_id}")
    ui.notify("Discarded — start a new job whenever you like.", type="info")
    refresh_cb()


# ==============================================================================
# MANUAL MODE ACTIONS
# ==============================================================================
# Manual mode is direct-drive: user prompt → ComfyUI, no LLM chain. Current
# project lives on disk (mm.current_project_id) so Discord + web stay synced.

# Last freeform generation (preview → "Add to board"). Session-scoped is fine:
# one GPU box, one operator.
MANUAL_LAST_GEN: dict = {}
# Optional reference image for the next generation (uploaded via the UI).
MANUAL_REF: dict = {}


def _manual_proj() -> Optional[dict]:
    pid = mm.current_project_id()
    return mm.load_project(pid) if pid else None


def manual_gen_image_action(prompt: str, backend_id: Optional[str],
                            negative: str, seed_text: str, refresh_cb):
    if not (prompt or "").strip():
        ui.notify("❌ Prompt required.", type="negative")
        return
    proj = _manual_proj()
    if proj is None:
        ui.notify("❌ No manual project — create one first.", type="negative")
        return
    if not _try_begin("manual image gen"):
        return
    try:
        seed = int(seed_text) if str(seed_text or "").strip() else None
    except ValueError:
        seed = None
    ref = MANUAL_REF.get("path")
    S.push(f"Manual gen — backend={backend_id or '(active)'} seed={seed or 'rand'}")
    refresh_cb()

    def worker():
        try:
            res = mm.generate_image(
                prompt=prompt.strip(),
                backend_id=backend_id or None,
                negative_prompt=(negative or "").strip() or None,
                aspect_ratio=proj.get("aspect_ratio", "16:9"),
                seed=seed,
                reference_image=ref,
                proj=proj,
            )
            MANUAL_LAST_GEN.clear()
            MANUAL_LAST_GEN.update({"path": str(res["path"]), "seed": res["seed"],
                                    "prompt": prompt.strip()})
            S.push(f"Manual image done — seed={res['seed']} ({res['backend']})")
        except Exception as e:
            S.push(f"Manual gen failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg_gpu("manual image gen", worker)


def manual_add_last_to_board(refresh_cb, duration: float = 5.0):
    if not MANUAL_LAST_GEN.get("path"):
        ui.notify("❌ Nothing generated yet.", type="negative")
        return
    proj = _manual_proj()
    if proj is None:
        ui.notify("❌ No manual project.", type="negative")
        return
    try:
        shot = mm.add_shot(proj, Path(MANUAL_LAST_GEN["path"]),
                           prompt=MANUAL_LAST_GEN.get("prompt", ""),
                           seed=MANUAL_LAST_GEN.get("seed"),
                           duration=duration)
        S.push(f"Added shot {shot['n']} to manual board")
    except Exception as e:
        ui.notify(f"Add failed: {e}", type="negative")
        return
    refresh_cb()


def manual_animate_action(shot_n: int, refresh_cb):
    proj = _manual_proj()
    if proj is None:
        ui.notify("❌ No manual project.", type="negative")
        return
    try:
        shot = mm.get_shot(proj, shot_n)
    except IndexError as e:
        ui.notify(str(e), type="negative")
        return
    if not (shot.get("motion_prompt") or "").strip():
        ui.notify("❌ Set a motion prompt first (💾 save it), then animate.",
                  type="negative")
        return
    if not _try_begin(f"manual animate shot {shot_n}"):
        return
    S.push(f"Animating manual shot {shot_n}…")
    refresh_cb()

    def worker():
        try:
            clip = mm.animate_shot(proj, shot_n)
            S.push(f"Manual shot {shot_n} animated → {clip.name}")
        except Exception as e:
            S.push(f"Animate failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg_gpu(f"manual animate shot {shot_n}", worker)


def manual_narrate_action(shot_n: int, voice: Optional[str], refresh_cb):
    proj = _manual_proj()
    if proj is None:
        ui.notify("❌ No manual project.", type="negative")
        return
    try:
        shot = mm.get_shot(proj, shot_n)
    except IndexError as e:
        ui.notify(str(e), type="negative")
        return
    if not (shot.get("narration") or "").strip():
        ui.notify("❌ Type narration text and 💾 save it first.", type="negative")
        return
    if not _try_begin(f"manual narration shot {shot_n}"):
        return
    S.push(f"TTS narration for manual shot {shot_n}…")
    refresh_cb()

    def worker():
        try:
            wav = mm.narrate_shot(proj, shot_n, voice=voice or None)
            S.push(f"Narration ready → {wav.name}")
        except Exception as e:
            S.push(f"Narration failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg_gpu(f"manual narration shot {shot_n}", worker)


def manual_music_action(tags: str, duration_text: str, refresh_cb):
    if not (tags or "").strip():
        ui.notify("❌ Style tags required (e.g. 'lofi, chill, piano').",
                  type="negative")
        return
    proj = _manual_proj()
    if proj is None:
        ui.notify("❌ No manual project.", type="negative")
        return
    if not _try_begin("manual music gen"):
        return
    try:
        dur = float(duration_text) if str(duration_text or "").strip() else None
    except ValueError:
        dur = None
    S.push(f"Manual music gen — tags='{tags[:50]}'")
    refresh_cb()

    def worker():
        try:
            path = mm.generate_music(proj, tags.strip(), duration_sec=dur)
            S.push(f"Music ready → {path.name}")
        except Exception as e:
            S.push(f"Music gen failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg_gpu("manual music gen", worker)


def manual_delete_project_action(pid: str, refresh_cb):
    """Irreversible. Only call from behind a confirm dialog."""
    if not pid:
        ui.notify("❌ No project selected.", type="negative")
        return
    if S.busy:
        ui.notify("⏳ Busy — wait for the current job to finish.", type="warning")
        return
    try:
        stats = mm.delete_project(pid)
    except Exception as e:
        ui.notify(f"Delete failed: {e}", type="negative")
        return
    # A deleted project can't stay staged in the preview panes.
    MANUAL_LAST_GEN.clear()
    MANUAL_REF.pop("path", None)
    S.push(f"Manual project DELETED: {pid} "
           f"({stats['files']} files, {stats['mb']} MB)")
    ui.notify(f"🗑️ Deleted '{stats['name']}' — {stats['files']} files, "
              f"{stats['mb']} MB freed.", type="warning")
    refresh_cb()


def manual_assemble_action(aspects: list, refresh_cb):
    proj = _manual_proj()
    if proj is None:
        ui.notify("❌ No manual project.", type="negative")
        return
    if not proj["shots"]:
        ui.notify("❌ Board is empty.", type="negative")
        return
    if not _try_begin("manual assembly"):
        return
    S.push(f"Assembling manual project {proj['_id']} ({', '.join(aspects)})…")
    refresh_cb()

    def worker():
        try:
            finals = mm.assemble(proj, aspects=aspects,
                                 progress_cb=lambda m: S.push(f"  {m}"))
            for a, p in finals.items():
                S.push(f"Manual final [{a}] → {p.name}")
        except Exception as e:
            S.push(f"Manual assembly failed: {e}")
        finally:
            _end()
            refresh_cb()
    _bg_gpu("manual assembly", worker)


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
                        ui.label("🦖 REXJAW STUDIO") \
                            .classes("text-2xl font-bold") \
                            .style("letter-spacing:1.5px;background:linear-gradient("
                                   "120deg,#ff6a2b,#ffd0a8);-webkit-background-clip:text;"
                                   "-webkit-text-fill-color:transparent;")
                        ui.label("AI shorts factory — facts · music · stories, right in your browser.") \
                            .classes("text-xs opacity-80")
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

    # ============== STATUS STRIP (running / idle — visible in every tab) ==============
    with ui.row().classes("w-full items-center gap-2") \
            .style("padding: 6px 16px; background: rgba(0,0,0,0.18); "
                   "border-bottom: 1px solid rgba(255,255,255,0.06);"):
        status_spinner = ui.spinner(size="sm")
        status_label = ui.label("Idle").classes("text-sm font-bold")

    def _refresh_status():
        waiting = job_lock.queue_depth()
        queued = f"  ·  🧾 {waiting} queued" if waiting else ""
        if S.busy:
            status_spinner.set_visibility(True)
            status_label.text = (f"⏳ Running: {S.current_action}…  "
                                 f"(watch the log below){queued}")
            status_label.style("color:#ffcf5c;")
        elif waiting:
            # Another front-end (Discord / scheduler) holds the GPU.
            status_spinner.set_visibility(True)
            status_label.text = f"⏳ Waiting for {job_lock.holder_label()}{queued}"
            status_label.style("color:#ffcf5c;")
        else:
            status_spinner.set_visibility(False)
            status_label.text = "✓ Idle — ready"
            status_label.style("color:#6ee7a8;")

    # ============== LEFT NAV (vertical tabs) ==============
    # On phones Quasar hides the drawer off-canvas; the header ☰ button toggles
    # it. On desktop it stays open (value=True). On mobile the drawer opens as an
    # overlay; tapping the page (outside it) closes it.
    with ui.left_drawer(value=True, bordered=False) \
            .classes("rex-drawer").props("width=185") as nav_drawer:
        with ui.tabs().props("vertical").classes("rex-nav w-full") as nav_tabs:
            tab_pipeline = ui.tab("Story", icon="movie")
            tab_facts = ui.tab("Facts", icon="lightbulb")
            tab_music = ui.tab("Music", icon="music_note")
            tab_manual = ui.tab("Manual", icon="tune")
            tab_models = ui.tab("Models", icon="swap_horiz")
            tab_queue = ui.tab("Queue", icon="pause_circle")

    # Empty panels — section cards are built below at page level and then
    # .move()'d into the right panel (avoids re-indenting the whole UI; a single
    # card can't straddle two panels anyway).
    panels = ui.tab_panels(nav_tabs, value=tab_pipeline) \
        .props("animated").classes("w-full").style("background: transparent;")
    with panels:
        pipeline_panel = ui.tab_panel(tab_pipeline).classes("w-full")
        facts_panel = ui.tab_panel(tab_facts).classes("w-full")
        music_panel = ui.tab_panel(tab_music).classes("w-full")
        manual_panel = ui.tab_panel(tab_manual).classes("w-full")
        models_panel = ui.tab_panel(tab_models).classes("w-full")
        queue_panel = ui.tab_panel(tab_queue).classes("w-full")

    # ============== INTERRUPTED-JOB BANNERS (one per mode tab) ==============
    # Shown only when a job of that mode was left behind by a dead process:
    # you shut the PC down, killed the bot, or lost power mid-render.
    recovery_cards: dict = {}
    recovery_bodies: dict = {}
    for _mode in ("story", "facts", "music", "horror", "manual"):
        with ui.card().classes("rex-card w-full") \
                .style("border-left:4px solid #ffcf5c;") as _card:
            with ui.row().classes("items-center"):
                ui.label("⚠️ Interrupted job").classes("text-lg font-bold") \
                    .style("color:#ffcf5c;")
            recovery_bodies[_mode] = ui.column().classes("w-full gap-2")
        _card.set_visibility(False)
        recovery_cards[_mode] = _card

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

    # --- reusable step tracker for the Facts + Music tabs ---
    FACTS_STAGES = ["Write", "Voice", "Images", "Assemble", "Done"]
    MUSIC_STAGES = ["Lyrics", "Song", "Visuals", "Assemble", "Done"]
    _FACTS_KEYS = ["write", "voice", "images", "assemble", "done"]
    _MUSIC_KEYS = ["lyrics", "song", "visuals", "assemble", "done"]

    def _build_stepper(labels):
        circles, lines = [], []
        with ui.row().classes("w-full items-center justify-between") \
                .style("padding:2px 6px 12px;"):
            for i, name in enumerate(labels):
                with ui.column().classes("items-center gap-1"):
                    circles.append(ui.label(str(i + 1)).classes("rex-step"))
                    ui.label(name).classes("text-xs opacity-75")
                if i < len(labels) - 1:
                    lines.append(ui.element("div").classes("rex-step-line")
                                 .style("flex:1;max-width:60px;"))
        return circles, lines

    def _paint_stepper(circles, lines, active_idx):
        for i, el in enumerate(circles):
            cls = "rex-step"
            if i < active_idx:
                cls += " done"
            elif i == active_idx:
                cls += " active"
            el.classes(replace=cls)
        for i, l in enumerate(lines):
            l.classes(replace="rex-step-line done" if i < active_idx else "rex-step-line")

    facts_steps = ([], [])
    music_steps = ([], [])

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
            style_choices = ["(auto)"] + sg.get_style_ids_for_mode("story")
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

    # ============== MUSIC VIDEO (mode: music_video) ==============
    with ui.card().classes("rex-card w-full") as card_musicvideo:
        with ui.row().classes("items-center"):
            ui.label("🎵 Music Video").classes("text-xl font-bold")
            ui.element("span").classes("rex-badge rex-badge-purple") \
                .style("margin-left: 8px;")._props["innerHTML"] = "SONG"
        ui.label("ACE-Step song + Ken Burns visuals (9x16/16x9/1x1). Style / vocal / "
                 "tempo / visual set in Settings or the Discord panel (or auto).") \
            .classes("text-xs opacity-70")
        music_steps = _build_stepper(MUSIC_STAGES)

        with ui.row().classes("w-full gap-2 items-end").style("margin-top: 6px;"):
            song_theme = ui.input(label="Song theme",
                                  placeholder="e.g. a rainy night drive through the city") \
                .classes("flex-1").props("outlined dark dense")
            ui.button("🎲", on_click=lambda: song_theme.set_value(get_random_theme())) \
                .props("flat color=accent").tooltip("Random theme")

        with ui.row().classes("gap-2"):
            ui.button("🎼 Generate Song",
                      on_click=lambda: generate_song_action(song_theme.value, full_refresh)) \
                .classes("rex-btn-primary")

            def _show_lyrics():
                if not S.song:
                    ui.notify("No song yet — Generate Song first.", type="warning")
                    return
                ui.notify(S.song.get("lyrics", "(empty)"), multiline=True,
                          close_button="OK", timeout=0)
            ui.button("📋 Show Lyrics", on_click=_show_lyrics).props("flat color=accent")

        with ui.row().classes("w-full gap-2 items-end").style("margin-top: 6px;"):
            lyric_fb = ui.input(label="Edit lyrics (instruction)",
                                placeholder="e.g. make the chorus catchier") \
                .classes("flex-1").props("outlined dark dense")
            def _rewrite():
                rewrite_song_lyrics_action(lyric_fb.value, full_refresh)
                lyric_fb.value = ""
            ui.button("🪄 Rewrite Lyrics", on_click=_rewrite).props("flat color=accent")

        ui.button("✅ Approve Song → Render Music Video",
                  on_click=lambda: render_musicvideo_action(full_refresh)) \
            .classes("rex-btn-primary").style("margin-top: 12px;")

        ui.label("Latest music video:").classes("text-xs opacity-70").style("margin-top: 10px;")
        music_container = ui.column().classes("w-full")

    # ============== HORROR STORY (mode: horror_story) ==============
    with ui.card().classes("rex-card w-full") as card_horror:
        with ui.row().classes("items-center"):
            ui.label("🎃 Horror Story").classes("text-xl font-bold")
            ui.element("span").classes("rex-badge rex-badge-pink") \
                .style("margin-left: 8px;")._props["innerHTML"] = "DREAD"
        ui.label("~30-min narrated horror: deep voice + photoreal Ken Burns (16x9). "
                 "Render is multi-hour.").classes("text-xs opacity-70")

        with ui.row().classes("w-full gap-2 items-end").style("margin-top: 6px;"):
            horror_theme = ui.input(label="Horror theme",
                                    placeholder="e.g. an abandoned lighthouse that calls people into the sea") \
                .classes("flex-1").props("outlined dark dense")
            ui.button("🎲", on_click=lambda: horror_theme.set_value(get_random_theme())) \
                .props("flat color=accent").tooltip("Random theme")

        with ui.row().classes("gap-2"):
            ui.button("✍️ Write Horror Story",
                      on_click=lambda: generate_horror_action(horror_theme.value, full_refresh)) \
                .classes("rex-btn-primary")

            def _show_horror():
                if not S.horror:
                    ui.notify("No story yet — Write Horror Story first.", type="warning")
                    return
                b = S.horror.get("beats", [])
                txt = f"{S.horror.get('title','')}\n\n" + "\n\n".join(x.get("narration","") for x in b[:6])
                ui.notify(txt, multiline=True, close_button="OK", timeout=0)
            ui.button("📋 Preview", on_click=_show_horror).props("flat color=accent")

            ambient_sw = ui.switch("Ambient drone", value=rs.get_horror_ambient_enabled())
            ambient_sw.on("update:model-value",
                          lambda e: rs.set_horror_ambient_enabled(bool(ambient_sw.value)))

        ui.button("✅ Approve Story → Render Horror Video",
                  on_click=lambda: render_horror_action(full_refresh)) \
            .classes("rex-btn-primary").style("margin-top: 12px;")

    # ============== FACTS SHORTS (own tab) ==============
    with ui.card().classes("rex-card w-full") as card_facts:
        with ui.row().classes("items-center"):
            ui.label("💡 Facts Shorts").classes("text-xl font-bold")
            ui.element("span").classes("rex-badge rex-badge-purple") \
                .style("margin-left: 8px;")._props["innerHTML"] = "9x16 · IG"
        ui.label("Text-forward reel: true facts + energetic voice + creative images + "
                 "read-along subtitles. Vertical, Instagram-ready. ~4 min.") \
            .classes("text-xs opacity-70")
        facts_steps = _build_stepper(FACTS_STAGES)

        with ui.row().classes("w-full gap-2 items-end").style("margin-top: 6px;"):
            facts_topic = ui.input(label="Topic",
                                   placeholder="e.g. the deep ocean, black holes, the human brain") \
                .classes("flex-1").props("outlined dark dense")

        with ui.row().classes("gap-2 items-center").style("margin-top: 4px;"):
            ui.button("💡 Generate Facts Reel",
                      on_click=lambda: generate_facts_action(facts_topic.value, full_refresh)) \
                .classes("rex-btn-primary")
            facts_voice_sel = ui.select(
                list(FACTS_VOICE_CHOICES),
                value=_sel(rs.get_facts_voice(), FACTS_VOICE_CHOICES,
                           voices.DEFAULT_FACTS_VOICE), label="Voice") \
                .props("outlined dark dense").style("min-width: 130px")
            facts_voice_sel.on("update:model-value",
                               lambda e: rs.set_facts_voice(facts_voice_sel.value))
            facts_pace_sel = ui.select(
                _FACTS_PACE,
                value=_sel(rs.get_facts_voice_speed(), _FACTS_PACE, 1.06),
                label="Pace") \
                .props("outlined dark dense").style("min-width: 120px")
            facts_pace_sel.on("update:model-value",
                              lambda e: rs.set_facts_voice_speed(float(facts_pace_sel.value)))
            facts_video_sel = ui.select(
                _FACTS_VIDEO,
                value=_sel(rs.get_facts_video_mode(), _FACTS_VIDEO, "kenburns"),
                label="Video") \
                .props("outlined dark dense").style("min-width: 150px")
            facts_video_sel.on("update:model-value",
                               lambda e: rs.set_facts_video_mode(facts_video_sel.value))

        ui.label("Latest reel:").classes("text-xs opacity-70").style("margin-top: 10px;")
        facts_container = ui.column().classes("w-full")

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
            _style_opts = ["(auto)"] + sg.get_style_ids_for_mode("story")
            _style_ov = rs.get_style_override()
            style_set = ui.select(
                _style_opts,
                # Guard against a stale saved style id no longer in the list
                # (would crash the page with "Invalid value").
                value=(_style_ov if _style_ov in _style_opts else "(auto)"),
                label="Style",
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
                value=_sel(rs.get_resolution_override(), ["16:9", "9:16", "1:1"], "16:9"),
                label="Aspect",
            ).props("outlined dark dense").style("min-width: 120px;")
            aspect_set.on("update:model-value",
                          lambda e: (rs.set_resolution_override(aspect_set.value),
                                     _notify_set("Aspect", aspect_set.value)))

            voice_set = ui.select(
                VOICE_CHOICES,
                value=_sel(rs.get_effective_voice(), VOICE_CHOICES, voices.DEFAULT_VOICE),
                label="Voice",
                with_input=True,
            ).props("outlined dark dense").style("min-width: 150px;")
            voice_set.on("update:model-value",
                         lambda e: (rs.set_voice_override(voice_set.value),
                                    _notify_set("Voice", voice_set.value)))

            music_set = ui.select(
                MUSIC_MOODS,
                value=_sel(rs.get_music_mood_override(), MUSIC_MOODS, MUSIC_MOODS[0]),
                label="Music mood",
            ).props("outlined dark dense").style("min-width: 150px;")
            music_set.on("update:model-value",
                         lambda e: (rs.set_music_mood_override(music_set.value),
                                    _notify_set("Music", music_set.value)))

        with ui.row().classes("w-full gap-3 flex-wrap items-end"):
            sync_set = ui.select(
                ["strict", "loose"],
                value=_sel(rs.get_effective_sync_mode(), ["strict", "loose"], "strict"),
                label="Sync mode",
            ).props("outlined dark dense").style("min-width: 130px;")
            sync_set.on("update:model-value",
                        lambda e: (rs.set_sync_mode_override(sync_set.value),
                                   _notify_set("Sync", sync_set.value)))

            trans_set = ui.select(
                ["crossfade", "cut"],
                value=_sel(rs.get_effective_transition_mode(), ["crossfade", "cut"],
                           "crossfade"),
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

            # Video generation resolution preset (parity with !set_video_resolution).
            # "(auto)" = the video model's own models.json default dims.
            _vres_ov = rs.get_video_resolution_override()
            vres_set = ui.select(
                ["(auto)"] + list(rs.VIDEO_RES_PRESETS.keys()),
                value=(_vres_ov if _vres_ov in rs.VIDEO_RES_PRESETS else "(auto)"),
                label="Video res",
            ).props("outlined dark dense").style("min-width: 120px;")
            vres_set.on("update:model-value", lambda e: (
                rs.clear_video_resolution_override() if vres_set.value == "(auto)"
                else rs.set_video_resolution_override(vres_set.value),
                _notify_set("Video res", vres_set.value)))

            # Clip-length override (parity with !set_clip_length). 0 = match
            # narration (audio-first sync). 1–30s forces a fixed clip length.
            clip_set = ui.number(label="Clip sec", value=rs.get_clip_length_override() or 0,
                                 min=0, max=30, step=0.5, format="%.1f") \
                .props("outlined dark dense").style("width: 110px;")
            clip_set.on("blur", lambda e: (
                rs.set_clip_length_override(float(clip_set.value))
                if clip_set.value and float(clip_set.value) >= 1.0
                else rs.clear_clip_length_override(),
                _notify_set("Clip sec", clip_set.value or "auto")))

        with ui.row().classes("w-full gap-4 items-center").style("margin-top: 6px;"):
            up_sw = ui.switch("Upscale (4×)", value=rs.get_upscale_enabled())
            up_sw.on("update:model-value",
                     lambda e: (rs.set_upscale_enabled(bool(up_sw.value)),
                                _notify_set("Upscale", up_sw.value)))
            ref_sw = ui.switch("Reference mode", value=rs.get_reference_mode_enabled())
            ref_sw.on("update:model-value",
                      lambda e: (rs.set_reference_mode_enabled(bool(ref_sw.value)),
                                 _notify_set("Reference", ref_sw.value)))

        # --- Music-video tuning (mode itself lives on the Pipeline tab) ---
        # Option lists are named so the widget and its _sel() clamp can't drift.
        _song_style_opts = ["(auto)"] + list(rs.VALID_SONG_STYLES)
        _vocal_opts = ["(auto)"] + [v for v in rs.VALID_VOCAL_TYPES if v != "auto"]
        _visual_opts = ["(auto)"] + list(rs.VALID_VISUAL_STYLES)
        with ui.row().classes("w-full gap-3 flex-wrap items-end").style("margin-top: 6px;"):
            ui.label("🎵 Music").classes("text-sm font-bold opacity-80")
            song_style_set = ui.select(
                _song_style_opts,
                value=_sel(rs.get_song_style_override() or "(auto)",
                           _song_style_opts, "(auto)"), label="Song style",
            ).props("outlined dark dense").style("min-width: 140px;")
            song_style_set.on("update:model-value", lambda e: (
                rs.clear_song_style_override() if song_style_set.value == "(auto)"
                else rs.set_song_style_override(song_style_set.value),
                _notify_set("Song style", song_style_set.value)))

            vocal_set = ui.select(
                _vocal_opts,
                value=_sel(rs.get_vocal_type_override() or "(auto)",
                           _vocal_opts, "(auto)"), label="Vocal type",
            ).props("outlined dark dense").style("min-width: 140px;")
            vocal_set.on("update:model-value", lambda e: (
                rs.clear_vocal_type_override() if vocal_set.value == "(auto)"
                else rs.set_vocal_type_override(vocal_set.value),
                _notify_set("Vocal", vocal_set.value)))

            visual_set = ui.select(
                _visual_opts,
                value=_sel(rs.get_visual_style_override() or "(auto)",
                           _visual_opts, "(auto)"), label="Visual style",
            ).props("outlined dark dense").style("min-width: 140px;")
            visual_set.on("update:model-value", lambda e: (
                rs.clear_visual_style_override() if visual_set.value == "(auto)"
                else rs.set_visual_style_override(visual_set.value),
                _notify_set("Visual", visual_set.value)))

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

            def _show_stats():
                # Parity with Discord !stats — read the shared on-disk counters.
                p = PROJECT_ROOT / "05_Config" / "stats.json"
                try:
                    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
                except Exception as e:
                    ui.notify(f"stats unreadable: {e}", type="negative")
                    return
                if not d:
                    ui.notify("No stats yet.", type="info")
                    return
                lines = [f"{k}: {v}" for k, v in d.items()]
                lines.append(f"theme bank: {get_theme_count()}")
                ui.notify("\n".join(lines), multiline=True,
                          close_button="OK", timeout=0)
            ui.button("📊 Stats", on_click=_show_stats).props("flat")

    # ============== MANUAL MODE ==============
    with ui.card().classes("rex-card w-full") as card_manual:
        with ui.row().classes("items-center"):
            ui.label("🎛️ Manual Mode").classes("text-xl font-bold")
            ui.element("span").classes("rex-badge rex-badge-purple") \
                .style("margin-left: 8px;")._props["innerHTML"] = "DIRECT DRIVE"
        ui.label("You are the director: pick model, write prompt, build the board "
                 "shot by shot, animate, narrate, add music, assemble.") \
            .classes("text-xs opacity-70")

        # ---- project row ----
        with ui.row().classes("w-full gap-2 items-end flex-wrap"):
            manual_proj_select = ui.select(options={}, label="Project",
                                           with_input=True) \
                .props("outlined dark dense").style("min-width: 260px;")

            def _pick_proj():
                if manual_proj_select.value:
                    mm.set_current(manual_proj_select.value)
                    S.push(f"Manual project → {manual_proj_select.value}")
                    full_refresh()
            manual_proj_select.on("update:model-value", lambda e: _pick_proj())

            manual_new_name = ui.input(label="New project name",
                                       placeholder="e.g. dragon test reel") \
                .props("outlined dark dense").style("min-width: 200px;")
            manual_aspect_sel = ui.select(["16:9", "9:16", "1:1"], value="16:9",
                                          label="Aspect") \
                .props("outlined dark dense").style("min-width: 100px;")

            def _new_proj():
                proj = mm.create_project(manual_new_name.value or "",
                                         aspect_ratio=manual_aspect_sel.value)
                manual_new_name.value = ""
                S.push(f"Manual project created: {proj['_id']}")
                ui.notify(f"Project ready: {proj['name']}", type="positive")
                full_refresh()
            ui.button("➕ New project", on_click=_new_proj).props("flat color=accent")

            def _confirm_delete():
                pid = mm.current_project_id()
                if not pid:
                    ui.notify("No project to delete.", type="negative")
                    return
                try:
                    st = mm.project_stats(pid)
                except Exception as e:
                    ui.notify(f"Cannot read project: {e}", type="negative")
                    return
                with ui.dialog() as dlg, ui.card().classes("rex-card"):
                    ui.label("🗑️ Delete this project?").classes("text-lg font-bold")
                    ui.label(f"'{st['name']}'  ({pid})").classes("text-sm opacity-80")
                    ui.label(f"{st['shots']} shots · {st['finals']} final render(s) · "
                             f"{st['files']} files · {st['mb']} MB") \
                        .classes("text-sm").style("color:#ffcf5c;")
                    ui.label("Permanent. Stills, clips, narration and finals all go. "
                             "This cannot be undone.") \
                        .classes("text-xs").style("color:#ff8a8a;")
                    with ui.row().classes("gap-2 justify-end w-full"):
                        ui.button("Cancel", on_click=dlg.close).props("flat")

                        def _go():
                            dlg.close()
                            manual_delete_project_action(pid, full_refresh)
                        ui.button("Delete forever", on_click=_go) \
                            .props("color=red unelevated")
                dlg.open()
            ui.button("🗑️ Delete project", on_click=_confirm_delete) \
                .props("flat color=red").tooltip("Permanently delete the current project")

        # ---- generate section ----
        ui.separator()
        ui.label("1 · Generate a still (or upload your own)") \
            .classes("text-sm font-bold opacity-80")
        with ui.row().classes("w-full gap-2 items-end flex-wrap"):
            def _img_backend_opts():
                try:
                    return ["(active)"] + list(
                        (model_registry.list_available("image_backend") or {}).keys())
                except Exception:
                    return ["(active)"]
            manual_backend_sel = ui.select(_img_backend_opts(), value="(active)",
                                           label="Image model") \
                .props("outlined dark dense").style("min-width: 220px;")
            manual_seed_in = ui.input(label="Seed (blank = random)") \
                .props("outlined dark dense").style("width: 160px;")
            manual_steps_in = ui.input(label="Steps (override)") \
                .props("outlined dark dense").style("width: 130px;")
            manual_cfg_in = ui.input(label="CFG (override)") \
                .props("outlined dark dense").style("width: 130px;")

            def _apply_overrides():
                # Same global knobs the Settings card writes — adapters read them.
                try:
                    if str(manual_steps_in.value or "").strip():
                        rs.set_steps_override(int(manual_steps_in.value))
                    if str(manual_cfg_in.value or "").strip():
                        rs.set_cfg_override(float(manual_cfg_in.value))
                    ui.notify("Overrides applied.", type="positive")
                except ValueError:
                    ui.notify("Steps/CFG must be numbers.", type="negative")
            ui.button("⚙️ Apply", on_click=_apply_overrides).props("flat dense") \
                .tooltip("Write steps/CFG overrides (same as Settings)")

        manual_prompt_in = ui.textarea(
            label="Image prompt",
            placeholder="e.g. a colossal rusted robot kneeling in a sunflower field, "
                        "golden hour, cinematic") \
            .classes("w-full").props("outlined dark dense autogrow")
        manual_negative_in = ui.input(label="Negative prompt (optional)") \
            .classes("w-full").props("outlined dark dense")

        with ui.row().classes("w-full gap-2 items-center flex-wrap"):
            # NiceGUI 3.x: the payload is e.file (a FileUpload); .save() is async.
            async def _on_ref_upload(e):
                proj = _manual_proj()
                if proj is None:
                    ui.notify("Create a project first.", type="negative")
                    return
                fname = mm.safe_filename(e.file.name)
                dest = mm.project_dir(proj["_id"]) / "images" / f"ref_{int(time.time())}_{fname}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                await e.file.save(dest)
                MANUAL_REF["path"] = str(dest)
                ui.notify(f"Reference set: {fname}", type="positive")
            ui.upload(label="Reference image (optional — USO/Kontext/IPA use it)",
                      auto_upload=True, on_upload=_on_ref_upload) \
                .props("accept=image/* dense").style("max-width: 320px;")

            def _clear_ref():
                MANUAL_REF.pop("path", None)
                ui.notify("Reference cleared.", type="info")
            ui.button("✖ Clear ref", on_click=_clear_ref).props("flat dense")

            async def _on_shot_upload(e):
                proj = _manual_proj()
                if proj is None:
                    ui.notify("Create a project first.", type="negative")
                    return
                fname = mm.safe_filename(e.file.name)
                tmp = mm.project_dir(proj["_id"]) / "images" / f"up_{int(time.time())}_{fname}"
                tmp.parent.mkdir(parents=True, exist_ok=True)
                await e.file.save(tmp)
                shot = mm.add_shot(proj, tmp, prompt=f"(uploaded) {fname}")
                S.push(f"Uploaded {fname} → shot {shot['n']}")
                ui.notify(f"Added as shot {shot['n']}", type="positive")
                full_refresh()
            ui.upload(label="Upload image straight to board",
                      auto_upload=True, on_upload=_on_shot_upload) \
                .props("accept=image/* dense").style("max-width: 320px;")

        with ui.row().classes("gap-2"):
            def _mgen():
                bid = manual_backend_sel.value
                manual_gen_image_action(
                    manual_prompt_in.value,
                    None if bid == "(active)" else bid,
                    manual_negative_in.value, manual_seed_in.value, full_refresh)
            ui.button("✨ Generate image", on_click=_mgen).classes("rex-btn-primary")

        manual_lastgen_container = ui.row().classes("w-full gap-3")

        # ---- board ----
        ui.separator()
        ui.label("2 · Storyboard (shots in order)").classes("text-sm font-bold opacity-80")
        manual_board_container = ui.column().classes("w-full gap-2")

        # ---- music ----
        ui.separator()
        ui.label("3 · Music (optional, ACE-Step)").classes("text-sm font-bold opacity-80")
        with ui.row().classes("w-full gap-2 items-end flex-wrap"):
            manual_music_tags = ui.input(
                label="Style tags",
                placeholder="e.g. lofi, chill, mellow piano, vinyl crackle") \
                .classes("flex-1").props("outlined dark dense")
            manual_music_dur = ui.input(label="Seconds (blank = board length)") \
                .props("outlined dark dense").style("width: 210px;")
            ui.button("🎵 Generate music",
                      on_click=lambda: manual_music_action(
                          manual_music_tags.value, manual_music_dur.value, full_refresh)) \
                .props("flat color=accent")
        manual_music_container = ui.row().classes("w-full gap-2")

        # ---- assemble ----
        ui.separator()
        ui.label("4 · Assemble").classes("text-sm font-bold opacity-80")
        with ui.row().classes("gap-3 items-center flex-wrap"):
            asp_169 = ui.checkbox("16:9", value=True)
            asp_916 = ui.checkbox("9:16", value=False)
            asp_11 = ui.checkbox("1:1", value=False)

            def _massemble():
                aspects = [a for a, cb in
                           [("16:9", asp_169), ("9:16", asp_916), ("1:1", asp_11)]
                           if cb.value]
                if not aspects:
                    ui.notify("Pick at least one aspect.", type="negative")
                    return
                manual_assemble_action(aspects, full_refresh)
            ui.button("🎬 Assemble final", on_click=_massemble).classes("rex-btn-primary")
        manual_final_container = ui.row().classes("w-full gap-3")

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
    card_musicvideo.move(music_panel)
    card_horror.move(pipeline_panel)
    card_facts.move(facts_panel)
    card_prompts.move(pipeline_panel)
    card_storyboard.move(pipeline_panel)
    card_video.move(pipeline_panel)
    card_final.move(pipeline_panel)
    card_settings.move(pipeline_panel)   # Settings are story-specific → inside Story
    card_manual.move(manual_panel)
    card_models.move(models_panel)
    card_queue.move(queue_panel)

    # Recovery banners sit at the TOP of the tab that owns the job. Horror and
    # story share the pipeline panel, so both land there.
    recovery_cards["story"].move(pipeline_panel)
    recovery_cards["horror"].move(pipeline_panel)
    recovery_cards["facts"].move(facts_panel)
    recovery_cards["music"].move(music_panel)
    recovery_cards["manual"].move(manual_panel)
    card_tools.move(pipeline_panel)   # story-specific tools → inside Story

    # --- Mode-aware visibility: show ONLY the active mode's cards ---
    _story_cards = [stepper_row, card_script, card_prompts, card_storyboard,
                    card_video, card_final]

    def _apply_mode_visibility():
        # Pipeline tab IS the story pipeline now (Music + Facts are their own tabs),
        # so the story cards are always visible regardless of pipeline_mode.
        for c in _story_cards:
            c.set_visibility(True)
        card_horror.set_visibility(rs.get_pipeline_mode() == "horror_story")

    _apply_mode_visibility()

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
        windur_by_shot: dict[int, float] = {}
        if sid:
            sp = SCRIPTS_DIR / f"script_{sid}.json"
            try:
                sd = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
                for sh in sd.get("shots", []) or []:
                    _n = int(sh.get("shot_number", 0))
                    narr_by_shot[_n] = sh.get("narration", "")
                    if sh.get("win_dur"):
                        windur_by_shot[_n] = float(sh["win_dur"])
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
                        if shot_n in windur_by_shot:
                            # audio-first: each shot's locked window length (s)
                            ui.label(f"⏱ {windur_by_shot[shot_n]:.1f}s") \
                                .classes("text-xs opacity-70").style("margin-left: 8px;")
                        if approved:
                            ui.label("✅ approved").classes("text-xs opacity-80") \
                                .style("margin-left: auto;")

                    # --- Narration (the spoken line) ---
                    _af = bool((S.script or {}).get("_audio_first"))
                    if _af:
                        # Audio-first: the master voiceover is already rendered, so
                        # the text is locked to it (editing would desync captions
                        # from audio). Show it read-only with a hint.
                        ui.label("🎙️ Narration · 🔒 voiced (audio-first)") \
                            .classes("text-xs opacity-75")
                        ui.textarea(value=narr_by_shot.get(shot_n, "")) \
                            .props("outlined dark dense autogrow readonly") \
                            .classes("w-full opacity-70")
                        ui.label("Locked to the voiced master track — regenerate "
                                 "the script to change the wording.") \
                            .classes("text-xs opacity-50")
                    else:
                      # edit or rewrite freely (classic path)
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
                        ui.button("🖼️ Gen This Shot",
                                  on_click=lambda s=shot_n: regen_storyboard_shot(s, full_refresh)) \
                            .props("color=indigo dense") \
                            .tooltip("Render ONLY this shot's storyboard image (not the whole board)")

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
                # Cache-busted static URL (?v=mtime) so a regenerated frame with
                # the same filename actually refreshes in the browser.
                try:
                    rel = p.relative_to(STORYBOARDS_DIR).as_posix()
                    src = f"/sb_static/{rel}?v={p.stat().st_mtime_ns}"
                except Exception:
                    src = str(p)
                with ui.element("div").classes("rex-shot-card") \
                        .style("width: 220px;"):
                    ui.image(src).style("border-radius: 8px; width: 100%;")
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
                    ui.video(_media_url(p)).style("border-radius: 8px; width: 100%;")
                    ui.link("⬇ Download", _media_url(p)).props("download") \
                        .classes("text-xs").style("color:#7cf;")
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
                    ui.video(_media_url(p)).style("border-radius: 8px; width: 100%;")
                    ui.link("⬇ Download", _media_url(p)).props("download") \
                        .classes("text-xs").style("color:#7cf;")
                    ui.label(p.name).classes("text-xs opacity-75")

    def render_facts_reel():
        fdir = PROJECT_ROOT / "04_Outputs" / "final"
        reels = (sorted(fdir.glob("facts_*_9x16.mp4"),
                        key=lambda p: p.stat().st_mtime, reverse=True)[:1]
                 if fdir.exists() else [])
        sig = tuple((str(p), p.stat().st_mtime_ns) for p in reels) or ("none",)
        if not _changed("facts", sig):
            return
        facts_container.clear()
        with facts_container:
            if not reels:
                ui.label("_(no reel yet — enter a topic and Generate)_").classes("opacity-60")
            else:
                p = reels[0]
                with ui.element("div").classes("rex-shot-card").style("width: 300px;"):
                    ui.video(_media_url(p)).style("border-radius: 8px; width: 100%;")
                    ui.link("⬇ Download", _media_url(p)).props("download") \
                        .classes("text-xs").style("color:#7cf;")
                    ui.label(p.name).classes("text-xs opacity-75")
                # Upload-ready description (copy-paste for YouTube/IG).
                dfile = fdir / (p.stem.replace("_9x16", "") + "_description.txt")
                if dfile.exists():
                    dtext = dfile.read_text(encoding="utf-8")
                    with ui.row().classes("items-center gap-2").style("margin-top:8px;"):
                        ui.label("Upload description").classes("text-sm font-bold opacity-80")
                        ui.button("📋 Copy", on_click=lambda j=json.dumps(dtext):
                                  ui.run_javascript(f"navigator.clipboard.writeText({j})")) \
                            .props("flat dense color=accent")
                    ui.textarea(value=dtext).props("readonly outlined dense autogrow") \
                        .classes("w-full").style("max-width:640px;")

    def render_music_finals():
        fdir = PROJECT_ROOT / "04_Outputs" / "final"
        reels = (sorted(fdir.glob("song_*_9x16.mp4"),
                        key=lambda p: p.stat().st_mtime, reverse=True)[:1]
                 if fdir.exists() else [])
        sig = tuple((str(p), p.stat().st_mtime_ns) for p in reels) or ("none",)
        if not _changed("music", sig):
            return
        music_container.clear()
        with music_container:
            if not reels:
                ui.label("_(no music video yet — Generate Song, then render)_").classes("opacity-60")
            else:
                p = reels[0]
                with ui.element("div").classes("rex-shot-card").style("width: 300px;"):
                    ui.video(_media_url(p)).style("border-radius: 8px; width: 100%;")
                    ui.link("⬇ Download", _media_url(p)).props("download") \
                        .classes("text-xs").style("color:#7cf;")
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

    def render_manual():
        pid = mm.current_project_id()
        proj = mm.load_project(pid) if pid else None

        # Project picker options (refresh when the list changes).
        opts = {p: f"{p} — {name}" for p, name in mm.list_projects()}
        if _changed("manual_opts", tuple(opts.keys()) + (pid,)):
            manual_proj_select.set_options(opts, value=pid)

        # Last generated preview.
        lg = MANUAL_LAST_GEN.get("path")
        if _changed("manual_lastgen", lg):
            manual_lastgen_container.clear()
            if lg and Path(lg).exists():
                with manual_lastgen_container:
                    with ui.element("div").classes("rex-shot-card").style("width: 280px;"):
                        ui.image(_media_url(Path(lg))).style("border-radius: 8px;")
                        ui.label(f"seed {MANUAL_LAST_GEN.get('seed', '?')}") \
                            .classes("text-xs opacity-75")
                        ui.button("➕ Add to board",
                                  on_click=lambda: manual_add_last_to_board(full_refresh)) \
                            .props("flat dense color=accent").classes("w-full")

        # Board + music + finals — one signature over the whole manifest.
        if proj is None:
            if _changed("manual_board", "none"):
                manual_board_container.clear()
                with manual_board_container:
                    ui.label("_(no project — create one above)_").classes("opacity-60")
                manual_music_container.clear()
                manual_final_container.clear()
            return

        # Final files keep a stable name across re-assembles — fold their mtimes
        # into the signature or re-renders would show the stale preview.
        _fin_mt = []
        for rel in (proj.get("final") or {}).values():
            fp = mm.abs_path(proj, rel)
            _fin_mt.append(fp.stat().st_mtime_ns if fp and fp.exists() else 0)
        sig = json.dumps(proj, sort_keys=True) + str(_fin_mt)
        if not _changed("manual_board", sig):
            return

        manual_board_container.clear()
        with manual_board_container:
            if not proj["shots"]:
                ui.label("_(board empty — generate or upload a still)_").classes("opacity-60")
            for shot in proj["shots"]:
                n = shot["n"]
                with ui.element("div").classes("rex-shot-card w-full"):
                    with ui.row().classes("w-full gap-3 flex-wrap"):
                        # Still + clip previews
                        with ui.column().classes("gap-1").style("width: 240px;"):
                            img = mm.abs_path(proj, shot.get("image"))
                            if img and img.exists():
                                ui.image(_media_url(img)).style("border-radius: 8px;")
                            clip = mm.abs_path(proj, shot.get("clip"))
                            if clip and clip.exists():
                                ui.video(_media_url(clip)) \
                                    .style("border-radius: 8px; width: 100%;")
                            ui.label(f"Shot {n} · seed {shot.get('seed') or '—'}") \
                                .classes("text-xs opacity-75")
                        # Controls
                        with ui.column().classes("flex-1 gap-1").style("min-width: 280px;"):
                            mot_box = ui.textarea(label="Motion prompt (for animate)",
                                                  value=shot.get("motion_prompt") or "") \
                                .classes("w-full").props("outlined dark dense autogrow")
                            narr_box = ui.textarea(label="Narration (optional)",
                                                   value=shot.get("narration") or "") \
                                .classes("w-full").props("outlined dark dense autogrow")
                            with ui.row().classes("gap-2 items-end flex-wrap"):
                                dur_in = ui.input(label="Seconds",
                                                  value=str(shot.get("duration", 5.0))) \
                                    .props("outlined dark dense").style("width: 90px;")
                                voice_sel = ui.select(VOICE_CHOICES, value="af_heart",
                                                      label="Voice") \
                                    .props("outlined dark dense").style("width: 140px;")

                                def _save(s=n, mb=mot_box, nb=narr_box, db=dur_in):
                                    try:
                                        mm.set_shot_fields(
                                            proj, s,
                                            motion_prompt=mb.value or "",
                                            narration=nb.value or "",
                                            duration=float(db.value or 5.0))
                                        ui.notify(f"Shot {s} saved.", type="positive")
                                        full_refresh()
                                    except Exception as ex:
                                        ui.notify(f"Save failed: {ex}", type="negative")
                                ui.button("💾 Save", on_click=_save).props("flat dense")

                                def _anim(s=n, mb=mot_box, nb=narr_box, db=dur_in):
                                    # Save first so animate uses what's on screen.
                                    try:
                                        mm.set_shot_fields(
                                            proj, s, motion_prompt=mb.value or "",
                                            narration=nb.value or "",
                                            duration=float(db.value or 5.0))
                                    except Exception:
                                        pass
                                    manual_animate_action(s, full_refresh)
                                ui.button("🎥 Animate", on_click=_anim) \
                                    .props("flat dense color=accent")

                                def _tts(s=n, vs=voice_sel, nb=narr_box):
                                    try:
                                        mm.set_shot_fields(proj, s, narration=nb.value or "")
                                    except Exception:
                                        pass
                                    manual_narrate_action(s, vs.value, full_refresh)
                                ui.button("🎙️ TTS", on_click=_tts).props("flat dense")

                                na = mm.abs_path(proj, shot.get("narration_audio"))
                                if na and na.exists():
                                    ui.audio(_media_url(na)).style("max-width: 220px;")
                            with ui.row().classes("gap-1"):
                                def _up(s=n):
                                    mm.move_shot(proj, s, "up"); full_refresh()
                                def _down(s=n):
                                    mm.move_shot(proj, s, "down"); full_refresh()
                                def _rm(s=n):
                                    mm.remove_shot(proj, s)
                                    S.push(f"Removed manual shot {s}")
                                    full_refresh()
                                def _repl(s=n):
                                    if not MANUAL_LAST_GEN.get("path"):
                                        ui.notify("Nothing generated yet.", type="negative")
                                        return
                                    mm.replace_shot_image(
                                        proj, s, Path(MANUAL_LAST_GEN["path"]),
                                        prompt=MANUAL_LAST_GEN.get("prompt"),
                                        seed=MANUAL_LAST_GEN.get("seed"))
                                    S.push(f"Shot {s} still replaced with last gen")
                                    full_refresh()
                                ui.button(icon="arrow_upward", on_click=_up).props("flat dense round")
                                ui.button(icon="arrow_downward", on_click=_down).props("flat dense round")
                                ui.button("♻️ Use last gen", on_click=_repl).props("flat dense")
                                ui.button(icon="delete", on_click=_rm) \
                                    .props("flat dense round color=red")

        manual_music_container.clear()
        music = mm.abs_path(proj, (proj.get("music") or {}).get("path"))
        if music and music.exists():
            with manual_music_container:
                ui.audio(_media_url(music)).style("max-width: 420px;")
                ui.label((proj.get("music") or {}).get("tags", "")[:70]) \
                    .classes("text-xs opacity-70")

        manual_final_container.clear()
        finals = proj.get("final") or {}
        with manual_final_container:
            for aspect, rel in finals.items():
                fp = mm.abs_path(proj, rel)
                if fp and fp.exists():
                    with ui.element("div").classes("rex-shot-card").style("width: 320px;"):
                        ui.video(_media_url(fp)).style("border-radius: 8px; width: 100%;")
                        ui.link(f"⬇ Download {aspect}", _media_url(fp)) \
                            .props("download").classes("text-xs").style("color:#7cf;")

    def render_recovery():
        """Offer Resume / Discard for jobs a dead process left behind."""
        try:
            interrupted = jr.list_interrupted()
        except Exception as e:
            log.warning(f"recovery scan failed: {e}")
            return
        sig = tuple((r["job_id"], r.get("stage", "")) for r in interrupted)
        if not _changed("recovery", sig):
            return

        by_mode: dict = {}
        for r in interrupted:
            by_mode.setdefault(r.get("mode", "story"), []).append(r)

        for mode, card in recovery_cards.items():
            recs = by_mode.get(mode, [])
            card.set_visibility(bool(recs))
            body = recovery_bodies[mode]
            body.clear()
            if not recs:
                continue
            with body:
                ui.label("This job was running when the bot stopped "
                         "(shutdown, crash, or power loss).") \
                    .classes("text-xs opacity-75")
                for r in recs:
                    jid = r["job_id"]
                    with ui.element("div").classes("rex-shot-card w-full"):
                        ui.label(jr.describe(r)).classes("text-sm font-bold")
                        ui.label("Resume re-runs the stage it died in. Finished "
                                 "stages keep their rendered files.") \
                            .classes("text-xs opacity-60")
                        with ui.row().classes("gap-2").style("margin-top:6px;"):
                            ui.button("▶ Resume",
                                      on_click=lambda j=jid: resume_job_action(j, full_refresh)) \
                                .props("unelevated color=positive dense")
                            ui.button("🗑 Discard & start new",
                                      on_click=lambda j=jid: discard_job_action(j, full_refresh)) \
                                .props("flat color=red dense")

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
            _refresh_status()
            refresh_stepper()
            _fa = _FACTS_KEYS.index(S.facts_stage) if S.facts_stage in _FACTS_KEYS else -1
            _paint_stepper(facts_steps[0], facts_steps[1], _fa)
            _ma = _MUSIC_KEYS.index(S.music_stage) if S.music_stage in _MUSIC_KEYS else -1
            _paint_stepper(music_steps[0], music_steps[1], _ma)
            render_script()
            render_prompts()
            render_storyboard()
            render_video()
            render_final()
            render_facts_reel()
            render_music_finals()
            render_manual()
            render_recovery()
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
