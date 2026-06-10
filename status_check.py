"""
Claw Bot — Project Status Scanner

Generates a handover-ready PROJECT_STATUS.md describing what's built, what's
running, and what's pending. Designed to be pasted into a new Claude chat so
there's zero ambiguity about current state.

Usage:
  python status_check.py

Output:
  ./PROJECT_STATUS.md  (overwritten each run)
"""

import json
import logging
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Optional: requests for service checks
try:
    import requests
except ImportError:
    requests = None

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODULES_DIR = PROJECT_ROOT / "02_Agent" / "modules"
OUTPUTS_DIR = PROJECT_ROOT / "04_Outputs"
CONFIG_DIR = PROJECT_ROOT / "05_Config"
COMFY_DIR = PROJECT_ROOT / "01_ComfyUI" / "ComfyUI_windows_portable" / "ComfyUI"
LOG_FILE = PROJECT_ROOT / "PROJECT_STATUS.md"


# ==============================================================================
# COMPONENT REGISTRY — single source of truth for what features exist
# ==============================================================================
# Each entry: (display_name, module_filename, "what it does")
COMPONENTS = [
    # Phase 3 — Scripting
    ("Script generator (LLM)", "script_generator.py", "Qwen2.5 14B turns themes into 5-10 shot scripts"),
    ("Prompt polisher", "prompt_polisher.py", "Rewrites visual prompts for Z-Image, enforces clothing anchors"),
    # Phase 4 — Storyboard
    ("Storyboard generator", "storyboard_generator.py", "One image per shot via active image backend"),
    # Phase 5 — Video
    ("Clip generator (TTS + video mux)", "clip_generator.py", "Each shot: TTS → video → ffmpeg mux into MP4 with audio"),
    ("TTS engine (Kokoro)", "tts_engine.py", "Local Kokoro-82M, multiple voices, voice override via Discord"),
    ("Video workflow", "video_workflow.py", "Per-storyboard video generation + Discord approval"),
    # Phase 5.5 — Upscale
    ("Upscaler (Real-ESRGAN)", "upscaler.py", "Per-clip 4x upscale via ComfyUI, auto-triggered on video approval"),
    # Phase 6 — Assembly + publish
    ("Assembly (concat clips → one MP4)", "assembly.py", "Stitch upscaled clips into final 30s short"),
    ("Background music", "music_generator.py", "MusicGen → mix under narration"),
    ("YouTube uploader", "youtube_uploader.py", "OAuth + upload + metadata"),
    # Cross-cutting
    ("Discord bot (claw_bot.py)", "../claw_bot.py", "Main controller — all commands route through here"),
    ("Image backend interface", "image_backend.py", "Pluggable image model loader"),
    ("Model registry", "model_registry.py", "Reads models.json, manages active backends"),
    ("GPU utils (VRAM cleanup)", "gpu_utils.py", "Frees VRAM between jobs"),
    ("Runtime settings", "runtime_settings.py", "Per-session overrides (voice, style, aspect ratio, etc.)"),
    ("Health monitor (dashboard)", "health_monitor.py", "Live #status channel updates every 30s"),
    ("Generation meta logger", "generation_meta.py", "Records every job's duration / VRAM / settings"),
    ("Agent router (chat NLU)", "agent_router.py", "Routes natural-language Discord messages to tools"),
    ("Conversational agent", "agent.py", "Qwen-driven chat in #claw-bot"),
]

# Image backends (subfolder)
IMAGE_BACKENDS = [
    ("Z-Image Turbo / Base", "image_backends/comfyui_zimage_base.py"),
    ("Flux.2 Klein", "image_backends/comfyui_flux2.py"),
    ("Flux Kontext (ref-image consistency)", "image_backends/comfyui_kontext_base.py"),
    ("SDXL Turbo", "image_backends/comfyui_sdxl_turbo.py"),
]

VIDEO_BACKENDS = [
    ("Wan 2.2 14B I2V", "video_backends/comfyui_wan22_14B.py"),
    ("Wan 2.2 5B", "video_backends/comfyui_wan22.py"),
    ("LTX-2 19B", "video_backends/comfyui_ltx_video.py"),
]


# ==============================================================================
# CHECKS
# ==============================================================================

def check_module(rel_path: str) -> tuple[bool, int]:
    """Returns (exists, line_count)."""
    p = MODULES_DIR / rel_path
    if not p.exists():
        return False, 0
    try:
        return True, len(p.read_text(encoding="utf-8", errors="ignore").splitlines())
    except Exception:
        return True, 0


def check_service(url: str, timeout: float = 3.0) -> tuple[bool, str]:
    """HTTP HEAD/GET check. Returns (is_up, status_str)."""
    if requests is None:
        return False, "requests lib not available"
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code < 500, f"HTTP {r.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "connection refused"
    except requests.exceptions.Timeout:
        return False, "timeout"
    except Exception as e:
        return False, f"error: {e}"


def check_ollama_models() -> list[str]:
    """List currently-installed Ollama models. Empty list if Ollama is down."""
    if requests is None:
        return []
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        if r.status_code != 200:
            return []
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def find_latest(folder: Path, pattern: str = "*") -> Path | None:
    if not folder.exists():
        return None
    items = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return items[0] if items else None


def gpu_info() -> str:
    """Use nvidia-smi to report GPU + driver + VRAM."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "nvidia-smi not available"


def get_active_backends() -> dict:
    cfg = CONFIG_DIR / "models.json"
    if not cfg.exists():
        return {"_error": "models.json not found"}
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
        return {
            "llm": data.get("llm_backend", {}).get("active", "?"),
            "image": data.get("image_backend", {}).get("active", "?"),
            "video": data.get("video_backend", {}).get("active", "?"),
        }
    except Exception as e:
        return {"_error": str(e)}


def installed_styles() -> list[str]:
    p = CONFIG_DIR / "styles.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return list(data.get("available", {}).keys())
    except Exception:
        return []


# ==============================================================================
# COUNT RECENT OUTPUTS
# ==============================================================================

def recent_output_summary() -> dict:
    """Look at 04_Outputs/ to confirm features actually ran recently."""
    summary = {}

    # Scripts
    scripts_dir = OUTPUTS_DIR / "scripts"
    if scripts_dir.exists():
        scripts = list(scripts_dir.glob("script_*.json"))
        summary["scripts_total"] = len(scripts)
        latest = find_latest(scripts_dir, "script_*.json")
        if latest:
            summary["scripts_latest"] = latest.stem
            try:
                d = json.loads(latest.read_text(encoding="utf-8"))
                summary["scripts_latest_polished"] = d.get("_polished", False)
                summary["scripts_latest_shots"] = len(d.get("shots", []))
            except Exception:
                pass

    # Storyboards
    sb_dir = OUTPUTS_DIR / "storyboards"
    if sb_dir.exists():
        manifests = list(sb_dir.glob("*/storyboard.json"))
        summary["storyboards_total"] = len(manifests)
        latest = find_latest(sb_dir, "*/storyboard.json")
        if latest:
            summary["storyboards_latest"] = latest.parent.name
            try:
                d = json.loads(latest.read_text(encoding="utf-8"))
                summary["storyboards_latest_frames"] = d.get("total_frames", 0)
                summary["storyboards_latest_success"] = d.get("success", False)
                # Sub-reports
                if (latest.parent / "qa_report.json").exists():
                    summary["qa_report_present"] = True
                if (latest.parent / "continuity_report.json").exists():
                    summary["continuity_report_present"] = True
            except Exception:
                pass

    # Clips (videos)
    clips_dir = OUTPUTS_DIR / "clips"
    if clips_dir.exists():
        clips = list(clips_dir.glob("clip_*.mp4"))
        summary["clips_total"] = len(clips)
        latest = find_latest(clips_dir, "clip_*.mp4")
        if latest:
            summary["clips_latest"] = latest.name
            summary["clips_latest_size_mb"] = round(latest.stat().st_size / 1024 / 1024, 1)

    # Approved storyboards / videos
    for kind in ("approved_storyboards", "approved_videos"):
        d = OUTPUTS_DIR / kind
        if d.exists():
            summary[f"{kind}_count"] = len(list(d.glob("*.approved")))

    return summary


# ==============================================================================
# MAIN — assemble the markdown report
# ==============================================================================

def build_report() -> str:
    now = datetime.now(timezone.utc).astimezone()
    lines = []
    lines.append("# Claw Bot — Project Status")
    lines.append(f"_Generated {now.strftime('%Y-%m-%d %H:%M:%S %Z')} on {platform.node()}_")
    lines.append("")
    lines.append("This file is auto-generated by `status_check.py`. Paste into a new chat for full context.")
    lines.append("")

    # System info
    lines.append("## System")
    lines.append(f"- **OS:** {platform.system()} {platform.release()}")
    lines.append(f"- **Python:** {sys.version.split()[0]}")
    lines.append(f"- **Project root:** `{PROJECT_ROOT}`")
    lines.append(f"- **GPU:** {gpu_info()}")
    lines.append("")

    # Services
    lines.append("## Services")
    ollama_up, ollama_status = check_service("http://127.0.0.1:11434/api/tags")
    comfy_up, comfy_status = check_service("http://127.0.0.1:8188/system_stats")
    lines.append(f"- **Ollama** (port 11434): {'✅' if ollama_up else '❌'} ({ollama_status})")
    lines.append(f"- **ComfyUI** (port 8188): {'✅' if comfy_up else '❌'} ({comfy_status})")
    if ollama_up:
        models = check_ollama_models()
        if models:
            lines.append(f"  - Installed Ollama models: `{', '.join(models)}`")
    lines.append("")

    # Active backends
    lines.append("## Active backends (from `models.json`)")
    ab = get_active_backends()
    if "_error" in ab:
        lines.append(f"- ❌ {ab['_error']}")
    else:
        lines.append(f"- **LLM:** `{ab['llm']}`")
        lines.append(f"- **Image:** `{ab['image']}`")
        lines.append(f"- **Video:** `{ab['video']}`")
    lines.append("")

    # Components
    lines.append("## Component status (✅ built / ❌ not built)")
    lines.append("")
    lines.append("| Component | Status | Module | Lines | Notes |")
    lines.append("|---|---|---|---:|---|")
    for name, mod, note in COMPONENTS:
        ok, n = check_module(mod)
        mark = "✅" if ok else "❌"
        n_str = str(n) if ok else "—"
        lines.append(f"| {name} | {mark} | `{mod}` | {n_str} | {note} |")
    lines.append("")

    # Image backends
    lines.append("### Image backends")
    lines.append("")
    lines.append("| Backend | Status |")
    lines.append("|---|---|")
    for name, mod in IMAGE_BACKENDS:
        ok, _ = check_module(mod)
        lines.append(f"| {name} | {'✅' if ok else '❌'} |")
    lines.append("")

    # Video backends
    lines.append("### Video backends")
    lines.append("")
    lines.append("| Backend | Status |")
    lines.append("|---|---|")
    for name, mod in VIDEO_BACKENDS:
        ok, _ = check_module(mod)
        lines.append(f"| {name} | {'✅' if ok else '❌'} |")
    lines.append("")

    # Styles
    styles = installed_styles()
    if styles:
        lines.append("## Styles available")
        lines.append(", ".join(f"`{s}`" for s in styles))
        lines.append("")

    # Recent outputs
    lines.append("## Recent output activity")
    summary = recent_output_summary()
    if not summary:
        lines.append("_No output activity found._")
    else:
        for k, v in summary.items():
            lines.append(f"- **{k}**: `{v}`")
    lines.append("")

    # Phase summary — what's left
    lines.append("## Phase status (high level)")
    lines.append("")
    phases = [
        ("Phase 1 — Environment", True, "ComfyUI + Python venv + Ollama installed"),
        ("Phase 2 — Discord bot", True, "Bot online, command system, channels wired"),
        ("Phase 3 — Script generation", True, "Qwen2.5 14B → polished prompts"),
        ("Phase 4 — Storyboard", True, "Z-Image + QA + Continuity + Kontext (optional)"),
        ("Phase 5 — Video clips with audio (TTS)", True, "Kokoro TTS → mux → MP4 per shot"),
        ("Phase 5.5 — Upscale", True, "Real-ESRGAN per-clip, auto on approval"),
        ("Phase 6A — Final assembly (concat clips)", False, "Need: stitch all clips into one 30s MP4"),
        ("Phase 6B — Background music", False, "Need: MusicGen or library track per story"),
        ("Phase 6C — YouTube upload", False, "Need: OAuth + upload + metadata"),
        ("Phase 7 — Polish / loops", False, "Daily auto-run, error recovery, analytics"),
    ]
    lines.append("| Phase | Status | Notes |")
    lines.append("|---|---|---|")
    for name, done, note in phases:
        mark = "✅" if done else "❌"
        lines.append(f"| {name} | {mark} | {note} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("**Regenerate this file:** `python 02_Agent/status_check.py`")
    return "\n".join(lines)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    report = build_report()
    LOG_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n\nReport saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
