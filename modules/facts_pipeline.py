"""
Claw Bot — Facts Shorts Pipeline (orchestrator)

Renders a facts reel:
  1. Narration — kokoro voices each beat separately; the real per-beat spans give
     exact scene timing + perfectly-synced on-screen text (reuses the horror
     pipeline's span machinery).
  2. Loose mood backdrop per beat (active image backend; NO character refs — there
     is no recurring subject). Falls back to a generated gradient if the backend
     is unavailable, so the reel renders even with ComfyUI down.
  3. facts_assembly — Ken Burns + BIG centered fact text + optional music → 9x16.

Front-end agnostic (returns paths).
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import image_backend
from modules import gpu_utils
from modules import runtime_settings as rs
from modules import facts_assembly as fasm
from modules.assembly import ASPECTS
# Reuse the horror pipeline's kokoro per-beat narration + span helpers. Facts
# voices with its OWN bright/fast voice (not horror's deep narrator).
from modules.horrorstory_pipeline import (
    _kokoro_narrate, _load_spans, _spans_path, _scene_durations,
)


def _voice_facts(narrations: list, out_path: Path):
    """Kokoro narration in the FACTS voice (bright + slightly fast = energetic).
    Returns (wav, spans)."""
    return _kokoro_narrate(narrations, out_path,
                           voice=rs.get_facts_voice(),
                           speed=rs.get_facts_voice_speed())

log = logging.getLogger("claw_bot.facts_pipeline")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
STILLS_DIR = PROJECT_ROOT / "04_Outputs" / "storyboards"
NARRATION_DIR = PROJECT_ROOT / "04_Outputs" / "audio"

# palette cycled for gradient fallbacks (dark, punchy, readable under white text)
_GRADIENTS = [
    ("0x1a1a2e", "0x16213e"), ("0x2b1055", "0x7597de"), ("0x0f2027", "0x2c5364"),
    ("0x232526", "0x414345"), ("0x141e30", "0x243b55"), ("0x3a1c71", "0xd76d77"),
]


def _gradient_bg(idx: int, w: int, h: int, out_path: Path) -> Path:
    """Generate one gradient still (pure ffmpeg) — image-independent fallback."""
    c0, c1 = _GRADIENTS[idx % len(_GRADIENTS)]
    cmd = [
        str(fasm.FFMPEG_EXE), "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"gradients=s={w}x{h}:c0={c0}:c1={c1}:x0=0:y0=0:x1={w}:y1={h}",
        "-frames:v", "1", str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not out_path.exists():
        # last-ditch: flat colour
        subprocess.run([str(fasm.FFMPEG_EXE), "-y", "-loglevel", "error",
                        "-f", "lavfi", "-i", f"color=c={c0}:s={w}x{h}",
                        "-frames:v", "1", str(out_path)], timeout=120)
    return out_path


def narrate_facts(story: dict, progress_cb: Optional[Callable[[str], None]] = None) -> Path:
    """Voice the whole reel (kokoro per-beat) + write the spans sidecar. Returns wav."""
    facts_id = story.get("facts_id") or story.get("_id")
    beats = story.get("beats", [])
    if not beats:
        raise ValueError("Facts reel has no beats.")
    gpu_utils.ensure_vram_free()
    NARRATION_DIR.mkdir(parents=True, exist_ok=True)
    out = NARRATION_DIR / f"facts_{facts_id}.wav"
    out, spans = _voice_facts([b.get("narration", "") for b in beats], out)
    if spans:
        _spans_path(out).write_text(__import__("json").dumps(spans), encoding="utf-8")
    if progress_cb:
        progress_cb(f"narration ready (kokoro:{rs.get_facts_voice()})")
    return out


def render_facts(
    story: dict,
    progress_cb: Optional[Callable[[str], None]] = None,
    narration_path: Optional[Path] = None,
    aspect: str = "9x16",
    music_path: Optional[Path] = None,
) -> dict:
    """Full render: kokoro narration -> per-beat spans -> mood backdrops -> 9x16
    Ken Burns + big centered fact text."""
    def _p(msg: str):
        log.info(msg)
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    facts_id = story.get("facts_id") or story.get("_id")
    beats = story.get("beats", [])
    if not beats:
        raise ValueError("Facts reel has no beats.")

    import json as _json
    gpu_utils.ensure_vram_free()
    NARRATION_DIR.mkdir(parents=True, exist_ok=True)
    spans = None
    if narration_path and Path(narration_path).exists():
        narration_path = Path(narration_path)
        spans = _load_spans(narration_path)
        _p("🎙️ using pre-rendered narration" + (" (+spans)" if spans else ""))
    else:
        narration_path = NARRATION_DIR / f"facts_{facts_id}.wav"
        _p(f"🎙️ voicing {len(beats)} beats (kokoro:{rs.get_facts_voice()})...")
        narration_path, spans = _voice_facts(
            [b.get("narration", "") for b in beats], narration_path)
        if spans:
            _spans_path(narration_path).write_text(_json.dumps(spans), encoding="utf-8")
    total_dur = fasm._probe_duration(narration_path)
    _p(f"🎙️ narration ready: {total_dur:.1f}s")

    # Per-beat windows from the real voice spans (exact) — text flips on the voice.
    if spans and len(spans) == len(beats):
        durations = _scene_durations(spans, total_dur)
        _p(f"🎯 {len(durations)} windows from real per-beat spans")
    else:
        wsum = sum(max(1, len(b.get("narration", "").split())) for b in beats) or len(beats)
        durations = [round(max(1, len(b.get("narration", "").split())) / wsum * total_dur, 3)
                     for b in beats]

    # ---- backdrops: active backend (loose, no refs) or gradient fallback ----
    w, h = ASPECTS.get(aspect, ASPECTS["9x16"])
    out_dir = STILLS_DIR / f"facts_{facts_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ar = aspect.replace("x", ":")

    backend = None
    try:
        gpu_utils.free_comfyui_vram()
        backend = image_backend.get_active_backend()
        ok, _msg = backend.health_check()
        if not ok:
            _p(f"image backend unhealthy ({_msg}); using gradient backdrops.")
            backend = None
    except Exception as e:
        _p(f"image backend unavailable ({e}); using gradient backdrops.")
        backend = None

    _p(f"🖼️ building {len(beats)} backdrops ({'backend' if backend else 'gradient'})...")
    backgrounds: list[Path] = []
    for i, b in enumerate(beats):
        dst = out_dir / f"bg_{i:04d}.png"
        if backend is not None:
            try:
                p = backend.generate(
                    prompt=b.get("image_prompt", "abstract cinematic background, no text"),
                    negative_prompt="text, words, letters, captions, subtitles, watermark, logo, "
                                    "blurry, low quality, distorted",
                    aspect_ratio=ar,
                    output_filename=f"facts_{facts_id}/bg_{i:04d}.png",
                )
                backgrounds.append(Path(p)); continue
            except Exception as e:
                _p(f"backdrop {i+1} backend failed ({e}); gradient.")
        backgrounds.append(_gradient_bg(i, w, h, dst))

    _p("🎬 assembling facts reel...")
    gpu_utils.free_comfyui_vram()
    out = fasm.assemble_facts(story, narration_path, durations, backgrounds,
                              aspect=aspect, music_path=music_path,
                              progress_cb=progress_cb)
    out["narration_audio"] = narration_path
    _p("✅ facts reel complete")
    return out
