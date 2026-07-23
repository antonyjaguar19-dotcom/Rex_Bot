"""LatentSync lip-sync bridge — drives the standalone LatentSync env over subprocess.

Same shape as `tts_chatterbox` → `chatterbox_cli`: LatentSync needs torch cu128 +
mediapipe + insightface + diffusers, a dependency set that cannot live in the bot's venv
without breaking it, so it lives in its OWN venv and we shell out to it.

WHAT IT IS FOR. A talking mascot in one shot needs three things a single local model will
not give together: body motion, the mascot's identity, and lips synced to the voice. The
lesson pipeline already makes the first two — Qwen-Edit draws the mascot, Wan animates the
body — and this adds the third as a SECOND PASS over the finished Wan clip: LatentSync
detects the face, and repaints the mouth to the audio.

THE HARD LIMIT, proven on the GPU (see IMAGE_FEEDBACK / agent memory). LatentSync runs a
HUMAN face detector (insightface/SCRFD) first. On a human-faced mascot (e.g. nakshu) it
locks on; on a NON-human mascot (a creature like RexJaw) it finds NOTHING and cannot sync —
0 faces on every reference image. So this is gated by `face_detectable()`: a creature
mascot silently keeps its silent Wan clip (the lesson then voices over it, as before), and
only a human-faced mascot gets the lip-sync pass. This is not tunable — it is what the
model is.

Alignment is free: the lesson already voices per beat (`beat_XX.wav`) and sizes each Wan
clip to that beat's duration, so `lipsync(clip, beat_wav)` matches with no looping, and the
concatenated narration the assembler lays over the whole video still lines up (the mouth and
the narration come from the same wav).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import logging

log = logging.getLogger("clawbot.latentsync")

# --- where the standalone install lives -------------------------------------------------
# Production home: 03_Models (beside venv_chatterbox), OUT of the experiment tree so a sweep
# of 08Experiment cannot break lessons. The venv was recreated here (a Python venv bakes
# absolute paths and cannot simply be moved); the repo + checkpoints were moved across.
_ROOT = Path(r"E:/Rexjaw_VFX/03_Models")
LATENTSYNC_DIR = _ROOT / "LatentSync"
ENV_PY = _ROOT / "venv_latentsync" / "Scripts" / "python.exe"
UNET_CKPT = LATENTSYNC_DIR / "checkpoints" / "latentsync_unet.pt"
UNET_CONFIG = LATENTSYNC_DIR / "configs" / "unet" / "stage2_512.yaml"

_FFMPEG_BIN = Path(r"E:/Rexjaw_VFX/00_Tools/ffmpeg/bin")

# LatentSync inference knobs — the values proven in the experiment.
INFERENCE_STEPS = 20
GUIDANCE_SCALE = 1.5

# Which lesson framings are safe to lip-sync. Measured the hard way: an ESTABLISHING wide
# shot (small figure that MOVES across the frame while Wan animates it) makes LatentSync lose
# the mouth — it "slides to the neck and back". A medium/closeup presenter holds the face
# steady and syncs clean. Face SIZE alone does not separate them (a working medium still and a
# sliding establishing still both measured ~0.22 of frame) — the FRAMING does, because it is
# about the face being large AND stable, not just large. So lip-sync runs on medium/closeup
# beats and leaves wide/establishing beats as silent clips, voiced over.
LIPSYNC_FRAMINGS = ("medium", "closeup")


def _env() -> dict:
    e = dict(os.environ)
    if _FFMPEG_BIN.exists():
        e["PATH"] = str(_FFMPEG_BIN) + os.pathsep + e.get("PATH", "")
    return e


def is_available() -> Tuple[bool, str]:
    """Is the standalone LatentSync install present and runnable?"""
    for p, what in ((ENV_PY, "env python"),
                    (LATENTSYNC_DIR, "repo"),
                    (UNET_CKPT, "unet checkpoint"),
                    (UNET_CONFIG, "unet config")):
        if not p.exists():
            return False, f"LatentSync {what} missing ({p})"
    return True, "ready"


def face_fracs(images: list, timeout: int = 300) -> list:
    """For each image, the largest face's size as a fraction of the frame — max(w/W, h/H),
    or 0.0 if no face (or on any failure: unknown is treated as no-sync, never a crash).

    ONE subprocess for the whole batch: insightface loads once (~13s), not per still. Runs in
    the LatentSync env because insightface only lives there. This is both gates in one number:
    0.0 == not a (human) face at all (a creature mascot), and a small-but-nonzero value == a
    face too small to lip-sync (a wide/establishing shot — see MIN_LIPSYNC_FACE_FRAC)."""
    ok, _ = is_available()
    if not ok:
        return [0.0] * len(images)
    code = (
        "import sys, json, cv2\n"
        "from insightface.app import FaceAnalysis\n"
        "app=FaceAnalysis(providers=['CPUExecutionProvider'])\n"
        "app.prepare(ctx_id=-1, det_size=(640,640))\n"
        "out=[]\n"
        "for p in sys.argv[1:]:\n"
        "    img=cv2.imread(p)\n"
        "    if img is None: out.append(0.0); continue\n"
        "    H,W=img.shape[:2]; best=0.0\n"
        "    for f in app.get(img):\n"
        "        x1,y1,x2,y2=f.bbox\n"
        "        best=max(best, max((x2-x1)/W, (y2-y1)/H))\n"
        "    out.append(round(float(best),3))\n"
        "print('FRACS', json.dumps(out))\n"
    )
    try:
        r = subprocess.run([str(ENV_PY), "-c", code, *[str(p) for p in images]],
                           cwd=str(LATENTSYNC_DIR), env=_env(),
                           capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        log.warning(f"face_fracs probe failed: {e}")
        return [0.0] * len(images)
    for line in (r.stdout or "").splitlines():
        if line.startswith("FRACS"):
            try:
                import json as _j
                vals = _j.loads(line[5:].strip())
                if len(vals) == len(images):
                    return [float(v) for v in vals]
            except Exception:
                break
    log.warning(f"face_fracs: no verdict (stderr tail: {(r.stderr or '')[-300:]})")
    return [0.0] * len(images)


def face_detectable(image: Path, timeout: int = 180) -> bool:
    """Is there any (human) face at all? Presence only — size is judged separately with
    MIN_LIPSYNC_FACE_FRAC. A creature mascot returns False. Thin wrapper over face_fracs."""
    return face_fracs([image], timeout=timeout)[0] > 0.0


def lipsync(video: Path, audio: Path, out: Path,
            steps: int = INFERENCE_STEPS, guidance: float = GUIDANCE_SCALE,
            timeout: int = 1800) -> Optional[Path]:
    """Repaint the mouth of `video` to `audio`. Returns `out` on success, else None.

    The caller must have freed the GPU first (LatentSync needs several GB; a resident Wan or
    Qwen model will OOM it). Never raises — a lip-sync failure must not lose the finished Wan
    clip; the caller keeps the silent clip and voices over it.
    """
    ok, why = is_available()
    if not ok:
        log.warning(f"lipsync skipped: {why}")
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(ENV_PY), "-m", "scripts.inference",
           "--unet_config_path", str(UNET_CONFIG),
           "--inference_ckpt_path", str(UNET_CKPT),
           "--inference_steps", str(steps),
           "--guidance_scale", str(guidance),
           "--enable_deepcache",
           "--video_path", str(video),
           "--audio_path", str(audio),
           "--video_out_path", str(out)]
    try:
        r = subprocess.run(cmd, cwd=str(LATENTSYNC_DIR), env=_env(),
                           capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        log.warning(f"lipsync subprocess failed: {e}")
        return None
    if out.exists() and out.stat().st_size > 0:
        return out
    log.warning(f"lipsync produced no file (rc={r.returncode}); "
                f"stderr tail: {(r.stderr or '')[-500:]}")
    return None
