"""
Claw Bot — Chatterbox TTS bridge (expressive, voice-cloning, offline)

Chatterbox (Resemble AI, MIT) gives expressive speech + voice cloning + an
`exaggeration` emotion knob — what VoxCPM's flat-reference clone lacked. It pins
torch==2.6, which clashes with our main venv's torch 2.11, so it lives in an
ISOLATED venv (03_Models/venv_chatterbox) and we call it over a subprocess.

This adapter (main venv) builds a job, runs chatterbox_cli.py with the isolated
venv's python, and reads back (master_wav, spans) — the same shape as
VoxCPM.synthesize_segments, so audio_first_pipeline can use it interchangeably.
"""

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

_AGENT_DIR = Path(__file__).parent.parent.resolve()
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

log = logging.getLogger("claw_bot.tts_chatterbox")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
VENV_PY = PROJECT_ROOT / "03_Models" / "venv_chatterbox" / "Scripts" / "python.exe"
CLI = _AGENT_DIR / "chatterbox_cli.py"
HF_CACHE = PROJECT_ROOT / "03_Models" / "hf_cache"
AUDIO_DIR = PROJECT_ROOT / "04_Outputs" / "audio"

# Emotion intensity (0.5 neutral .. ~0.9 strong). cfg_weight lower = more
# expressive pacing. Tunable later via runtime_settings.
DEFAULT_EXAGGERATION = 0.6
DEFAULT_CFG_WEIGHT = 0.4
GROUP_GAP_SEC = 0.34
# Generous: model load (~cold) + per-group gen on GPU, whole script in one call.
TIMEOUT_SEC = 1800


def is_available() -> bool:
    """True if the isolated Chatterbox venv + CLI are present."""
    return VENV_PY.exists() and CLI.exists()


def health_check() -> tuple[bool, str]:
    if not VENV_PY.exists():
        return False, f"chatterbox venv missing: {VENV_PY}"
    if not CLI.exists():
        return False, f"chatterbox_cli.py missing: {CLI}"
    return True, "chatterbox bridge ready"


def synthesize_segments(
    groups: list,
    output_path: Optional[Path] = None,
    gap_sec: float = GROUP_GAP_SEC,
    exaggeration: float = DEFAULT_EXAGGERATION,
    cfg_weight: float = DEFAULT_CFG_WEIGHT,
) -> tuple[Path, list[tuple[str, float, float]]]:
    """Voice each breath-group with Chatterbox (cloning its per-speaker reference
    + the emotion knob), stitched with silence gaps. `groups` items are
    `(text, ref_wav, ref_text)` triples (ref_text ignored — Chatterbox clones
    from audio only) or plain strings. Returns (master_wav, spans)."""
    ok, msg = health_check()
    if not ok:
        raise RuntimeError(msg)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        from datetime import datetime
        output_path = AUDIO_DIR / f"master_{datetime.now():%Y%m%d_%H%M%S}.wav"
    output_path = Path(output_path)

    spec = []
    for g in groups:
        if isinstance(g, (tuple, list)):
            text = (g[0] or "").strip()
            ref = g[1] if len(g) > 1 else None
        else:
            text, ref = (g or "").strip(), None
        if not text:
            continue
        spec.append({
            "text": text,
            "ref_wav": str(ref) if ref else None,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
        })
    if not spec:
        raise ValueError("synthesize_segments() got no non-empty groups.")

    data = _run(spec, gap_sec, output_path)
    spans = [(t, float(a), float(b)) for t, a, b in data["spans"]]
    log.info(f"Chatterbox wrote {output_path.name} — {len(spans)} groups.")
    return output_path, spans


def _run(spec: list, gap_sec: float, out_wav: Path,
         segments_dir: Optional[Path] = None) -> dict:
    """Drive chatterbox_cli in the isolated venv; return the parsed spans JSON."""
    import os
    with tempfile.TemporaryDirectory() as td:
        job_path = Path(td) / "job.json"
        spans_path = Path(td) / "spans.json"
        job_path.write_text(json.dumps({
            "groups": spec, "gap_sec": gap_sec,
            "out_wav": str(out_wav), "spans_out": str(spans_path),
            "segments_dir": str(segments_dir) if segments_dir else None,
        }), encoding="utf-8")

        full_env = {
            **os.environ,
            "HF_HOME": str(HF_CACHE),
            "HF_HUB_CACHE": str(HF_CACHE / "hub"),
            "PYTHONIOENCODING": "utf-8",
        }
        r = subprocess.run(
            [str(VENV_PY), str(CLI), str(job_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=TIMEOUT_SEC, env=full_env,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"chatterbox_cli failed (rc={r.returncode}): "
                f"{(r.stderr or r.stdout or '')[-600:]}"
            )
        if not spans_path.exists():
            raise RuntimeError(f"chatterbox_cli produced no spans: {r.stdout[-300:]}")
        return json.loads(spans_path.read_text(encoding="utf-8"))


def synthesize_each(
    texts: list[str],
    out_dir: Path,
    ref_wav: Optional[Path] = None,
    exaggeration: float = DEFAULT_EXAGGERATION,
    cfg_weight: float = DEFAULT_CFG_WEIGHT,
) -> list[Path]:
    """One WAV per line, all cloned from `ref_wav`, in a single model load.

    Mirrors tts_qwen.synthesize_each so the facts mascot can swap engines without
    the caller caring. The facts pipeline cuts one video shot per beat, so it needs
    the beats as separate files — not a master it has to slice.
    """
    ok, msg = health_check()
    if not ok:
        raise RuntimeError(msg)

    spec = [{
        "text": t.strip(),
        "ref_wav": str(ref_wav) if ref_wav else None,
        "exaggeration": exaggeration,
        "cfg_weight": cfg_weight,
    } for t in texts if (t or "").strip()]
    if not spec:
        raise ValueError("synthesize_each() got no non-empty texts.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Chatterbox voicing {len(spec)} beats "
             f"(clone={'yes' if ref_wav else 'no'}, exaggeration={exaggeration})...")
    data = _run(spec, GROUP_GAP_SEC, out_dir / "_master.wav", segments_dir=out_dir)

    files = [Path(f) for f in data.get("files", [])]
    if len(files) != len(spec):
        raise RuntimeError(
            f"chatterbox returned {len(files)} beat wavs, expected {len(spec)}"
        )
    return files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(health_check())
