"""
Claw Bot — Lyric Aligner (WhisperX forced alignment, music-video captions)

Music mode has no per-line timing at all: ACE-Step renders sung audio with no
alignment info, so lyric captions previously had to spread proportionally
across the WHOLE song by character count (not real sync). This gets REAL
per-line timestamps by force-aligning the ALREADY-KNOWN lyrics text against
the rendered song audio (no ASR/transcription step — we know the words, we
just need where they land in time).

WhisperX needs a torch/torchaudio/ctranslate2 stack that would clash with the
main venv (same reason Qwen3-TTS and Chatterbox live in isolated venvs), so it
lives in 03_Models/venv_whisperx and is driven over a subprocess, exactly like
tts_qwen.py / tts_chatterbox.py.

Usage:
    from modules import lyric_aligner
    spans = lyric_aligner.align_lyrics(song_audio_path, lyric_lines)
    # spans = [(text, t_start, t_end), ...] or None if alignment unavailable/failed
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

log = logging.getLogger("claw_bot.lyric_aligner")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
VENV_PY = PROJECT_ROOT / "03_Models" / "venv_whisperx" / "Scripts" / "python.exe"
CLI = _AGENT_DIR / "whisperx_align_cli.py"
TIMEOUT_SEC = 900  # a few minutes of CPU alignment for a ~2min song, generous headroom


def is_available() -> bool:
    return VENV_PY.exists() and CLI.exists()


def health_check() -> tuple[bool, str]:
    if not VENV_PY.exists():
        return False, f"whisperx venv missing: {VENV_PY}"
    if not CLI.exists():
        return False, f"whisperx_align_cli.py missing: {CLI}"
    return True, "whisperx aligner bridge ready"


def align_lyrics(audio_path: Path, lines: list) -> Optional[list]:
    """Force-align `lines` (already-known lyric text, in order) against the
    rendered song at `audio_path`. Returns [(text, t_start, t_end), ...] per
    line, or None if the aligner isn't installed / alignment fails — callers
    should fall back to the proportional char-length spread on None (this must
    never break music-video assembly)."""
    ok, msg = health_check()
    if not ok:
        log.info(f"Lyric alignment skipped: {msg}")
        return None
    lines = [ln for ln in lines if (ln or "").strip()]
    if not lines:
        return None

    with tempfile.TemporaryDirectory() as td:
        job = Path(td) / "job.json"
        out_json = Path(td) / "spans.json"
        job.write_text(json.dumps({
            "audio_path": str(Path(audio_path).resolve()),
            "lines": lines,
            "out_json": str(out_json),
        }), encoding="utf-8")

        log.info(f"Aligning {len(lines)} lyric lines against {Path(audio_path).name}...")
        try:
            r = subprocess.run(
                [str(VENV_PY), str(CLI), str(job)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            log.warning("Lyric alignment timed out; falling back to proportional spread.")
            return None
        if r.returncode != 0:
            log.warning(f"Lyric alignment failed (rc={r.returncode}): "
                       f"{(r.stderr or r.stdout or '')[-600:]}")
            return None
        if not out_json.exists():
            log.warning(f"Lyric alignment produced no output: {r.stdout[-300:]}")
            return None
        data = json.loads(out_json.read_text(encoding="utf-8"))

    spans = [(t, float(a), float(b)) for t, a, b in data.get("spans", [])]
    log.info(f"Lyric alignment ready — {len(spans)} lines.")
    return spans or None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(health_check())
