"""
Claw Bot — Lyric Aligner (WhisperX, music-video caption sync)

Music mode has no per-line timing: ACE-Step renders sung audio with no
alignment, AND the vocals do NOT track the lyrics 1:1 across the whole file —
there are instrumental intros/outros/breaks. Spreading the known lyrics across
[0, song_dur] therefore put captions on the intro music (out of sync).

Fix: transcribe the ACTUAL vocals with WhisperX (ASR + voice-activity
detection). VAD only emits words where singing happens, so we get the REAL
singing regions with instrumental gaps excluded. This module returns those
regions as "vocal windows"; the caller places the clean known lyrics inside
them (subtitles.distribute_lines_over_windows), leaving instrumental empty.

WhisperX needs a torch/ctranslate2 stack that clashes with the main venv (same
reason Qwen3-TTS / Chatterbox live in isolated venvs), so it runs in
03_Models/venv_whisperx over a subprocess, like tts_qwen.py.

Usage:
    from modules import lyric_aligner
    windows = lyric_aligner.get_vocal_windows(song_audio_path)
    # windows = [(t_start, t_end), ...] real singing spans, or None if
    # unavailable / no vocals detected -> caller falls back to a proportional
    # spread across the whole song.
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
TIMEOUT_SEC = 900  # ASR+align of a ~2min song on CPU, generous headroom
# A silence between sung words longer than this = an instrumental break, so we
# start a new vocal window (captions go empty across the gap).
DEFAULT_GAP_SEC = 1.2


def is_available() -> bool:
    return VENV_PY.exists() and CLI.exists()


def health_check() -> tuple[bool, str]:
    if not VENV_PY.exists():
        return False, f"whisperx venv missing: {VENV_PY}"
    if not CLI.exists():
        return False, f"whisperx_align_cli.py missing: {CLI}"
    return True, "whisperx aligner bridge ready"


def _words_from_audio(audio_path: Path) -> Optional[list]:
    """Run the CLI → list of [word, t_start, t_end] for the ACTUAL sung vocals,
    or None on any failure (caller falls back)."""
    ok, msg = health_check()
    if not ok:
        log.info(f"Lyric alignment skipped: {msg}")
        return None
    with tempfile.TemporaryDirectory() as td:
        job = Path(td) / "job.json"
        out_json = Path(td) / "words.json"
        job.write_text(json.dumps({
            "audio_path": str(Path(audio_path).resolve()),
            "out_json": str(out_json),
        }), encoding="utf-8")
        log.info(f"Transcribing vocals in {Path(audio_path).name}...")
        try:
            r = subprocess.run(
                [str(VENV_PY), str(CLI), str(job)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            log.warning("Vocal transcription timed out; falling back.")
            return None
        if r.returncode != 0:
            log.warning(f"Vocal transcription failed (rc={r.returncode}): "
                       f"{(r.stderr or r.stdout or '')[-600:]}")
            return None
        if not out_json.exists():
            log.warning(f"Vocal transcription produced no output: {r.stdout[-300:]}")
            return None
        data = json.loads(out_json.read_text(encoding="utf-8"))
    words = data.get("words") or []
    return words or None


def group_windows(words: list, gap_sec: float = DEFAULT_GAP_SEC) -> list:
    """Group word spans into singing windows: consecutive words separated by a
    silence > gap_sec start a new window. Returns [(t_start, t_end), ...]."""
    windows = []
    cur_start = cur_end = None
    for _w, s, e in words:
        s, e = float(s), float(e)
        if cur_start is None:
            cur_start, cur_end = s, e
        elif s - cur_end > gap_sec:
            windows.append((round(cur_start, 3), round(cur_end, 3)))
            cur_start, cur_end = s, e
        else:
            cur_end = max(cur_end, e)
    if cur_start is not None:
        windows.append((round(cur_start, 3), round(cur_end, 3)))
    return windows


def get_vocal_windows(audio_path: Path, gap_sec: float = DEFAULT_GAP_SEC) -> Optional[list]:
    """Real singing spans [(t_start, t_end), ...] for the song, with
    instrumental intros/outros/breaks excluded. None if the aligner isn't
    installed, transcription fails, or no vocals were detected."""
    words = _words_from_audio(audio_path)
    if not words:
        return None
    windows = group_windows(words, gap_sec)
    if not windows:
        return None
    log.info(f"Vocal windows: {[(round(a, 1), round(b, 1)) for a, b in windows]}")
    return windows


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(health_check())
    if len(sys.argv) > 1:
        print(get_vocal_windows(Path(sys.argv[1])))
