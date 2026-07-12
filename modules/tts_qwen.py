"""
Claw Bot — Qwen3-TTS bridge (production horror narrator)

Qwen3-TTS Custom Voice = a preset, deterministic voice → identical narrator
every beat (no drift), high quality. It needs transformers 4.56.1, which clashes
with the main venv, so it lives in an ISOLATED venv (03_Models/venv_qwen_tts)
and we drive it over a subprocess — exactly like the Chatterbox bridge. ComfyUI
is never touched.

Returns (master_wav, spans) like the other engines so the horror pipeline can
use it interchangeably.
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

log = logging.getLogger("claw_bot.tts_qwen")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
VENV_PY = PROJECT_ROOT / "03_Models" / "venv_qwen_tts" / "Scripts" / "python.exe"
CLI = _AGENT_DIR / "qwen_tts_cli.py"
AUDIO_DIR = PROJECT_ROOT / "04_Outputs" / "audio"
GROUP_GAP_SEC = 0.4
TIMEOUT_SEC = 7200  # Qwen3-TTS is slow per-beat; a full ~55-beat story needs headroom


def is_available() -> bool:
    return VENV_PY.exists() and CLI.exists()


def health_check() -> tuple[bool, str]:
    if not VENV_PY.exists():
        return False, f"qwen venv missing: {VENV_PY}"
    if not CLI.exists():
        return False, f"qwen_tts_cli.py missing: {CLI}"
    return True, "qwen3-tts bridge ready"


def _run_cli(texts, output_path, speaker, language, gap_sec, instruct,
             segments_dir=None):
    """Drive qwen_tts_cli in its isolated venv. Returns (master, spans, files)."""
    ok, msg = health_check()
    if not ok:
        raise RuntimeError(msg)
    texts = [t for t in texts if (t or "").strip()]
    if not texts:
        raise ValueError("qwen tts got no non-empty texts.")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        job = Path(td) / "job.json"
        spans_out = Path(td) / "spans.json"
        payload = {
            "texts": texts, "speaker": speaker, "language": language,
            "gap_sec": gap_sec, "instruct": instruct or "",
            "out_wav": str(output_path), "spans_out": str(spans_out),
        }
        if segments_dir:
            payload["segments_dir"] = str(segments_dir)
        job.write_text(json.dumps(payload), encoding="utf-8")

        log.info(f"Qwen3-TTS voicing {len(texts)} segments (speaker={speaker})...")
        r = subprocess.run(
            [str(VENV_PY), str(CLI), str(job)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=TIMEOUT_SEC,
        )
        if r.returncode != 0:
            raise RuntimeError(f"qwen_tts_cli failed (rc={r.returncode}): "
                               f"{(r.stderr or r.stdout or '')[-600:]}")
        if not spans_out.exists():
            raise RuntimeError(f"qwen_tts_cli produced no spans: {r.stdout[-300:]}")
        data = json.loads(spans_out.read_text(encoding="utf-8"))

    spans = [(t, float(a), float(b)) for t, a, b in data["spans"]]
    return output_path, spans, data.get("files", [])


def synthesize_each(
    texts: list,
    out_dir: Path,
    speaker: str = "eric",
    language: str = "english",
    instruct: Optional[str] = None,
) -> list:
    """One WAV per text, in ONE model load. Returns the list of WAV paths.

    The CLI already voices every text on its own pass, so this just asks it to
    save each pass instead of only the concatenated master. Slicing the master
    back apart on spans was clipping words at the seams — this has no seams.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    master = out_dir / "qwen_master.wav"
    _, _, files = _run_cli(texts, master, speaker, language, GROUP_GAP_SEC,
                           instruct, segments_dir=out_dir)
    if len(files) != len([t for t in texts if (t or "").strip()]):
        raise RuntimeError(f"qwen_tts_cli returned {len(files)} segment files "
                           f"for {len(texts)} texts")
    return [Path(f) for f in files]


def synthesize_segments(
    texts: list,
    output_path: Optional[Path] = None,
    speaker: str = "eric",
    language: str = "english",
    gap_sec: float = GROUP_GAP_SEC,
    instruct: Optional[str] = None,
    pitch: float = 1.0,
) -> tuple[Path, list[tuple[str, float, float]]]:
    """Voice each text with Qwen3-TTS (preset Custom Voice) in the isolated venv,
    stitched with silence gaps. `instruct` steers tone/emotion (e.g. a horror
    delivery). Returns (master_wav, spans)."""
    ok, msg = health_check()
    if not ok:
        raise RuntimeError(msg)
    texts = [t for t in texts if (t or "").strip()]
    if not texts:
        raise ValueError("synthesize_segments() got no non-empty texts.")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        from datetime import datetime
        output_path = AUDIO_DIR / f"qwen_{datetime.now():%Y%m%d_%H%M%S}.wav"
    output_path = Path(output_path)

    output_path, spans, _ = _run_cli(texts, output_path, speaker, language,
                                     gap_sec, instruct)
    # Deepen + slow the voice (one ffmpeg asetrate step) for a graver narrator.
    if pitch and abs(pitch - 1.0) > 1e-3:
        output_path = _deepen(output_path, pitch)
    log.info(f"Qwen3-TTS wrote {output_path.name} — {len(spans)} segments.")
    return output_path, spans


def _deepen(wav: Path, factor: float) -> Path:
    """Lower the pitch but KEEP the original tempo (no slow-mo): asetrate drops
    pitch+speed, then atempo restores the speed. factor < 1.0 = deeper voice at
    natural pace. Pacing/slowness comes from the TTS instruct, not time-stretch.
    Rewrites the file in place."""
    import subprocess
    from modules.assembly import FFMPEG_EXE, FFPROBE_EXE
    try:
        sr = int(float(subprocess.run(
            [str(FFPROBE_EXE), "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate", "-of", "csv=p=0", str(wav)],
            capture_output=True, text=True, timeout=60).stdout.strip() or 24000))
    except Exception:
        sr = 24000
    tmp = wav.with_name(wav.stem + "_deep" + wav.suffix)
    new_rate = int(sr * factor)
    tempo = 1.0 / factor  # undo the speed change asetrate introduced
    cmd = [str(FFMPEG_EXE), "-y", "-loglevel", "error", "-i", str(wav),
           "-af", f"asetrate={new_rate},aresample={sr},atempo={tempo:.4f}", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode == 0 and tmp.exists():
        tmp.replace(wav)
    else:
        log.warning(f"deepen failed: {r.stderr[-200:]}")
    return wav


def synthesize_full(text: str, output_path: Path, speaker: str = "eric") -> Path:
    """Whole-text convenience (single segment). Returns wav path."""
    wav, _ = synthesize_segments([text], output_path=output_path, speaker=speaker)
    return wav


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(health_check())
