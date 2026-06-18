"""
Claw Bot — Voice Bank (reference anchors for VoxCPM)

VoxCPM picks a *new* random voice every call when given no reference, which is
why the narration drifts shot-to-shot. The fix: give VoxCPM a fixed REFERENCE
wav per speaker so it clones the same voice every time.

We mint those references with Kokoro (deterministic, gender-controlled voices —
`am_*` male, `af_*`/`bf_*` female, `af_heart` narrator) once, cache them on
disk, and hand them to VoxCPM as `prompt_wav_path` + `prompt_text`. Result:
each speaker keeps ONE stable voice, characters differ by gender, and the
narrator is distinct — all still local/offline.

    from modules.voice_bank import reference_for
    wav, text = reference_for("am_adam")   # mints + caches on first call
"""

import logging
import sys
from pathlib import Path
from typing import Optional

_AGENT_DIR = Path(__file__).parent.parent.resolve()
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

log = logging.getLogger("claw_bot.voice_bank")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
# References persist across runs (small, reusable) — keep them in config, not
# the transient outputs folder.
REFS_DIR = PROJECT_ROOT / "05_Config" / "voice_refs"

# A neutral, well-punctuated line ~5s long — enough for VoxCPM to lock the
# timbre. Same text for every voice so prompt_text is consistent.
REF_TEXT = ("The quiet morning light spread softly across the hills, "
            "and a gentle breeze moved through the tall green trees.")

_tts = None  # shared Kokoro engine, lazy


def _engine():
    global _tts
    if _tts is None:
        from modules.tts_engine import TTSEngine
        _tts = TTSEngine()
    return _tts


def reference_for(voice_id: str) -> tuple[Optional[Path], Optional[str]]:
    """Return (reference_wav, reference_text) for a Kokoro voice id, minting and
    caching the wav on first use. On any failure returns (None, None) so the
    caller can fall back to VoxCPM's default (still works, just not pinned)."""
    if not voice_id:
        return None, None
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    wav = REFS_DIR / f"{voice_id}.wav"
    if wav.exists() and wav.stat().st_size > 1024:
        return wav, REF_TEXT
    try:
        log.info(f"Minting voice reference for '{voice_id}' (Kokoro)...")
        _engine().synthesize(REF_TEXT, output_path=wav, voice=voice_id)
        if wav.exists() and wav.stat().st_size > 1024:
            return wav, REF_TEXT
        log.warning(f"Reference for '{voice_id}' came out empty.")
    except Exception as e:
        log.warning(f"Could not mint reference for '{voice_id}': "
                    f"{type(e).__name__}: {e}")
    return None, None


def prime(voice_ids: list[str]) -> dict[str, tuple[Optional[Path], Optional[str]]]:
    """Mint references for a whole cast up front; returns {voice_id: (wav, text)}."""
    out = {}
    for vid in dict.fromkeys(v for v in voice_ids if v):
        out[vid] = reference_for(vid)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for vid in ("af_heart", "am_adam", "af_bella"):
        w, t = reference_for(vid)
        print(f"{vid}: {w} ({'ok' if w else 'FAILED'})")
