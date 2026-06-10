"""
Claw Bot — TTS Engine (Kokoro-82M)

Wraps Kokoro for narration generation. Keeps espeak-ng dependency
self-contained: env vars are set at import time, no system-wide config
needed.

Usage:
    from modules.tts_engine import TTSEngine
    tts = TTSEngine()
    out = tts.synthesize("Hello world.", output_path=Path("test.wav"))
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# --- espeak-ng setup BEFORE importing kokoro/phonemizer ---------------------
# Phonemizer needs to find espeak-ng. Prefer the bundled copy under
# E:\Rexjaw_VFX\00_Tools\espeak-ng (self-contained); fall back to system install.
_PROJECT_ROOT_PRE = Path(__file__).parent.parent.parent.resolve()
_BUNDLED_ESPEAK_DIR = _PROJECT_ROOT_PRE / "00_Tools" / "espeak-ng"
_SYSTEM_ESPEAK_DIR = Path(r"C:\Program Files\eSpeak NG")

for _candidate in (_BUNDLED_ESPEAK_DIR, _SYSTEM_ESPEAK_DIR):
    _exe = _candidate / "espeak-ng.exe"
    _dll = _candidate / "libespeak-ng.dll"
    if _dll.exists() and _exe.exists():
        _ESPEAK_DIR = _candidate
        _ESPEAK_EXE = _exe
        _ESPEAK_DLL = _dll
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", str(_ESPEAK_DLL))
        os.environ.setdefault("PHONEMIZER_ESPEAK_PATH", str(_ESPEAK_EXE))
        break
else:
    # Nothing found — health_check will report this clearly later.
    _ESPEAK_DIR = _BUNDLED_ESPEAK_DIR
    _ESPEAK_EXE = _ESPEAK_DIR / "espeak-ng.exe"
    _ESPEAK_DLL = _ESPEAK_DIR / "libespeak-ng.dll"
# ---------------------------------------------------------------------------

# Ensure 02_Agent is on sys.path
_AGENT_DIR = Path(__file__).parent.parent.resolve()
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

log = logging.getLogger("claw_bot.tts")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "04_Outputs" / "audio"


class TTSEngine:
    """
    Kokoro-82M wrapper. Tiny, fast, Apache 2.0, runs on CPU comfortably.
    Voices are bundled with the package — no extra downloads needed at
    runtime (first import will pull weights from HF if not cached).
    """

    DEFAULT_VOICE = "af_heart"      # warm, child-friendly American female
    DEFAULT_LANG = "a"               # 'a' = American English, 'b' = British
    DEFAULT_SPEED = 1.0
    SAMPLE_RATE = 24000              # Kokoro outputs at 24 kHz

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        lang_code: Optional[str] = None,
        speed: float = DEFAULT_SPEED,
    ):
        self.voice = voice
        # Derive lang_code from voice prefix unless caller forced one.
        # af_/am_ = American English ('a'), bf_/bm_ = British English ('b').
        self.lang_code = lang_code or self._lang_for_voice(voice)
        self.speed = speed
        self._pipeline = None         # lazy-init on first synthesize()
        log.info(
            f"TTSEngine configured — voice={self.voice} lang={self.lang_code} speed={speed}"
        )

    @staticmethod
    def _lang_for_voice(voice: str) -> str:
        v = (voice or "").lower()
        if v.startswith(("bf_", "bm_")):
            return "b"
        return "a"

    # -------- Lazy init: avoid loading model until first use --------

    def _ensure_loaded(self):
        if self._pipeline is not None:
            return
        try:
            from kokoro import KPipeline
        except ImportError as e:
            raise RuntimeError(
                "kokoro package not installed. Run: pip install kokoro soundfile"
            ) from e
        log.info("Loading Kokoro pipeline (first call may download weights)...")
        self._pipeline = KPipeline(lang_code=self.lang_code)
        log.info("Kokoro pipeline ready.")

    # -------- Health check --------

    def health_check(self) -> tuple[bool, str]:
        """Verify Kokoro + espeak-ng are wired correctly."""
        if not _ESPEAK_EXE.exists():
            return False, f"espeak-ng.exe not found at {_ESPEAK_EXE}"
        try:
            self._ensure_loaded()
            return True, f"Kokoro ready (voice={self.voice})"
        except Exception as e:
            return False, f"Kokoro init failed: {type(e).__name__}: {e}"

    # -------- Public API --------

    def synthesize(
        self,
        text: str,
        output_path: Optional[Path] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> Path:
        """
        Convert text to speech, write WAV file, return path.

        Args:
            text: The narration string. Sentence-level chunks work best.
            output_path: Where to save. If None, auto-named in 04_Outputs/audio/.
            voice: Override default voice for this call.
            speed: Override default speed for this call (1.0 = normal).

        Returns:
            Path to the written WAV file.
        """
        import numpy as np
        import soundfile as sf

        if not text or not text.strip():
            raise ValueError("synthesize() got empty text.")

        eff_voice = voice or self.voice
        eff_speed = speed if speed is not None else self.speed

        # If voice changed since pipeline was loaded for a different language,
        # rebuild the pipeline so phonemizer uses the right dictionary.
        eff_lang = self._lang_for_voice(eff_voice)
        if eff_lang != self.lang_code:
            log.info(f"Voice '{eff_voice}' needs lang_code={eff_lang}; reloading pipeline.")
            self.lang_code = eff_lang
            self._pipeline = None
        self._ensure_loaded()

        if output_path is None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            from hashlib import md5
            stub = md5(text.encode()).hexdigest()[:10]
            output_path = OUTPUT_DIR / f"tts_{stub}.wav"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        log.info(
            f"Synthesizing ({len(text)} chars, voice={eff_voice}, speed={eff_speed})..."
        )

        # KPipeline yields generator of (graphemes, phonemes, audio_tensor)
        # for each chunk. Concatenate all audio into one waveform.
        chunks = []
        for _, _, audio in self._pipeline(text, voice=eff_voice, speed=eff_speed):
            # audio is a torch tensor, shape (samples,)
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype=np.float32))

        if not chunks:
            raise RuntimeError("Kokoro produced no audio chunks.")

        full_audio = np.concatenate(chunks)
        sf.write(str(output_path), full_audio, self.SAMPLE_RATE)
        duration = len(full_audio) / self.SAMPLE_RATE
        log.info(
            f"Wrote {output_path.name} — {duration:.2f}s, "
            f"{output_path.stat().st_size/1024:.1f} KB"
        )
        return output_path


# ==============================================================================
# Standalone test
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    tts = TTSEngine()
    ok, msg = tts.health_check()
    print(f"Health: {'OK' if ok else 'FAIL'} — {msg}")
    if not ok:
        raise SystemExit(1)
    out = tts.synthesize(
        "Hello! Today we'll learn about being kind to others. "
        "Sharing is caring, and a small smile can make someone's day."
    )
    print(f"Wrote: {out}")
