"""
Claw Bot — ElevenLabs TTS adapter (audio-first pipeline)

CONTAINMENT EXCEPTION (Core Rule #1): this is the ONE cloud dependency in the
pipeline. Narration TEXT is sent to ElevenLabs servers to synthesise speech.
Everything else stays inside E:\\Rexjaw_VFX. Chosen for best voice quality +
native character-level timestamps (the pause data that drives audio-first cuts).

Free tier = NON-COMMERCIAL + watermark + "elevenlabs.io" attribution, ~10k
credits/mo (~10 min). Monetised YouTube needs the $5/mo Starter plan. Chatterbox
is the local, commercial-OK fallback when the key is missing or quota is gone.

Drop-in compatible with TTSEngine.synthesize(text, output_path, voice) so it can
slot into clip_generator, PLUS synthesize_with_timestamps() which returns the
per-character alignment that narration_segmenter consumes.

Endpoint: POST /v1/text-to-speech/{voice_id}/with-timestamps
Docs: https://elevenlabs.io/docs/api-reference/text-to-speech/convert-with-timestamps
"""

import base64
import logging
import os
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

_AGENT_DIR = Path(__file__).parent.parent.resolve()
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

log = logging.getLogger("claw_bot.tts_elevenlabs")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "04_Outputs" / "audio"

API_BASE = "https://api.elevenlabs.io/v1"
# Long-standing default voice ("Rachel"). Override via ELEVENLABS_VOICE_ID.
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
DEFAULT_MODEL_ID = "eleven_multilingual_v2"
# pcm_24000 = raw 16-bit mono PCM @ 24 kHz — matches Kokoro's sample rate so the
# rest of the pipeline (ffprobe/ffmpeg) treats both engines identically. We wrap
# the raw PCM in a WAV container ourselves (no extra mp3 decode dependency).
OUTPUT_FORMAT = "pcm_24000"
SAMPLE_RATE = 24000


@dataclass
class Alignment:
    """Per-character timing returned by the with-timestamps endpoint.

    characters[i] spans [starts[i], ends[i]] seconds in the rendered audio.
    This is the raw material narration_segmenter uses to find pauses.
    """
    characters: list = field(default_factory=list)
    starts: list = field(default_factory=list)     # character_start_times_seconds
    ends: list = field(default_factory=list)        # character_end_times_seconds

    @property
    def duration(self) -> float:
        return float(self.ends[-1]) if self.ends else 0.0

    def to_dict(self) -> dict:
        return {"characters": self.characters, "starts": self.starts, "ends": self.ends}

    @classmethod
    def from_dict(cls, d: dict) -> "Alignment":
        return cls(
            characters=list(d.get("characters", [])),
            starts=list(d.get("starts", []) or d.get("character_start_times_seconds", [])),
            ends=list(d.get("ends", []) or d.get("character_end_times_seconds", [])),
        )


class ElevenLabsTTS:
    """ElevenLabs cloud TTS with character-level timestamps.

    API key is read from the ELEVENLABS_API_KEY env var (loaded from
    05_Config/secrets.env by claw_bot via python-dotenv). Never hard-code it.
    """

    SAMPLE_RATE = SAMPLE_RATE

    def __init__(
        self,
        voice_id: Optional[str] = None,
        model_id: str = DEFAULT_MODEL_ID,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID") or DEFAULT_VOICE_ID
        self.model_id = model_id
        self.timeout = timeout
        log.info(
            f"ElevenLabsTTS configured — voice_id={self.voice_id} model={self.model_id} "
            f"key={'present' if self.api_key else 'MISSING'}"
        )

    # -------- Health --------

    def health_check(self) -> tuple[bool, str]:
        """Verify the key works by hitting the lightweight /user endpoint."""
        if not self.api_key:
            return False, "ELEVENLABS_API_KEY not set in 05_Config/secrets.env"
        try:
            r = requests.get(
                f"{API_BASE}/user",
                headers={"xi-api-key": self.api_key},
                timeout=30,
            )
            if r.status_code == 401:
                return False, "ElevenLabs key rejected (401) — check the key"
            r.raise_for_status()
            sub = (r.json() or {}).get("subscription", {})
            used = sub.get("character_count", "?")
            cap = sub.get("character_limit", "?")
            return True, f"ElevenLabs ready (credits used {used}/{cap})"
        except Exception as e:
            return False, f"ElevenLabs health check failed: {type(e).__name__}: {e}"

    # -------- Core: synth + timestamps --------

    def synthesize_with_timestamps(
        self,
        text: str,
        output_path: Optional[Path] = None,
        voice_id: Optional[str] = None,
    ) -> tuple[Path, Alignment]:
        """Synthesise `text` to a WAV file AND return its character alignment.

        Returns (wav_path, Alignment). The alignment is the pause data the
        segmenter uses to decide where shots cut.
        """
        if not text or not text.strip():
            raise ValueError("synthesize_with_timestamps() got empty text.")
        if not self.api_key:
            raise RuntimeError(
                "ELEVENLABS_API_KEY missing. Add it to 05_Config/secrets.env, or "
                "fall back to the local Chatterbox engine."
            )

        vid = voice_id or self.voice_id
        if output_path is None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            from hashlib import md5
            stub = md5(text.encode("utf-8")).hexdigest()[:10]
            output_path = OUTPUT_DIR / f"el_{stub}.wav"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        url = f"{API_BASE}/text-to-speech/{vid}/with-timestamps"
        payload = {
            "text": text,
            "model_id": self.model_id,
            "output_format": OUTPUT_FORMAT,
        }
        log.info(
            f"ElevenLabs synth ({len(text)} chars ≈ {len(text)} credits, voice={vid})..."
        )
        r = requests.post(
            url,
            headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
            json=payload,
            params={"output_format": OUTPUT_FORMAT},
            timeout=self.timeout,
        )
        if r.status_code == 401:
            raise RuntimeError("ElevenLabs key rejected (401).")
        if r.status_code == 429:
            raise RuntimeError("ElevenLabs quota exhausted (429) — fall back to Chatterbox.")
        r.raise_for_status()
        data = r.json()

        audio_b64 = data.get("audio_base64")
        if not audio_b64:
            raise RuntimeError(f"ElevenLabs response missing audio_base64: {str(data)[:200]}")
        pcm = base64.b64decode(audio_b64)
        self._write_wav(pcm, output_path)

        al = data.get("alignment") or data.get("normalized_alignment") or {}
        alignment = Alignment(
            characters=al.get("characters", []),
            starts=al.get("character_start_times_seconds", []),
            ends=al.get("character_end_times_seconds", []),
        )
        log.info(
            f"Wrote {output_path.name} — {alignment.duration:.2f}s, "
            f"{len(alignment.characters)} char timestamps"
        )
        return output_path, alignment

    def synthesize(
        self,
        text: str,
        output_path: Optional[Path] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,   # accepted for TTSEngine compatibility; ignored
    ) -> Path:
        """Drop-in compatible with TTSEngine.synthesize — returns just the Path.

        `voice` here is treated as an ElevenLabs voice_id. Timestamps are
        discarded; use synthesize_with_timestamps() when you need them.
        """
        path, _ = self.synthesize_with_timestamps(text, output_path, voice_id=voice)
        return path

    # -------- Helpers --------

    def _write_wav(self, pcm_bytes: bytes, out_path: Path):
        """Wrap raw 16-bit mono PCM @ 24 kHz in a WAV container (stdlib only)."""
        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)            # 16-bit
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm_bytes)


# ==============================================================================
# Standalone test (needs ELEVENLABS_API_KEY in env)
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=PROJECT_ROOT / "05_Config" / "secrets.env")

    tts = ElevenLabsTTS()
    ok, msg = tts.health_check()
    print(f"Health: {'OK' if ok else 'FAIL'} — {msg}")
    if not ok:
        raise SystemExit(1)
    out, al = tts.synthesize_with_timestamps(
        "Morning dew sparkled on the meadow. The animals gathered to watch the race."
    )
    print(f"Wrote: {out}  dur={al.duration:.2f}s  chars={len(al.characters)}")
