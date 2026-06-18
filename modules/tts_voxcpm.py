"""
Claw Bot — VoxCPM TTS (local, offline, Apache-2.0)

The audio-first primary voice engine. VoxCPM is a tokenizer-free TTS that runs
fully on the local GPU (no API key, no cloud, no quota, monetizable output —
unlike ElevenLabs' free tier). Weights auto-download once into the contained
HF cache under 03_Models.

VoxCPM has NO native character timestamps. So for the audio-first pipeline we
voice each breath-group (sentence / comma clause) SEPARATELY and stitch them
with a controlled silence gap. That gives us EXACT (text, t_start, t_end) spans
with zero guessing — the segmenter then tiles the cut windows on those spans.

Usage:
    from modules.tts_voxcpm import VoxCPMTTS
    tts = VoxCPMTTS()
    # drop-in (whole text, one wav):
    tts.synthesize("Hello world.", output_path=Path("out.wav"))
    # audio-first (per-group, exact spans):
    wav, spans = tts.synthesize_segments(["Hi there.", "Run fast!"], Path("m.wav"))
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# --- Containment: keep HF weights inside E:\Rexjaw_VFX (not C:\Users cache) ---
_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
_HF_CACHE = _PROJECT_ROOT / "03_Models" / "hf_cache"
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(_HF_CACHE / "hub"))
# ----------------------------------------------------------------------------

_AGENT_DIR = Path(__file__).parent.parent.resolve()
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

log = logging.getLogger("claw_bot.tts_voxcpm")

OUTPUT_DIR = _PROJECT_ROOT / "04_Outputs" / "audio"

# Model repo (overridable via env). 0.5B = the one used by the public demo space.
DEFAULT_MODEL_ID = os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM-0.5B")
# Generation knobs (fast defaults; quality scales with timesteps).
DEFAULT_CFG = float(os.environ.get("VOXCPM_CFG", "2.0"))
DEFAULT_TIMESTEPS = int(os.environ.get("VOXCPM_TIMESTEPS", "10"))
# Silence inserted between breath-groups (seconds) — this IS the cut point.
GROUP_GAP_SEC = 0.34


class VoxCPMTTS:
    """Local VoxCPM wrapper. Lazy-loads the model on first synth call."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        cfg_value: float = DEFAULT_CFG,
        inference_timesteps: int = DEFAULT_TIMESTEPS,
        voice_text: Optional[str] = None,
        reference_wav: Optional[Path] = None,
    ):
        self.model_id = model_id
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps
        # Optional zero-shot voice: a reference wav (clone) OR leave None for the
        # model's built-in voice. voice_text is reserved for future Voice-Design.
        self.voice_text = voice_text
        self.reference_wav = Path(reference_wav) if reference_wav else None
        self._model = None
        self._sr = None
        log.info(f"VoxCPMTTS configured — model={model_id} cfg={cfg_value} "
                 f"timesteps={inference_timesteps}")

    # -------- Lazy init --------

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from voxcpm import VoxCPM
        except ImportError as e:
            raise RuntimeError(
                "voxcpm package not installed. Run: "
                "venv\\Scripts\\pip install voxcpm"
            ) from e
        log.info(f"Loading VoxCPM ({self.model_id}) — first call may download weights...")
        self._model = VoxCPM.from_pretrained(self.model_id, load_denoiser=False)
        # sample_rate lives on the inner tts_model
        self._sr = int(getattr(getattr(self._model, "tts_model", None),
                               "sample_rate", 16000))
        log.info(f"VoxCPM ready (sample_rate={self._sr}).")

    @property
    def sample_rate(self) -> int:
        self._ensure_loaded()
        return self._sr

    # -------- Health check --------

    def health_check(self, load_model: bool = False) -> tuple[bool, str]:
        """Verify voxcpm is importable. With load_model=True also loads weights
        (slow first time — downloads the model)."""
        try:
            import voxcpm  # noqa: F401
        except ImportError:
            return False, "voxcpm package not installed (pip install voxcpm)"
        if not load_model:
            return True, "voxcpm import OK (model not yet loaded)"
        try:
            self._ensure_loaded()
            return True, f"VoxCPM ready (sr={self._sr}, model={self.model_id})"
        except Exception as e:
            return False, f"VoxCPM load failed: {type(e).__name__}: {e}"

    # -------- Low-level: one group -> numpy waveform --------

    def _gen(self, text: str):
        import numpy as np
        self._ensure_loaded()
        kwargs = dict(
            text=text,
            cfg_value=self.cfg_value,
            inference_timesteps=self.inference_timesteps,
        )
        if self.reference_wav and self.reference_wav.exists():
            kwargs["prompt_wav_path"] = str(self.reference_wav)
            if self.voice_text:
                kwargs["prompt_text"] = self.voice_text
        wav = self._model.generate(**kwargs)
        return np.asarray(wav, dtype=np.float32).reshape(-1)

    # -------- Public: drop-in whole-text synth --------

    def synthesize(
        self,
        text: str,
        output_path: Optional[Path] = None,
        voice: Optional[str] = None,   # accepted for interface parity; unused
        speed: Optional[float] = None,  # accepted for interface parity; unused
    ) -> Path:
        """Voice the whole text into one wav (TTSEngine-compatible signature)."""
        import soundfile as sf
        if not text or not text.strip():
            raise ValueError("synthesize() got empty text.")
        wav = self._gen(text.strip())
        out = self._resolve_out(output_path, text)
        sf.write(str(out), wav, self._sr)
        log.info(f"Wrote {out.name} — {len(wav)/self._sr:.2f}s")
        return out

    # -------- Public: per-group synth with exact spans (audio-first) --------

    def synthesize_segments(
        self,
        groups: list[str],
        output_path: Optional[Path] = None,
        gap_sec: float = GROUP_GAP_SEC,
    ) -> tuple[Path, list[tuple[str, float, float]]]:
        """Voice each breath-group separately and stitch with a silence gap.

        Returns (master_wav_path, spans) where spans = [(text, t_start, t_end)]
        in seconds — exact, no detection needed. The inserted gaps are the cut
        points the segmenter tiles its windows around.
        """
        import numpy as np
        import soundfile as sf

        groups = [g.strip() for g in groups if g and g.strip()]
        if not groups:
            raise ValueError("synthesize_segments() got no non-empty groups.")
        self._ensure_loaded()

        gap_samples = int(round(gap_sec * self._sr))
        gap = np.zeros(gap_samples, dtype=np.float32)

        pieces: list = []
        spans: list[tuple[str, float, float]] = []
        cursor = 0  # running sample count
        for k, g in enumerate(groups):
            wav = self._gen(g)
            t_start = cursor / self._sr
            pieces.append(wav)
            cursor += len(wav)
            t_end = cursor / self._sr
            spans.append((g, round(t_start, 3), round(t_end, 3)))
            if k < len(groups) - 1:          # gap after every group but the last
                pieces.append(gap)
                cursor += gap_samples

        master = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
        out = self._resolve_out(output_path, " ".join(groups))
        sf.write(str(out), master, self._sr)
        log.info(f"Wrote master {out.name} — {len(master)/self._sr:.2f}s, "
                 f"{len(spans)} groups.")
        return out, spans

    # -------- helpers --------

    def _resolve_out(self, output_path: Optional[Path], text: str) -> Path:
        if output_path is None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            from hashlib import md5
            stub = md5(text.encode()).hexdigest()[:10]
            output_path = OUTPUT_DIR / f"vox_{stub}.wav"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path


# ==============================================================================
# Standalone smoke test
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    tts = VoxCPMTTS()
    ok, msg = tts.health_check(load_model=True)
    print(f"Health: {'OK' if ok else 'FAIL'} — {msg}")
    if not ok:
        raise SystemExit(1)
    wav, spans = tts.synthesize_segments(
        ["Hello there, little one.", "Today we learn to be brave!"],
    )
    print(f"Wrote: {wav}")
    for t, a, b in spans:
        print(f"  [{a:.2f}-{b:.2f}] {t}")
