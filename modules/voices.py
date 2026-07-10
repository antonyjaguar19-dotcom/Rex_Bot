"""
Claw Bot — canonical Kokoro voice list

There was no single source of truth for voice ids: the dashboard hardcoded one
list, the control panel hinted another, tts_engine held the default, and nothing
validated what got written to runtime_settings. So `!set_facts_voice Af_bella`
(capital A) persisted happily, and the next dashboard page build died with
"Invalid value: Af_bella" — a 500 with no obvious cause.

This module is deliberately dependency-free (no torch, no kokoro) so config code
can import it without dragging in the TTS stack.

Kokoro voice id grammar:  <lang><gender>_<name>
  a = American english, b = British   |   f = female, m = male
"""

# Every voice bundled with Kokoro-82M that this project uses.
KOKORO_VOICES: tuple[str, ...] = (
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael",
    "bf_emma", "bf_isabella",
    "bm_george", "bm_lewis",
)

# Energetic narrators that suit the fast, punchy Facts Shorts read.
FACTS_VOICES: tuple[str, ...] = (
    "af_bella", "af_nicole", "af_sky", "am_adam", "am_michael",
)

DEFAULT_VOICE = "af_heart"
DEFAULT_FACTS_VOICE = "af_bella"


def normalize(voice: str, allowed: tuple[str, ...] = KOKORO_VOICES):
    """Return the canonical id, or None when it isn't a real voice.

    Case and stray whitespace are forgiven ("  Af_Bella " -> "af_bella"),
    because that is what people actually type. Anything else is rejected
    rather than silently stored.
    """
    if not voice:
        return None
    v = str(voice).strip().lower()
    return v if v in allowed else None


def coerce(voice: str, default: str, allowed: tuple[str, ...] = KOKORO_VOICES) -> str:
    """Like normalize() but never fails — used on READ so a config that already
    holds a bad value degrades to the default instead of crashing a page build."""
    return normalize(voice, allowed) or default


def suggest(voice: str, allowed: tuple[str, ...] = KOKORO_VOICES) -> str:
    """A short 'did you mean' hint for error messages."""
    if voice:
        stem = str(voice).strip().lower()
        near = [v for v in allowed if v.startswith(stem[:3])]
        if near:
            return f"Did you mean: {', '.join(near)}?"
    return f"Valid voices: {', '.join(allowed)}"
