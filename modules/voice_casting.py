"""
Claw Bot — Voice Casting (P2 of the character-dialogue build)

Assigns a distinct Kokoro voice to each character so they SOUND different when
they talk. The narrator keeps the global runtime voice; characters get voices
from curated pools, picked by a light gender guess and kept distinct from each
other and from the narrator.

Data home: `script["characters"][i]["voice"]` (a Kokoro voice id). Set once at
script-validation time; never overrides a voice the user already chose.

Pure logic — no GPU, no network. `resolve_voice()` is what the render path calls
to turn a shot's `speaker` into the actual voice id for TTS.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Optional

_AGENT = Path(__file__).parent.parent.resolve()
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

log = logging.getLogger("claw_bot.voice_casting")

# Curated Kokoro voice pools (voices bundled with the kokoro package).
FEMALE_VOICES = [
    "af_bella", "af_nicole", "af_sarah", "af_sky", "af_aoede", "af_kore",
    "bf_emma", "bf_isabella", "bf_alice",
]
MALE_VOICES = [
    "am_adam", "am_michael", "am_echo", "am_eric", "am_liam", "am_puck",
    "bm_george", "bm_daniel", "bm_lewis",
]
# Default narrator voice (kept out of the character pools for contrast).
NARRATOR_DEFAULT = "af_heart"

# All known/selectable voices (for UI dropdowns + validation).
ALL_VOICES = [NARRATOR_DEFAULT, "af_jessica", "af_river"] + FEMALE_VOICES + MALE_VOICES

_MALE_HINTS = re.compile(
    r"\b(boy|man|male|he|him|his|father|dad|papa|king|prince|brother|son|"
    r"grandpa|grandfather|uncle|mr|sir|gentleman|rooster|bull|stag|buck)\b",
    re.IGNORECASE,
)
_FEMALE_HINTS = re.compile(
    r"\b(girl|woman|female|she|her|hers|mother|mom|mama|queen|princess|sister|"
    r"daughter|grandma|grandmother|aunt|mrs|miss|maam|madam|lady|hen|doe|mare)\b",
    re.IGNORECASE,
)


def list_voices() -> list[str]:
    """Selectable voice ids (narrator default first)."""
    return list(dict.fromkeys(ALL_VOICES))


def is_valid_voice(voice: str) -> bool:
    return bool(voice) and voice in set(ALL_VOICES)


def guess_gender(character: dict, context: str = "") -> str:
    """'male' | 'female' | 'neutral' from the character fields PLUS optional
    story context (narration pronouns like 'his'/'her' are the strongest signal,
    since appearance text rarely states gender)."""
    # An explicit gender field (set by the structurer / audio-first stamp) is
    # authoritative — trust it over heuristics so voice + visual never disagree.
    explicit = str(character.get("gender", "")).strip().lower()
    if explicit in ("male", "female"):
        return explicit

    # SELF fields first (appearance/token/name/type). A character's OWN
    # description ("a 7-year-old boy...") is decisive and must outrank the story
    # context — otherwise a male child whose dialogue addresses "Mom" gets a
    # female hint from that word and is mis-voiced. Context is only a tiebreaker.
    self_blob = " ".join(str(character.get(k, "")) for k in
                         ("appearance", "locked_visual_token", "name", "type"))
    male = len(_MALE_HINTS.findall(self_blob))
    female = len(_FEMALE_HINTS.findall(self_blob))
    if male > female:
        return "male"
    if female > male:
        return "female"

    # Self fields are gender-silent or tied — fall back to story context
    # (narration pronouns 'his'/'her' the character is described with).
    if context:
        cmale = len(_MALE_HINTS.findall(context))
        cfemale = len(_FEMALE_HINTS.findall(context))
        if cmale > cfemale:
            return "male"
        if cfemale > cmale:
            return "female"
    return "neutral"


def _gender_context(script: dict, name: str) -> str:
    """Collect narration from shots that this character speaks in or is named in,
    so pronoun cues ('his'/'her') can disambiguate gender."""
    if not name:
        return ""
    want = name.strip().lower()
    parts = []
    for sh in script.get("shots", []):
        if not isinstance(sh, dict):
            continue
        narr = str(sh.get("narration", ""))
        spk = str(sh.get("speaker", "")).strip().lower()
        if spk == want or want in narr.lower():
            parts.append(narr)
    return " ".join(parts)


def assign_voices(script: dict, narrator_voice: Optional[str] = None) -> dict:
    """Give every character a distinct `voice`. Idempotent: skips characters
    that already have one. Returns the same script (mutated)."""
    chars = [c for c in script.get("characters", []) if isinstance(c, dict)]
    if not chars:
        return script

    nv = narrator_voice or _narrator_voice()
    taken = {c["voice"] for c in chars if c.get("voice")}
    taken.add(nv)  # keep characters distinct from the narrator

    # Round-robin cursors per pool so repeated characters of the same gender
    # still get different voices.
    cursors = {"female": 0, "male": 0, "neutral": 0}

    def _pick(pool: list[str], key: str) -> str:
        n = len(pool)
        for _ in range(n):
            cand = pool[cursors[key] % n]
            cursors[key] += 1
            if cand not in taken:
                return cand
        # Pool exhausted (more chars than voices) — allow reuse of the next one.
        cand = pool[cursors[key] % n]
        cursors[key] += 1
        return cand

    for c in chars:
        if c.get("voice"):
            continue
        g = guess_gender(c, _gender_context(script, c.get("name", "")))
        # Persist a decided gender so the pronoun deriver (prompt_assembler.
        # _pronoun) and any re-cast use it directly instead of re-guessing.
        if g in ("male", "female") and not str(c.get("gender", "")).strip():
            c["gender"] = g
        if g == "male":
            voice = _pick(MALE_VOICES, "male")
        elif g == "female":
            voice = _pick(FEMALE_VOICES, "female")
        else:
            # Neutral (often animals): alternate female/male pools for variety.
            pool, key = ((FEMALE_VOICES, "female")
                         if cursors["neutral"] % 2 == 0 else (MALE_VOICES, "male"))
            cursors["neutral"] += 1
            voice = _pick(pool, key)
        c["voice"] = voice
        taken.add(voice)
        log.info(f"Cast voice '{voice}' → {c.get('name', '?')} (guess={g})")
    return script


def is_character_speaker(speaker: Optional[str]) -> bool:
    """True when this shot is spoken by a named character (lip-sync candidate),
    False for narrator/empty (voiceover, no lip-sync)."""
    return bool(speaker) and speaker.strip().lower() != "narrator"


def resolve_voice(script: dict, speaker: Optional[str]) -> str:
    """Turn a shot's `speaker` into the Kokoro voice id for TTS.
    narrator/unknown → narrator voice; character → their assigned voice."""
    nv = _narrator_voice()
    if not speaker or speaker.strip().lower() == "narrator":
        return nv
    want = speaker.strip().lower()
    for c in script.get("characters", []):
        if isinstance(c, dict) and (c.get("name") or "").strip().lower() == want:
            v = (c.get("voice") or "").strip()
            return v if is_valid_voice(v) else nv
    return nv


def _narrator_voice() -> str:
    """Effective narrator voice from runtime settings (default if unset)."""
    try:
        from modules import runtime_settings as rs
        v = rs.get_effective_voice()
        return v if v else NARRATOR_DEFAULT
    except Exception:
        return NARRATOR_DEFAULT
