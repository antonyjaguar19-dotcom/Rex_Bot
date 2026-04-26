"""
Claw Bot — Safety Filter
Secondary check on generated scripts before posting to Discord.
Catches borderline content the first prompt might miss.

Philosophy: false positives (occasional unnecessary flags) are ACCEPTABLE.
False negatives (unsafe content slipping through) are NOT.
"""

import logging
import re
from typing import Tuple, List

log = logging.getLogger("claw_bot.safety_filter")


# ==============================================================================
# BLOCKED TERMS — hard rejects
# ==============================================================================

# Any occurrence of these in narration or visual descriptions triggers rejection
HARD_BLOCKED_TERMS = [
    # Real-world violence that kills or permanently harms
    "kill", "killed", "murder", "murdered", "stab", "stabbed",
    "shot him", "shot her", "shot them", "shoot him", "shoot her",
    "strangle", "strangled", "torture", "tortured",
    # Graphic injury
    "blood", "bleeding", "bloody", "wound", "gunshot", "gore",
    # Death in severe/explicit contexts (the word 'dying' allowed — 'dying embers', 'dying leaves')
    "funeral", "grave", "buried alive", "corpse",
    # Weapons used to harm
    "gun", "rifle", "pistol", "firearm", "bomb", "explosive",
    # Adult / sexual content
    "sex", "sexual", "sexy", "naked", "nude", "erotic",
    # Substance abuse
    "beer", "wine", "alcohol", "drunk", "vodka", "whiskey",
    "smoking cigarette", "cigarette", "cigar",
    "drug", "drugs", "cocaine", "heroin", "marijuana",
    # Self-harm — explicit indicators only
    "cut himself", "cut herself", "cut myself",
    "hate myself", "want to die", "kill myself", "kill himself",
    "suicide", "suicidal",
]

# Warn-level — allowed in the story but logged for your review.
# These are normal storytelling vocabulary that occasionally appear in kids' content.
SOFT_WARNING_TERMS = [
    # Fear words (essential for emotional arcs)
    "scary", "scared", "afraid", "afraid of", "fear", "frightened",
    "terrified", "terrifying", "nervous", "worried",
    # Fantasy creatures (fine in fiction)
    "monster", "monsters", "demon", "ghost", "haunt", "haunted",
    "nightmare", "evil", "witch", "vampire", "zombie",
    # Dragon / fire story vocabulary
    "smoke", "smoking", "fire", "flames", "burning", "burned",
    "dying", "died", "dead", "death",
    # Conflict vocabulary (kids stories have these)
    "fight", "fighting", "punch", "kick", "slap",
    "hurt", "hurt badly", "beat up", "hit him", "hit her",
    "weapon", "knife", "sword", "war", "battle",
    # Affection (grandma-kissing-forehead territory)
    "kiss", "romance", "romantic",
    # Existing soft warnings preserved
    "angry tears", "screamed", "screaming", "crying loudly",
    "bullied", "bully", "teased cruelly", "fell hard",
    "dangerous", "abandoned", "alone in the dark",
]


# ==============================================================================
# STRUCTURAL CHECKS
# ==============================================================================

def _collect_all_text(script: dict) -> str:
    """Gathers all user-visible text from a script into one lowercase string."""
    parts = [
        script.get("title", ""),
        script.get("theme", ""),
        script.get("setting", ""),
        script.get("moral", ""),
    ]
    for ch in script.get("characters", []):
        if isinstance(ch, dict):
            parts.append(ch.get("name", ""))
            parts.append(ch.get("appearance", ""))
    for shot in script.get("shots", []):
        parts.append(shot.get("narration", ""))
        parts.append(shot.get("visual_description", ""))
        parts.append(shot.get("first_frame_prompt", ""))
        parts.append(shot.get("last_frame_prompt", ""))
    return " ".join(parts).lower()


# ==============================================================================
# MAIN API
# ==============================================================================

def check_safety(script: dict) -> Tuple[bool, List[str], List[str]]:
    """
    Check a generated script for safety issues.

    Returns:
        (is_safe, blocked_terms_found, warnings)
        is_safe: False if any HARD_BLOCKED_TERM is found.
        blocked_terms_found: list of hard-blocked terms detected.
        warnings: list of soft warnings (safe to proceed, just flag for review).
    """
    all_text = _collect_all_text(script)

    # Use word-boundary regex to avoid false positives
    # (e.g., "die" matching "died" or "studies" — we want whole-word matches)
    blocked_found = []
    for term in HARD_BLOCKED_TERMS:
        # \b = word boundary; re.escape handles special chars like spaces
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, all_text):
            blocked_found.append(term)

    warnings_found = []
    for term in SOFT_WARNING_TERMS:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, all_text):
            warnings_found.append(term)

    is_safe = len(blocked_found) == 0

    if not is_safe:
        log.warning(f"Script failed safety check. Blocked terms: {blocked_found}")
    if warnings_found:
        log.info(f"Script has soft warnings: {warnings_found}")

    return is_safe, blocked_found, warnings_found


def safety_summary_for_discord(blocked: List[str], warnings: List[str]) -> str:
    """Format safety check results for Discord display."""
    if not blocked and not warnings:
        return "🛡️ Safety: **Passed** (no issues)."

    lines = []
    if blocked:
        lines.append(f"🚨 **BLOCKED TERMS DETECTED:** `{', '.join(blocked)}`")
    if warnings:
        lines.append(f"⚠️ Soft warnings: `{', '.join(warnings)}` — review carefully.")

    return "\n".join(lines)