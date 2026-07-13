"""
Claw Bot — facts memory

What the bot has ALREADY told people, so it never says it twice.

Ask for "octopuses" three times and Qwen writes the same three facts three times:
they are the famous ones, and a model reaching for the most surprising fact about
a topic reaches for the same one every run. That is fine the first time and
useless the third — the second reel is a re-upload with new pictures.

So every fact that ships is written down here, keyed by topic, and the next reel
about that topic is asked for facts that are NOT in the list. What comes back is
checked as well as asked for: an LLM told "don't repeat these" will happily
reword one, and a reworded fact is the same fact. The check is on MEANING-ish
tokens (content words, order-free), not on the sentence.

    04_Outputs/facts/_memory.json
    { "topics": { "octopuses": [ {fact, facts_id, at}, ... ] } }

Deleting a reel does NOT forget its facts. If a reel was bad you want the next
one to be DIFFERENT, so what it said stays on the do-not-repeat list — the
memory is about the facts, not the file.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from modules.file_utils import atomic_write_json

log = logging.getLogger("claw_bot.facts_memory")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
MEMORY_PATH = PROJECT_ROOT / "04_Outputs" / "facts" / "_memory.json"

# How many past facts are shown to the writer. All of them are still checked
# against — this only bounds the PROMPT, which the model stops reading somewhere.
PROMPT_LIMIT = 60

# Two facts are "the same" when their content words mostly agree. Measured:
#   "Octopuses have three hearts"           vs "An octopus has three hearts"     -> 1.00
#   "Octopuses have three hearts"           vs "Octopuses, it turns out, have
#                                               three hearts"                     -> 0.75
#   "Bees flap their wings 230 times a sec" vs "A bee beats its wings 230 times
#                                               per second"                       -> 0.71
#   "Octopuses have three hearts"           vs "Two of an octopus's hearts stop
#                                               beating when it swims"            -> 0.25
#   "Octopuses have three hearts"           vs "Octopuses have nine brains"       -> 0.20
# 0.55 catches every rewording above and lets both genuinely new facts through.
SIMILARITY = 0.55

# Words that carry no fact. Numbers are deliberately KEPT — "three hearts" and
# "nine brains" differ by the number, and dropping it merges them.
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "has", "have",
    "had", "do", "does", "did", "can", "could", "will", "would", "of", "in",
    "on", "at", "to", "for", "with", "from", "by", "as", "its", "it", "their",
    "them", "they", "you", "your", "and", "or", "but", "that", "this", "these",
    "those", "than", "then", "so", "up", "out", "about", "into", "over", "per",
    "each", "every", "some", "most", "much", "many", "more", "one", "only",
    "even", "just", "actually", "really", "very", "almost", "nearly", "around",
    "about", "s",
}


def _load() -> dict:
    if not MEMORY_PATH.exists():
        return {"topics": {}}
    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("topics"), dict):
            return data
        log.warning("facts memory malformed; starting a fresh one")
    except Exception as e:
        log.warning(f"facts memory unreadable ({e}); starting a fresh one")
    return {"topics": {}}


def _save(data: dict) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(MEMORY_PATH, data)


def _singular(word: str) -> str:
    """Fold a plural onto its singular, so 'octopus' and 'octopuses' are one word.

    Naively stripping /(es|s)$/ does NOT do that: it turns 'octopuses' into
    'octopus' and 'octopus' into 'octopu', so the two never meet — which is how
    the memory looked up the wrong topic and let a repeat through. The -us / -ss /
    -is endings are singulars already and must be left alone.
    """
    w = word or ""
    if len(w) <= 3 or w.isdigit():
        return w
    if w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is", "as")):
        return w[:-1]
    return w


def topic_key(topic: str) -> str:
    """'The Deep Ocean!' and 'deep ocean' are one topic.

    Deliberately crude: plurals fold, nothing else does. 'octopus' and
    'cephalopods' are NOT the same topic, and pretending otherwise would silently
    starve a legitimate new reel of facts it never told.
    """
    words = [_singular(w) for w in re.findall(r"[a-z0-9]+", (topic or "").lower())
             if w not in _STOP]
    return " ".join(words) or (topic or "").strip().lower()


def _tokens(fact: str) -> set:
    return {_singular(w) for w in re.findall(r"[a-z0-9]+", (fact or "").lower())
            if w not in _STOP}


def similarity(a: str, b: str) -> float:
    """Order-free overlap of content words. 1.0 = the same fact, reworded."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_repeat(fact: str, prior: list) -> Optional[str]:
    """The earlier fact this one repeats, or None.

    Returns the MATCH, not a bool, so the caller can tell the writer exactly what
    it just said again — "you repeated X" beats "try harder".
    """
    for old in prior or []:
        if similarity(fact, old) >= SIMILARITY:
            return old
    return None


def seen_facts(topic: str, limit: int = 0) -> list:
    """Every fact already told about this topic, newest last."""
    entries = _load()["topics"].get(topic_key(topic), [])
    facts = [e.get("fact", "") for e in entries if e.get("fact")]
    return facts[-limit:] if limit else facts


def all_topics() -> dict:
    """{topic key: how many facts told} — for the dashboard/`!facts_memory`."""
    return {k: len(v) for k, v in _load()["topics"].items()}


def record(topic: str, facts_id: str, facts: list) -> int:
    """Write down the facts a reel just used. Returns how many were new.

    A fact already on the list is not stored twice — a re-render of the same reel
    must not double it up, and the prompt has a budget.
    """
    data = _load()
    key = topic_key(topic)
    entries = data["topics"].setdefault(key, [])
    known = [e.get("fact", "") for e in entries]
    added = 0
    now = datetime.now().isoformat(timespec="seconds")
    for f in facts:
        f = (f or "").strip()
        if not f or is_repeat(f, known):
            continue
        entries.append({"fact": f, "facts_id": facts_id, "at": now})
        known.append(f)
        added += 1
    if added:
        _save(data)
        log.info(f"facts memory: +{added} for '{key}' "
                 f"({len(entries)} known about it now)")
    return added


def forget_reel(facts_id: str) -> int:
    """Drop everything one reel contributed. Returns how many facts were dropped.

    NOT called when a reel is deleted — a bad reel's facts stay on the list so the
    next one about that topic says something else. This exists for the case where
    the memory itself is wrong (a placeholder run, a test) and is poisoning the
    prompt.
    """
    data = _load()
    dropped = 0
    for key, entries in list(data["topics"].items()):
        keep = [e for e in entries if e.get("facts_id") != facts_id]
        dropped += len(entries) - len(keep)
        if keep:
            data["topics"][key] = keep
        else:
            data["topics"].pop(key)
    if dropped:
        _save(data)
        log.info(f"facts memory: forgot {dropped} fact(s) from {facts_id}")
    return dropped


def forget_topic(topic: str) -> int:
    """Wipe one topic's history — say it all again from scratch."""
    data = _load()
    entries = data["topics"].pop(topic_key(topic), [])
    if entries:
        _save(data)
        log.info(f"facts memory: forgot all {len(entries)} facts about "
                 f"'{topic_key(topic)}'")
    return len(entries)


def backfill_from_reels() -> int:
    """Teach the memory what the reels ALREADY on disk said. Runs once.

    Without this, the memory starts empty on a machine with 31 reels in it, and
    the first "octopuses" reel after the feature ships happily repeats the
    octopus reel from last week — which is the exact thing this module exists to
    stop. Placeholder reels are skipped: their narration is not a fact.
    """
    data = _load()
    if data.get("_backfilled"):
        return 0

    facts_dir = PROJECT_ROOT / "04_Outputs" / "facts"
    added = 0
    for p in sorted(facts_dir.glob("facts_*.json")) if facts_dir.is_dir() else []:
        try:
            story = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"backfill: {p.name} unreadable ({e})")
            continue
        if story.get("_placeholder") or not story.get("topic"):
            continue
        key = topic_key(story["topic"])
        entries = data["topics"].setdefault(key, [])
        known = [e.get("fact", "") for e in entries]
        for b in story.get("beats", []):
            if b.get("kind") != "fact":
                continue
            fact = (b.get("narration") or "").strip()
            if not fact or is_repeat(fact, known):
                continue
            entries.append({"fact": fact,
                            "facts_id": story.get("facts_id", p.stem),
                            "at": story.get("_generated_at", "")})
            known.append(fact)
            added += 1

    data["_backfilled"] = True
    _save(data)
    log.info(f"facts memory: backfilled {added} fact(s) from reels already on disk "
             f"({len(data['topics'])} topics)")
    return added


def avoid_clause(topic: str) -> str:
    """The 'already said this' block for the writer's prompt, or ''."""
    prior = seen_facts(topic, limit=PROMPT_LIMIT)
    if not prior:
        return ""
    lines = "\n".join(f"- {f}" for f in prior)
    return (
        f"\n\nYou have ALREADY made reels about this topic. These facts were used "
        f"and must NOT appear again — not the same fact in different words, not a "
        f"different number for the same claim:\n{lines}\n"
        f"Every fact you write now must be genuinely NEW: a different mechanism, "
        f"a different body part, a different behaviour, a different record. If the "
        f"obvious facts are taken, go deeper — the strange ones are better anyway."
    )
