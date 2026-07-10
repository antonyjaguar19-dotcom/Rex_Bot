"""
Claw Bot — Facts Shorts Writer

Writes a short, punchy "did-you-know" facts reel with Qwen. NO characters, NO
plot — just a hook + N standalone facts + a call-to-action outro. Because there
is no recurring subject, the pipeline has zero character-consistency burden; the
image is only a loose mood backdrop behind big on-screen text + narration.

Output JSON (04_Outputs/facts/facts_{id}.json), shaped so the facts pipeline can
treat each entry as a `beat` exactly like the horror pipeline does:
  { title, topic, beats: [ {narration, on_screen, image_prompt, kind}, ... ] }

  narration   — spoken line (kokoro voices it; its real audio span = scene length)
  on_screen   — the BIG centered caption (short; the fact's punchy core)
  image_prompt— a loose, text-free background prompt (mood only, never the subject)
  kind        — "hook" | "fact" | "outro"
"""

import json
import logging
import re
import sys
import time as _t
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules import safety_filter as sf
from modules.file_utils import atomic_write_json
from modules.script_generator import _call_llm, _extract_json

log = logging.getLogger("claw_bot.facts_writer")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
OUTPUTS_DIR = PROJECT_ROOT / "04_Outputs" / "facts"

DEFAULT_N_FACTS = 6
MIN_FACTS = 4
MAX_FACTS = 8

_SYS = (
    "You are a viral short-form video scriptwriter. You write tight, accurate, "
    "surprising 'did you know' facts reels for YouTube Shorts / TikTok / Reels. "
    "Every fact must be TRUE and verifiable — never invent numbers. Punchy, "
    "conversational, no fluff. Output ONLY valid JSON, no prose around it."
)


def _prompt(topic: str, n: int) -> str:
    return (
        f"Write a facts reel about: {topic}.\n\n"
        f"Return JSON with EXACTLY this shape:\n"
        f"{{\n"
        f'  "title": "short scroll-stopping title (max 8 words)",\n'
        f'  "hook": "a GENERAL teaser opening line (max 16 words). Do NOT state a specific fact — just promise the reel (e.g. \'Here are {n} things about X you won\'t believe\'). Never claim something not in the facts below.",\n'
        f'  "facts": [\n'
        f'     {{"spoken": "the fact as one spoken sentence (12-28 words), surprising and TRUE",\n'
        f'       "caption": "the same fact boiled to a punchy 3-8 word on-screen line",\n'
        f'       "backdrop": "a CREATIVE, eye-catching, playful photographic scene that makes this fact\'s IDEA memorable. Stay on-topic but be imaginative — anthropomorphize or add a fun prop/costume (e.g. intelligence -> an octopus wearing a tiny scholar graduation cap solving a puzzle; venom -> a menacing glowing blue-ringed octopus; camouflage -> an octopus half-vanished into coral). Vivid, whimsical, cinematic. No text/words in the image."}}\n'
        f"  ],\n"
        f'  "outro": "one spoken call-to-action (e.g. follow for more), max 12 words",\n'
        f'  "description": "a catchy 2-3 sentence video description for YouTube Shorts / TikTok / Instagram Reels that teases the facts and drives follows. End it with a NEW line containing 10-14 relevant lowercase hashtags separated by spaces (mix topic-specific + broad like #facts #shorts #reels #didyouknow)."\n'
        f"}}\n\n"
        f"Give EXACTLY {n} facts. Order them weakest-to-strongest so the best fact is last. "
        f"Keep every 'caption' SHORT — it is displayed as large centered text. "
        f"Every 'backdrop' must clearly ILLUSTRATE its fact (relevant, not abstract)."
    )


def _fallback(topic: str, n: int) -> dict:
    """Deterministic minimal reel when the LLM is unavailable — keeps the pipeline
    testable offline. Facts are intentionally generic placeholders."""
    facts = []
    for i in range(n):
        facts.append({
            "spoken": f"Here is an interesting thing about {topic} number {i+1}.",
            "caption": f"{topic.title()} · Fact {i+1}",
            "backdrop": f"abstract cinematic background evoking {topic}, soft depth of field, no text",
        })
    return {
        "title": f"{topic.title()} Facts",
        "hook": f"{n} things you didn't know about {topic}.",
        "facts": facts,
        "outro": "Follow for more.",
    }


def _to_beats(data: dict, topic: str) -> list[dict]:
    beats: list[dict] = []
    hook = (data.get("hook") or "").strip()
    if hook:
        beats.append({"kind": "hook", "narration": hook,
                      "on_screen": (data.get("title") or hook)[:60],
                      "image_prompt": f"a striking, dramatic cinematic photograph of {topic}, "
                                      f"vivid detail, dynamic lighting, no text, no words"})
    for i, f in enumerate(data.get("facts", [])):
        spoken = (f.get("spoken") or "").strip()
        if not spoken:
            continue
        beats.append({
            "kind": "fact",
            "narration": spoken,
            "on_screen": (f.get("caption") or spoken)[:80],
            "image_prompt": (f.get("backdrop") or f"a cinematic photograph of {topic}")
                             + f" — imaginative and eye-catching, on-topic to {topic}, "
                               f"cinematic photograph, sharp, vivid, highly detailed, "
                               f"no text, no words",
            "index": i + 1,
        })
    outro = (data.get("outro") or "").strip()
    if outro:
        beats.append({"kind": "outro", "narration": outro,
                      "on_screen": outro[:40],
                      "image_prompt": f"a beautiful cinematic photograph of {topic}, "
                                      f"vivid, detailed, no text, no words"})
    return beats


def _hashtags(topic: str) -> str:
    tags = ["#" + w for w in re.findall(r"[a-z0-9]+", topic.lower()) if len(w) > 2]
    tags += ["#facts", "#didyouknow", "#shorts", "#reels", "#trivia",
             "#viral", "#funfacts", "#learnontiktok"]
    seen, out = set(), []
    for t in tags:
        if t not in seen:
            seen.add(t); out.append(t)
    return " ".join(out[:14])


def _social_description(data: dict, title: str, topic: str, beats: list) -> str:
    """Ready-to-paste upload description. Uses the LLM's description if given,
    else builds one from the title/hook/facts + hashtags."""
    d = (data.get("description") or "").strip()
    if d:
        return d
    hook = next((b.get("narration", "") for b in beats if b.get("kind") == "hook"), "")
    n = len([b for b in beats if b.get("kind") == "fact"])
    lines = [title, "",
             hook or f"{n} surprising facts about {topic} you didn't know!", "",
             "In this short:"]
    lines += [f"• {b.get('on_screen') or b.get('narration', '')}"
              for b in beats if b.get("kind") == "fact"]
    lines += ["", "Follow for more 🔔", "", _hashtags(topic)]
    return "\n".join(lines)


class FactsUnavailable(RuntimeError):
    """The LLM could not write real facts. Raised instead of quietly shipping
    placeholder narration ("Here is an interesting thing about bees number 1")."""


def generate_facts_short(
    topic: str,
    n_facts: int = DEFAULT_N_FACTS,
    progress_cb: Optional[Callable[[str], None]] = None,
    allow_placeholder: bool = False,
) -> dict:
    """Write + save a facts reel for `topic`. Returns the story dict.

    If the LLM is unreachable or returns junk this RAISES. It used to fall back
    to `_fallback()` silently, so a reel rendered with Ollama down narrated
    "Here is an interesting thing about honeybees number 1" — a full render,
    voiced, subtitled and posted, with placeholder text.

    `allow_placeholder=True` restores the old behaviour for offline tests.
    """
    t0 = _t.time()
    n = max(MIN_FACTS, min(int(n_facts or DEFAULT_N_FACTS), MAX_FACTS))
    if progress_cb:
        progress_cb(f"writing {n} facts about {topic}...")
    placeholder = False
    try:
        raw = _call_llm(_prompt(topic, n), _SYS, role="creative")
        data = _extract_json(raw)
        if not data.get("facts"):
            raise ValueError("LLM returned no facts")
    except Exception as e:
        if not allow_placeholder:
            raise FactsUnavailable(
                f"Could not write facts about '{topic}': {e}. "
                f"Is Ollama running? (nothing was rendered)"
            ) from e
        log.warning(f"Facts LLM failed ({e}); using offline placeholder.")
        data = _fallback(topic, n)
        placeholder = True

    beats = _to_beats(data, topic)
    if len([b for b in beats if b["kind"] == "fact"]) < MIN_FACTS:
        if not allow_placeholder:
            raise FactsUnavailable(
                f"LLM wrote only {len([b for b in beats if b['kind'] == 'fact'])} "
                f"usable facts about '{topic}' (need {MIN_FACTS}). Nothing rendered."
            )
        data = _fallback(topic, n)
        beats = _to_beats(data, topic)
        placeholder = True

    now = datetime.now()
    facts_id = now.strftime("%Y%m%d_%H%M%S")
    title = (data.get("title") or f"{topic.title()} Facts").strip()
    story = {
        "facts_id": facts_id,
        "_id": facts_id,
        "title": title,
        "topic": topic,
        "beats": beats,
        "characters": [],   # none — that's the whole point
        "locations": [],
        "description": _social_description(data, title, topic, beats),
        "_generated_at": now.isoformat(),
        # True only when allow_placeholder let the offline stub through, so a
        # stub reel can never be mistaken for a real one on disk.
        "_placeholder": placeholder,
    }
    if placeholder:
        log.warning(f"Facts reel {facts_id} contains PLACEHOLDER narration "
                    f"(LLM unavailable) — do not publish.")

    # Safety is ADVISORY for facts (educational): the kids word-list flags benign
    # science terms ("blood", etc.). Log, never block. (A real profanity/harm gate
    # can be added later if facts topics get user-supplied.)
    try:
        is_safe, blocked, _warn = sf.check_safety(story)
        if not is_safe:
            log.info(f"Facts soft safety flags (ignored): {blocked}")
    except Exception:
        pass

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(OUTPUTS_DIR / f"facts_{facts_id}.json", story)
    if progress_cb:
        progress_cb(f"facts written: '{story['title']}' — {len(beats)} beats "
                    f"in {_t.time()-t0:.0f}s")
    log.info(f"Facts reel saved: facts_{facts_id}.json ({len(beats)} beats)")
    return story


def load_facts(facts_id: str) -> Optional[dict]:
    p = OUTPUTS_DIR / f"facts_{facts_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
