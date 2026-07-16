"""
Claw Bot — lesson mode: writing the lesson

A topic from the textbook goes in; a script a six-year-old can follow comes out.

Two stages, the same shape as the story pipeline (`story_writer` → the structurer):
first the lesson is written as PROSE for a child, then it is cut into beats. Doing it
in one pass gives you a list of facts with a mascot in front of them — a lesson has to
build: here is the thing, here is what it does, here is one you know from home, now
check you got it.

Three rules this file exists to enforce:

**The source is not all the book's words.** A page read by the vision model looks like

    Brush your teeth twice a day.
    [picture: a child standing at a basin brushing their teeth]

and that second line is a machine's description, not the textbook's text. The writer
is told so. It may teach from it — on a Class-1 page the picture IS most of the lesson
— but it may never quote it as the book's words.

**A thin topic is refused, not padded.** "Flowers and Their Petals" is 53 words in the
whole book. A ninety-second lesson out of that is ninety percent invention with a
textbook's name on it. Merge it with a neighbour instead.

**Length is set in WORDS, here, and never by speeding the voice up later.** The cloned
mascot reads ~1.9 words/sec, so a 90-second lesson is about 160 spoken words. Nothing
downstream is allowed to squeeze a lesson into a facts reel's 40-second ceiling.
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

from modules import lesson_book as lb
from modules import lesson_topics as lt
from modules.file_utils import atomic_write_json
from modules.script_generator import _call_llm, _extract_json

log = logging.getLogger("claw_bot.lesson_writer")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
LESSONS_DIR = PROJECT_ROOT / "04_Outputs" / "lessons"

# A Class-1 lesson. Long enough to teach one idea properly, short enough that it is
# still the BOOK being taught: the source for a topic is a few hundred words, and a
# five-minute lesson out of that is mostly the model talking.
MIN_SECONDS = 45.0
MAX_SECONDS = 180.0
DEFAULT_SECONDS = 90.0

# Measured on real reels: the cloned mascot reads 1.7-1.9 words/sec, and every beat
# carries ~0.7s of head/tail padding (facts_pipeline.PAD_HEAD_SEC + PAD_TAIL_SEC).
WORDS_PER_SEC = 1.9
PAD_PER_BEAT = 0.70

# One spoken line per beat. Shorter than this and the mascot is barking; longer and a
# six-year-old has lost the thread before the sentence lands.
WORDS_PER_BEAT = 16

# Below this, the book has not said enough for a lesson to be ABOUT the book.
MIN_SOURCE_WORDS = 120

# The source text of one topic, capped so the prompt fits num_ctx (8192 ≈ 28k chars,
# and the prose answer has to live in there too).
MAX_SOURCE_CHARS = 8000

LLM_ATTEMPTS = 3

# How far from the word budget a draft may land before it is re-rolled. The voice
# turns words into seconds at a fixed rate, so ±15% of the words is ±15% of the
# lesson — 90s becomes 77-104s, which is fine, and 150s is not.
LENGTH_TOLERANCE = 0.15


class LessonUnavailable(RuntimeError):
    """The lesson could not be written. Raised instead of saving a stub: a
    placeholder script would be rendered, voiced and posted like a real one."""


class TopicTooThin(LessonUnavailable):
    """The book says too little about this topic to teach it honestly."""


def word_budget(target_sec: float, n_beats: int) -> int:
    """How many spoken words fit in `target_sec`, once the padding is paid for."""
    speech = max(5.0, float(target_sec) - PAD_PER_BEAT * max(1, n_beats))
    return int(speech * WORDS_PER_SEC)


def plan_shape(target_sec: float) -> tuple:
    """(beats, words) for a lesson of this length."""
    target_sec = max(MIN_SECONDS, min(float(target_sec), MAX_SECONDS))
    beats = max(6, round(target_sec * WORDS_PER_SEC / WORDS_PER_BEAT))
    return beats, word_budget(target_sec, beats)


_PROSE_SYS = (
    "You are a warm, patient primary-school teacher explaining ONE topic to a child of "
    "six. You are teaching from their own textbook, and you teach what IT teaches.\n"
    "Write PLAIN PROSE. No headings, no bullet points, no JSON, no stage directions.\n"
    "Shape the lesson like this, in order:\n"
    "  1. A hook: one friendly line that makes the child want to know.\n"
    "  2. What the thing IS, in the simplest words that are still true.\n"
    "  3. Two to four small ideas from the book, one at a time. Give each ONE example "
    "from a child's own day — home, school, food, play, their own body.\n"
    "  4. One question to check they got it, and then the answer.\n"
    "  5. A short recap of what they learned.\n"
    "Rules:\n"
    "- Words a six-year-old knows. One idea per sentence. Short sentences.\n"
    "- Never use a word the book does not use without explaining it in the same breath.\n"
    "- Teach what the book teaches. Do not bring in facts the book never mentions.\n"
    "- If the book gives a LIST — the ways a thing is alive, the parts of a plant, the "
    "senses, the steps — teach EVERY item on it. Leaving one out hands the child half a "
    "definition. The dull item still counts: 'living things breathe' matters as much as "
    "'living things eat', even though breathing has no fun example.\n"
    "- Warm and encouraging. Speak TO the child ('you', 'your'), never about them.\n"
    "- No numbers, statistics or long names. This is a six-year-old.\n"
    "- Do NOT mention the textbook, the page, the pictures, or the exercises. The child "
    "is being taught the subject, not read a book report."
)

_PICTURE_NOTE = (
    "\n\nIMPORTANT — about the source below. Lines that begin with `[picture:` are a "
    "MACHINE'S DESCRIPTION of an illustration on the page. They are NOT the textbook's "
    "words. Use them to know what the child is being shown (on a page for six-year-olds "
    "the picture is most of the lesson) — but never quote them, and never write "
    "`[picture:` in your answer."
)

# The lesson is cut into beats MECHANICALLY, by sentence — the model never touches the
# narration again after the prose stage.
#
# It was an LLM job at first, and it failed three ways in a row on the real book. Asked
# to "cut it up, don't rewrite", it silently dropped a third of the teaching (a 90s
# lesson came back at 59s). Told to keep everything, it obeyed the beat count by
# repeating itself — three recaps, then more teaching after them. Told the exact number
# of beats, it started compressing sentences to fit: "Hello there! Amazing bodies
# today's topic." That is not a sentence you can read to a six-year-old.
#
# The prose is already correct: it teaches the book, it is grammatical, and it is the
# right length. Splitting it on sentence boundaries cannot lose a word, cannot invent
# one, and cannot mangle the grammar. The model is then asked only for what it is
# actually good at: the caption and the picture for each line.
MIN_BEAT_WORDS = 7
MAX_BEAT_WORDS = 22

_DRESS_SYS = (
    "For each numbered line of a children's lesson you write two things.\n"
    'Output ONLY valid JSON: {"title": "...", "beats": [{"n": 1, "on_screen": "...", '
    '"image_prompt": "..."}], "objects": [{"noun": "...", "desc": "..."}]}\n'
    "Rules:\n"
    "- n: the NUMBER of the line this entry is for. Every line gets exactly one entry, "
    "and the number must match.\n"
    "- on_screen: 2-4 words, the idea of THAT line, for big text on screen. Not a "
    "sentence.\n"
    "- image_prompt: a friendly cartoon scene a small child would like, showing ONE "
    "action with ONE object, matching that line. No text, letters or numbers in the "
    "image.\n"
    "- Never change, quote or repeat the lines themselves.\n"
    "- title: 3-6 words, what the whole lesson is called.\n"
    "- objects: the concrete THINGS this lesson keeps coming back to across several lines "
    "— the doll and the puppy in a lesson about what is alive, the plant in a lesson about "
    "growing. For EACH, give a single common noun ('doll', 'puppy') and ONE vivid fixed "
    "description of exactly what it looks like, so the SAME one can be drawn every time it "
    "appears: 'a soft cuddly cloth rag doll with a friendly embroidered smile, round felt "
    "eyes and floppy cream arms'. Make every object CUTE, soft and friendly for a small "
    "child — never scary or creepy (no sharp black button eyes, no stitched-shut mouth). "
    "Only things that actually appear in the lines, at most three. A one-off prop mentioned "
    "in a single line is NOT one of these."
)


def _source_for(book_id: str, topic_id: str) -> tuple:
    """(the book's material for this topic, its word count). Raises when too thin."""
    book = lb.load_book(book_id)
    if not book:
        raise LessonUnavailable(f"no book {book_id!r}")
    topic = next((t for t in book.get("topics", []) if t.get("id") == topic_id), None)
    if not topic:
        raise LessonUnavailable(f"no topic {topic_id!r} in '{book['title']}'")

    text = lt.topic_text(book_id, topic_id)
    words = len(text.split())
    if words < MIN_SOURCE_WORDS:
        raise TopicTooThin(
            f"'{topic['title']}' is only {words} words in the book "
            f"(pages {topic['first_page']}-{topic['last_page']}). A lesson out of that "
            f"would be mostly invented, with the textbook's name on it. Merge it with "
            f"the topic next to it, or add its pages, and try again."
        )
    return text[:MAX_SOURCE_CHARS], words, topic, book


def _prose_ask(topic: dict, source: str, budget: int) -> str:
    """The instruction the prose writer is given. Shared by the first draft and by the
    coverage re-roll, so a re-roll asks for the same lesson — just without the gap."""
    return (f"Topic: {topic['title']}\n"
            f"What the child's textbook says about it:\n---\n{source}\n---"
            f"{_PICTURE_NOTE}\n\n"
            f"Teach this topic to a six-year-old, out loud, in about {budget} words.")


def _write_prose(topic: dict, source: str, budget: int, _p) -> str:
    """Stage 1: the lesson, as a teacher would say it. Prose, no structure."""
    ask = _prose_ask(topic, source, budget)

    best, best_gap, last = "", 10**9, None
    for attempt in range(1, LLM_ATTEMPTS + 1):
        try:
            raw = _call_llm(ask, _PROSE_SYS, role="creative")
        except Exception as e:
            # A dead Ollama must come out as LessonUnavailable, not a bare
            # RuntimeError — the caller has one exception to catch, and a lesson
            # that failed for ANY reason must leave nothing on disk.
            last = e
            _p(f"⚠️ the model did not answer ({e})")
            continue
        prose = _clean_prose(raw)
        n = len(prose.split())
        if not n:
            continue
        gap = abs(n - budget)
        if gap < best_gap:
            best, best_gap = prose, gap
        if abs(n - budget) <= budget * LENGTH_TOLERANCE:
            _p(f"✍️ lesson written — {n} words (asked {budget})")
            return prose

        # Say WHICH WAY it missed. "Try again" gets the same length back.
        how = ("too long — cut the least important idea and shorten every sentence"
               if n > budget else
               "too short — add one more small idea from the book, with an example")
        _p(f"✍️ draft {attempt}: {n} words, {how.split(' —')[0]}; re-rolling")
        ask = (f"{ask}\n\nYour last draft was {n} words. It is {how}. "
               f"Write it again in about {budget} words.")

    if not best:
        raise LessonUnavailable(
            f"the lesson could not be written: {last or 'the model wrote nothing usable'}. "
            f"Nothing was saved.")
    _p(f"✍️ lesson written — {len(best.split())} words (asked {budget})")
    return best


# ==============================================================================
# COVERAGE — did the lesson keep everything the book teaches?
# ==============================================================================
#
# The length loop above optimises WORDS, not completeness, and to hit a word budget the
# model drops the least vivid idea. On the real Class-1 book, "Living and Non-living
# Things", the page lists SIX ways a thing is alive — eat, grow, move, have young ones,
# BREATHE, feel — and breathing is the one with no cheerful child's-day example, so it
# went. The lesson taught five of six, its own recap claimed to have taught the whole
# definition, and nothing errored: the child then meets "Does it breathe?" on the book's
# own self-test, untaught.
#
# The prompt now ASKS for the whole list (a rule in _PROSE_SYS). This is the CHECK — the
# same doctrine the fact-memory and the subscribe outro are built on: a model told "keep
# everything" still silently drops one, so the prompt is the mechanism and the check is
# the feature.

COVERAGE_ATTEMPTS = 2

_COVERAGE_SYS = (
    "You check whether a lesson written for a six-year-old left out anything important "
    "the textbook teaches.\n"
    "You are given (a) what the book says about a topic and (b) the lesson written from "
    "it. Find KEY THINGS the book clearly teaches about this topic that the lesson does "
    "not teach AT ALL.\n"
    'Answer with JSON only: {"missing": ["...", "..."]}\n'
    "- A thing is MISSING only if the lesson never teaches it in ANY words. If the "
    "lesson says the same thing differently — the book says 'have young ones', the "
    "lesson says 'have babies' — it is NOT missing.\n"
    "- Only list things the BOOK itself teaches. Never invent a fact the book does not "
    "mention.\n"
    "- When the book gives a LIST — the ways living things are alive, the parts of a "
    "plant, the senses — every item on it matters, and a dropped item is missing.\n"
    "- Each entry is 1-3 words: the thing itself ('breathe', 'roots'), not a sentence.\n"
    '- Nothing left out? Answer {"missing": []}.'
)


def _missing_key_ideas(source: str, prose: str, _p) -> list:
    """Things the book teaches that the lesson dropped ENTIRELY.

    Guarded against the checker itself: an item is kept only if its word is ACTUALLY in
    the book and ACTUALLY absent from the lesson. That stops a hallucinated requirement
    ('photosynthesis' on a page that never says it) from re-rolling forever, and folds
    away a synonym the lesson already taught (the substring match forgives feel/feels)."""
    try:
        raw = _call_llm(
            f"What the book says:\n---\n{source}\n---\n\n"
            f"The lesson written from it:\n---\n{prose}\n---\n\n"
            f"List anything important the lesson leaves out.",
            _COVERAGE_SYS, role="structurer")
        items = (_extract_json(raw) or {}).get("missing") or []
    except Exception as e:
        _p(f"⚠️ could not check the lesson for gaps ({e})")
        return []

    src_low, prose_low = source.lower(), prose.lower()
    kept = []
    for it in items:
        head = re.sub(r"[^a-z]", "", str(it).lower().split()[0]) if str(it).strip() else ""
        if len(head) >= 3 and head in src_low and head not in prose_low:
            kept.append(str(it).strip())
    return kept


def _close_gaps(topic: dict, source: str, prose: str, budget: int, _p) -> str:
    """Re-roll the lesson until it teaches everything the book does — or say what it
    still misses. Never raises: a lesson with a gap is worse than one without, but a
    lesson that could not be re-rolled is not worse than no lesson."""
    for _ in range(COVERAGE_ATTEMPTS):
        missing = _missing_key_ideas(source, prose, _p)
        if not missing:
            return prose
        joined = ", ".join(missing)
        _p(f"🔎 the lesson left out: {joined} — the book teaches it; re-rolling")
        add = (f"\n\nYour last lesson left out something the book teaches: {joined}. "
               f"Write the lesson again, still about {budget} words, and this time teach "
               f"{joined} too — each as one small idea with an example a child knows.")
        try:
            got = _clean_prose(_call_llm(_prose_ask(topic, source, budget) + add,
                                         _PROSE_SYS, role="creative"))
        except Exception as e:
            _p(f"⚠️ could not re-roll for the gap ({e}); teaching what we have")
            return prose
        if got:
            prose = got

    left = _missing_key_ideas(source, prose, _p)
    if left:
        # No silent downgrade. A lesson short of the book's own list ships with a warning
        # on screen, not quietly.
        _p(f"⚠️ this lesson may still not cover: {', '.join(left)} — the book says more "
           f"about it than the lesson does. Check it, or add a beat for it by hand.")
    return prose


_HEADING = re.compile(r"^\s*(#+|\d+[.)]|[-*•])\s*", re.M)


def _clean_prose(raw: str) -> str:
    """Strip the shapes the model reaches for when it forgets it was asked for prose."""
    text = (raw or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = _HEADING.sub("", text)
    # The picture marker is the book's provenance tag. It must never be spoken.
    text = re.sub(r"\[picture:[^\]]*\]", "", text, flags=re.I)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_VALID_KINDS = ("intro", "teach", "example", "check", "recap", "outro")


def split_into_lines(prose: str) -> list:
    """The lesson's own sentences, grouped into speakable lines.

    Mechanical on purpose — see the note above `_DRESS_SYS`. A sentence is never
    reworded; a very long one is split at a comma so nobody has to hold twenty-two
    words in their head; a very short one is joined to its neighbour so the mascot
    is not barking three-word fragments.
    """
    from modules.subtitles import raw_sentences

    lines: list = []
    for sent in raw_sentences(_clean_prose(prose)):
        sent = sent.strip()
        if not sent:
            continue
        words = sent.split()

        if len(words) > MAX_BEAT_WORDS:
            # Split at the comma nearest the middle — a natural breath, and the two
            # halves are still the lesson's own words.
            commas = [i for i, w in enumerate(words) if w.endswith(",")]
            if commas:
                cut = min(commas, key=lambda i: abs(i - len(words) // 2)) + 1
            else:
                cut = len(words) // 2
            parts = [" ".join(words[:cut]).rstrip(","), " ".join(words[cut:])]
        else:
            parts = [sent]

        for part in parts:
            if (lines and len(part.split()) < MIN_BEAT_WORDS
                    and len(lines[-1].split()) + len(part.split()) <= MAX_BEAT_WORDS):
                lines[-1] = f"{lines[-1]} {part}"
            else:
                lines.append(part)

    # A short line joins its NEIGHBOUR — and the first line has no previous one to join,
    # so it was left standing alone. Measured: a real lesson opened with "Hey there!",
    # two words, which then got a whole shot of its own — a still, and eight minutes of
    # Wan if you ticked it. Merge it forward instead.
    while (len(lines) > 1 and len(lines[0].split()) < MIN_BEAT_WORDS
           and len(lines[0].split()) + len(lines[1].split()) <= MAX_BEAT_WORDS):
        lines[0] = f"{lines[0]} {lines[1]}"
        del lines[1]
    return lines


def _kinds_for(lines: list) -> list:
    """intro … teach … check … outro.

    A lesson has ONE check — the moment it asks the child whether they got it. Marking
    every line with a "?" as a check gave a real lesson THREE, because a teacher asks
    rhetorical questions all the way through ("Do you know that everything isn't
    alive?"). The real check is the LAST question asked, right before the recap.
    """
    kinds = ["teach"] * len(lines)
    kinds[0] = "intro"
    # The last spoken line of the LESSON is the recap — the part that actually teaches.
    # It used to be labelled "outro" purely because it came last, which meant the lesson
    # had no ending, only a stop. The real outro is a beat of its own now (ensure_outro).
    if len(lines) > 1:
        kinds[-1] = "recap"

    questions = [i for i in range(1, len(lines) - 1) if "?" in lines[i]]
    if questions:
        kinds[questions[-1]] = "check"
    return kinds


# ==============================================================================
# CAMERA FRAMING — a lesson is a shot list, not thirteen identical frames
# ==============================================================================
#
# Every lesson beat used to render as the SAME picture: full-body, camera-facing "mascot
# presents". Now each beat carries a framing — establishing / wide / medium / closeup —
# that changes the actual camera distance at render time (mascot.FRAMING_CLAUSE, injected
# by mascot.build_presenter_prompt). This is the storyboard grammar the kids pipeline has
# always had (prompt_assembler.SHOT_TYPE_FRAMING) and lesson mode never did.
#
# Assigned by KIND, then de-duplicated so two neighbours never share a framing (the same
# rule kids mode enforces in script_generator). closeup and establishing are left alone —
# they are already rare and load-bearing (the FACE beat, the bookend) — so only the
# medium/wide workhorses are nudged when they repeat.

# The one framing each kind wants. See mascot.FRAMING_CLAUSE for the safety reasoning:
# closeup is reserved for the question beat (the face is the shot, hands out of frame, so
# no T-pose risk); establishing is the bookend only (it is the one framing that risks both
# idle hands and identity-at-distance).
_FRAMING_BY_KIND = {
    "intro": "establishing",
    "teach": "medium",
    "example": "wide",
    "check": "closeup",
    "recap": "wide",
    "outro": "medium",
}
_DEFAULT_FRAMING = "medium"

# The two workhorse framings, nudged apart when they repeat. closeup/establishing are never
# touched (rare and deliberate), so a run of them is left as written.
_NUDGE = {"medium": "wide", "wide": "medium"}


def _framing_for(kinds: list) -> list:
    """A camera framing per beat: assigned by kind, then no two neighbours the same.

    Mechanical and deterministic — the same shape as _kinds_for. A beat whose kind is not
    in the map falls back to `medium`, which is the old implicit default (mascot injects it
    as a swap for the full-body clause, so an unknown kind renders as it does today).

    The lesson's own maps, run through the shared shot grammar so the vocabulary and the
    de-dup rule live in one place (modules/shot_grammar); behaviour is byte-for-byte the
    same as before the lift."""
    from modules import shot_grammar
    return shot_grammar.framing_for(kinds, _FRAMING_BY_KIND, _DEFAULT_FRAMING, _NUDGE)


# ==============================================================================
# THE OUTRO
# ==============================================================================

_OUTRO_SYS = (
    "You write the last line a children's lesson video says out loud.\n"
    "Answer with JSON only: {\"outro\": \"...\"}\n"
    "\n"
    "- It asks the child to SUBSCRIBE. The word 'subscribe' must be in it.\n"
    "- It is about THIS TOPIC — not a generic sign-off. Tie it to what was just taught.\n"
    "- Warm, simple, spoken to a six-year-old. 14 words at the very most.\n"
    "- No hashtags, no emoji, no channel name, no 'smash that button'.\n"
    "\n"
    "Topic 'Living and Non-living Things' -> "
    "{\"outro\": \"Subscribe, and we will find more living things together!\"}\n"
    "Topic 'My Wonderful Body' -> "
    "{\"outro\": \"Subscribe to learn what else your amazing body can do!\"}\n"
    "Topic 'Animals Around Us' -> "
    "{\"outro\": \"Subscribe, and we will meet more animals next time!\"}"
)


def _fallback_outro(topic: str) -> str:
    return f"Subscribe, and we will learn more about {topic.lower()} next time!"


def subscribe_outro(topic: str) -> str:
    """The line the lesson ENDS on. Never raises — a lesson must not lose its ending to a
    dead Ollama."""
    topic = (topic or "this").strip()
    try:
        from modules.script_generator import _call_llm, _extract_json
        raw = _call_llm(f"Topic: {topic}\n\nWrite the closing line.",
                        _OUTRO_SYS, role="creative")
        got = ((_extract_json(raw) or {}).get("outro") or "").strip()
        # The CHECK, not the prompt. A model told to say "subscribe" writes a lovely
        # sign-off that never says it — the same lesson the fact-memory taught (a model
        # told "don't repeat these" rewords the fact instead).
        if got and "subscribe" in got.lower() and len(got.split()) <= 16:
            return got
        log.info(f"outro from the model was unusable ({got!r}); using the plain one")
    except Exception as e:
        log.info(f"outro LLM failed ({e}); using the plain one")
    return _fallback_outro(topic)


def ensure_outro(lesson_id: str) -> bool:
    """Give the lesson an ending. Idempotent, so a lesson written before this existed
    gains one the next time it is prepared, without being rewritten."""
    lesson = load_lesson(lesson_id)
    if not lesson:
        return False
    beats = lesson.get("beats", [])
    if beats and beats[-1].get("kind") == "outro":
        return False                        # it already ends on one

    topic = lesson.get("topic") or lesson.get("title") or "this"
    line = subscribe_outro(topic)
    beats.append({
        "kind": "outro",
        "framing": _FRAMING_BY_KIND["outro"],
        "narration": line,
        "on_screen": "Subscribe!",
        "image_prompt": "",
        # Waving is BUSY HANDS by construction — an idle scene goes home to the reference
        # photo, and the reference photo is a T-pose (mascot.never_empty_handed).
        "mascot_scene": ("the mascot character facing the camera waving both hands high "
                         "above her head, beaming with a big happy smile"),
        "animate": False,
        "approved": False,
        "index": len(beats) + 1,
    })
    lesson["beats"] = beats
    lesson["word_count"] = sum(len(b["narration"].split()) for b in beats)
    lesson["estimated_seconds"] = round(
        lesson["word_count"] / WORDS_PER_SEC + PAD_PER_BEAT * len(beats), 1)
    _save(lesson)
    log.info(f"{lesson_id}: ends on — {line}")
    return True


def ensure_framing(lesson_id: str) -> bool:
    """Give every beat a camera framing. Idempotent, so a lesson written before framing
    existed (e.g. 20260714_113840) gains one the next time it is prepared, without being
    rewritten — and a framing you edited by hand is left exactly as you set it.

    Derives from each beat's own `kind`, through the same de-duplicating _framing_for, so a
    backfilled lesson gets the same varied shot list a fresh one would."""
    lesson = load_lesson(lesson_id)
    if not lesson:
        return False
    beats = lesson.get("beats", [])
    if not beats or all(b.get("framing") for b in beats):
        return False                        # nothing missing — every beat already framed
    framings = _framing_for([b.get("kind", "teach") for b in beats])
    filled = 0
    for b, f in zip(beats, framings):
        if not b.get("framing"):
            b["framing"] = f
            filled += 1
    _save(lesson)
    log.info(f"{lesson_id}: framing backfilled onto {filled} beat(s)")
    return True


# ==============================================================================
# SEQUENCES — the lesson as a storyboard, grouped like "Sequence 1: Introduction"
# ==============================================================================
#
# Organizational, and deliberately CHEAP: no LLM pass (same doctrine as the coverage and
# outro checks — a model asked to segment a lesson is one more silent failure). Sequences
# are computed on the fly from each beat's `kind`, so nothing is stored on disk and a
# reorder — which already moves `kind` with the beat — regroups correctly on the next read.
_SEQUENCE_TITLES = {
    "intro": "Introduction",
    "recap": "Recap",
    "outro": "Subscribe",
}


def sequence_title_for(kind: str, topic: str) -> str:
    """The sequence a beat of this kind belongs to. The teaching kinds (teach/example/
    check) all fall under the lesson's own topic, so they read as one block."""
    return _SEQUENCE_TITLES.get(kind, (topic or "The Lesson").strip() or "The Lesson")


def sequences(lesson: dict) -> list:
    """The lesson grouped into named sequences for a storyboard readout.

    Returns [{"title": str, "shots": [beat_index, ...]}], newest-first order preserved.
    Consecutive beats with the same sequence title merge into one group."""
    from modules import shot_grammar
    topic = (lesson.get("topic") or lesson.get("title") or "The Lesson").strip()
    return shot_grammar.sequences(
        lesson.get("beats", []),
        lambda b: sequence_title_for(b.get("kind", "teach"), topic))


def _dress_by_line(lines: list, dress: dict) -> dict:
    """{line index: {on_screen, image_prompt}} — matched by the line's NUMBER.

    Matching by POSITION silently shifted a real lesson: the model returned one entry
    fewer than there were lines, and from that point on every caption belonged to the
    line before it. The mascot said "they move around too" while the screen read
    "Grow Big". A caption is burned into the video — it cannot be allowed to drift.

    So each entry carries the number of the line it is for, and anything that does not
    line up is dropped rather than guessed at.
    """
    out: dict = {}
    for entry in (dress or {}).get("beats") or []:
        if not isinstance(entry, dict):
            continue
        try:
            i = int(entry.get("n", 0)) - 1          # the model counts from 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(lines) and i not in out:
            out[i] = entry
    return out


def _to_beats(lines: list, dress: dict) -> list:
    """The lesson's own lines + the model's caption and picture for each."""
    by_line = _dress_by_line(lines, dress)
    kinds = _kinds_for(lines)
    framings = _framing_for(kinds)
    beats = []
    for i, line in enumerate(lines):
        extra = by_line.get(i, {})
        # A line the model forgot still gets a caption — from its own words. Empty text
        # on screen is a hole a child sees.
        on_screen = (extra.get("on_screen") or "").strip()[:40]
        if not on_screen:
            on_screen = " ".join(line.split()[:3]).strip(" ,.!?—-").title()
        image_prompt = (extra.get("image_prompt") or "").strip()
        if not image_prompt:
            image_prompt = (f"a friendly cartoon child, {line.rstrip('.?!')}, "
                            f"bright and simple, no text")

        beats.append({
            "kind": kinds[i],
            # The camera distance this shot renders at (mascot.FRAMING_CLAUSE). Assigned by
            # kind, de-duplicated so neighbours differ. Editable, and read back with a
            # `medium` default so a lesson written before this existed renders unchanged.
            "framing": framings[i],
            "narration": line,                 # the lesson's words, untouched
            "on_screen": on_screen,
            "image_prompt": image_prompt,
            # The tickbox. OFF by default and that is deliberate: ON would mean an
            # ~8-minute Wan render for every beat the moment you pressed Render.
            "animate": False,
            # THE GATE. A picture is not confirmed until a human has LOOKED at it and
            # said so. Every serious defect this pipeline has produced — mouse ears, a
            # twin mother, a doll rendered as a living child, a headless doll, a boulder
            # — rendered cleanly, logged nothing, and was wrong only in the file. Wan
            # then spent 8.5 minutes animating it. Nothing may start that on a picture
            # nobody has seen.
            "approved": False,
            "index": i + 1,
        })

    return beats


# ==============================================================================
# RECURRING OBJECTS — the things the lesson keeps coming back to
# ==============================================================================
#
# "Living vs Non-living" is built on ONE doll and ONE puppy. If the doll is a plush rabbit
# in shot 2 and a peg doll in shot 8, the contrast the whole lesson rests on is gone. So the
# lesson pins its recurring objects and keeps them the same shot to shot (modules/
# lesson_objects). The writer PROPOSES them in _DRESS; this is the CHECK — the prompt is the
# mechanism, the check is the feature, exactly like _missing_key_ideas and the fact memory.

MAX_LESSON_OBJECTS = 2      # Qwen-Edit affords 2 extra refs; a contrast is usually a pair
MIN_OBJECT_BEATS = 2        # named in one line only = a one-off prop, not a through-line


def _obj_slug(noun: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (noun or "").lower()).strip("-")


# The canonical LOOK for the two objects a "living vs non-living" lesson rests on, forced
# here rather than left to the model, so the PINNED reference is right: a doll is drawn as a
# faceless building block (human-shaped dolls captured the mascot's face or read like a
# horror-film doll — jeffy 2026-07-16), and the puppy is one fixed breed so it is the SAME
# puppy every shot. The prompt proposes, this is the CHECK — same doctrine as the >=2-beat
# rule below. Single source of truth: mascot.FACELESS_TOY_DESC / LABRADOR_PUPPY_DESC.
_TOY_NOUNS = re.compile(r"\b(?:doll|dolly|teddy|puppet|figurine|toy|plush|stuffed)\b", re.I)
_PUPPY_NOUNS = re.compile(r"\b(?:puppy|pup|doggy|dog)\b", re.I)


def _canonical_desc(noun: str, desc: str) -> str:
    from modules import mascot
    if _TOY_NOUNS.search(noun or ""):
        return mascot.FACELESS_TOY_DESC
    if _PUPPY_NOUNS.search(noun or ""):
        return mascot.LABRADOR_PUPPY_DESC
    return desc


def _recurring_objects(dress: dict, lines: list) -> list:
    """The objects worth pinning: proposed by the model, then CHECKED against the lesson's
    own words. Keep only a noun that really appears in >=2 lines (drops one-offs and anything
    the model invented), rank by how often it recurs, cap the list. Each becomes
    {key, noun, desc, aliases} for lesson_objects."""
    from modules.facts_memory import _singular
    line_tokens = [{_singular(w) for w in re.findall(r"[a-z]+", ln.lower())} for ln in lines]

    scored, seen = [], set()
    for it in (dress or {}).get("objects") or []:
        if not isinstance(it, dict):
            continue
        noun = (it.get("noun") or "").strip().lower()
        desc = (it.get("desc") or "").strip()
        if not noun or not desc:
            continue
        head = _singular(re.sub(r"[^a-z]", "", noun.split()[-1]))   # the matchable word
        if len(head) < 3 or head in seen:
            continue
        count = sum(1 for toks in line_tokens if head in toks)      # THE CHECK
        if count < MIN_OBJECT_BEATS:
            continue
        seen.add(head)
        aliases = [head] if head != noun else []
        scored.append((count, {"key": _obj_slug(noun), "noun": noun,
                               "desc": _canonical_desc(noun, desc), "aliases": aliases}))
    scored.sort(key=lambda x: -x[0])
    return [o for _, o in scored[:MAX_LESSON_OBJECTS]]


def write_lesson(book_id: str, topic_id: str,
                 target_seconds: float = DEFAULT_SECONDS,
                 progress_cb: Optional[Callable[[str], None]] = None) -> dict:
    """Write the lesson for one topic. Saves it and returns it.

    Raises rather than saving a stub: a placeholder script gets voiced, rendered and
    posted exactly like a real one, and nothing downstream would notice.
    """
    def _p(msg: str):
        log.info(msg)
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    t0 = _t.time()
    source, n_words, topic, book = _source_for(book_id, topic_id)
    beats_wanted, budget = plan_shape(target_seconds)
    _p(f"📖 '{topic['title']}' — {n_words} words in the book "
       f"(pages {topic['first_page']}-{topic['last_page']})")
    _p(f"🎯 aiming for {target_seconds:.0f}s: about {budget} spoken words "
       f"in ~{beats_wanted} beats")

    prose = _write_prose(topic, source, budget, _p)

    # Did the lesson keep everything the book teaches? The length loop optimises words,
    # not completeness, and drops the least vivid idea to hit a budget (breathing, which
    # has no fun example, was the one lost on the real book). Re-roll naming the gap.
    prose = _close_gaps(topic, source, prose, budget, _p)

    # Cut the lesson into lines MECHANICALLY. The model does not get to touch the
    # narration again: asked to cut without rewriting it dropped a third of the
    # teaching, then padded with repeated recaps, then compressed sentences into
    # telegraphese to hit a beat count. Sentences cannot lose a word.
    lines = split_into_lines(prose)
    if len(lines) < 4:
        raise LessonUnavailable(
            f"the lesson came out as only {len(lines)} line(s) — too little to teach. "
            f"Nothing was saved.")

    numbered = "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(lines))
    ask = (f"The lesson, line by line:\n{numbered}\n\n"
           f"Write the on-screen words and the picture for each of the {len(lines)} "
           f"lines, in order.")

    dress, last = {}, None
    for attempt in range(1, LLM_ATTEMPTS + 1):
        try:
            dress = _extract_json(_call_llm(ask, _DRESS_SYS, role="structurer")) or {}
            # Every line must get its own caption, matched by NUMBER. A missing one is
            # a silent shift waiting to happen — re-ask rather than paper over it.
            matched = _dress_by_line(lines, dress)
            if len(matched) == len(lines):
                break
            _p(f"🏷️ {len(matched)} of {len(lines)} lines were captioned "
               f"(try {attempt}/{LLM_ATTEMPTS})")
        except Exception as e:
            last = e
        _p(f"⚠️ could not caption the lines (try {attempt}/{LLM_ATTEMPTS})")

    # A missing caption is a cosmetic loss, not a broken lesson — the words are the
    # lesson, and they are already safe. Fill the gaps and carry on.
    beats = _to_beats(lines, dress)     # fills any caption the model forgot
    cut_title = (dress.get("title") or "").strip()

    # The recurring objects this lesson is built on — pinned so the doll (and the puppy) look
    # the SAME in every shot. Proposed by the model, kept only if really named in >=2 lines.
    objects = _recurring_objects(dress, lines)
    if objects:
        _p("🧸 recurring objects to keep consistent: "
           + ", ".join(o["noun"] for o in objects))

    spoken = sum(len(b["narration"].split()) for b in beats)
    est = spoken / WORDS_PER_SEC + PAD_PER_BEAT * len(beats)
    if abs(est - target_seconds) > target_seconds * 0.3:
        # Not fatal — but say it, rather than quietly handing over a 59-second lesson
        # when 90 was asked for.
        _p(f"⚠️ this lesson comes to about {est:.0f}s, not the {target_seconds:.0f}s "
           f"asked for — the book may not have enough to say for a longer one.")

    now = datetime.now()
    lesson_id = now.strftime("%Y%m%d_%H%M%S")
    lesson = {
        "lesson_id": lesson_id,
        "_id": lesson_id,
        "book_id": book_id,
        "book_title": book.get("title", ""),
        "topic_id": topic_id,
        "topic": topic["title"],
        "pages": [topic["first_page"], topic["last_page"]],
        "title": (cut_title or topic["title"]).strip()[:60],
        "target_seconds": float(target_seconds),
        "estimated_seconds": round(est, 1),
        "word_count": spoken,
        "beats": beats,
        "objects": objects,       # recurring props to keep identical (lesson_objects)
        "_prose": prose,          # keep it: a re-cut should not re-invent the lesson
        "stage": "written",
        "_generated_at": now.isoformat(timespec="seconds"),
    }
    d = LESSONS_DIR / lesson_id
    d.mkdir(parents=True, exist_ok=True)
    atomic_write_json(d / "lesson.json", lesson)

    _p(f"✅ '{lesson['title']}' — {len(beats)} beats, {spoken} words, "
       f"about {est:.0f}s (asked {target_seconds:.0f}s) in {_t.time()-t0:.0f}s")
    return lesson


# ==============================================================================
# EDITING — the script is yours before anything is rendered
# ==============================================================================

def load_lesson(lesson_id: str) -> Optional[dict]:
    p = LESSONS_DIR / lesson_id / "lesson.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _save(lesson: dict) -> None:
    atomic_write_json(LESSONS_DIR / lesson["lesson_id"] / "lesson.json", lesson)


EDITABLE_BEAT_FIELDS = ("narration", "on_screen", "image_prompt", "mascot_scene",
                        "motion_prompt", "framing")


def set_beat_field(lesson_id: str, beat_index: int, field: str, text: str) -> bool:
    """Hand-edit one beat. Read-modify-write, so an edit made in one front-end is
    seen by the other (the same rule the story pipeline follows)."""
    if field not in EDITABLE_BEAT_FIELDS:
        raise ValueError(f"field must be one of {EDITABLE_BEAT_FIELDS}")
    lesson = load_lesson(lesson_id)
    if not lesson:
        return False
    beats = lesson.get("beats", [])
    if not (0 <= beat_index < len(beats)):
        return False

    value = (text or "").strip()
    if field == "framing":
        # A framing is a fixed vocabulary, not free text — validate it, or a typo would
        # render as the old full-body default with nothing to say why (a silent downgrade).
        from modules.mascot import LESSON_FRAMINGS
        value = value.lower()
        if value not in LESSON_FRAMINGS:
            raise ValueError(f"framing must be one of {LESSON_FRAMINGS}")
        # A different framing is a different picture — un-confirm it, the way set_scene does.
        beats[beat_index]["approved"] = False

    beats[beat_index][field] = value
    if field == "narration":
        lesson["word_count"] = sum(len(b["narration"].split()) for b in beats)
        lesson["estimated_seconds"] = round(
            lesson["word_count"] / WORDS_PER_SEC + PAD_PER_BEAT * len(beats), 1)
    _save(lesson)
    return True


def set_beat_animate(lesson_id: str, beat_index: int, animate: bool) -> bool:
    """The tickbox: does this beat get a real animated shot (Wan, ~8 min of GPU) or a
    still with a slow pan (free)? Persisted, so the choice survives a page reload and
    the render reads it back rather than trusting the browser."""
    lesson = load_lesson(lesson_id)
    if not lesson:
        return False
    beats = lesson.get("beats", [])
    if not (0 <= beat_index < len(beats)):
        return False
    beats[beat_index]["animate"] = bool(animate)
    _save(lesson)
    return True


def set_beat_approved(lesson_id: str, beat_index: int, approved: bool) -> bool:
    """You have LOOKED at this picture and it is right.

    Nothing may start Wan on a picture nobody has seen. Every serious defect this pipeline
    has produced rendered cleanly and was wrong only in the file — the gate cannot be a
    thumbnail glanced at on the way to pressing Render.
    """
    lesson = load_lesson(lesson_id)
    if not lesson:
        return False
    beats = lesson.get("beats", [])
    if not (0 <= beat_index < len(beats)):
        return False
    beats[beat_index]["approved"] = bool(approved)
    _save(lesson)
    return True


def unapproved(lesson: dict) -> list:
    """The 1-based line numbers whose picture nobody has confirmed."""
    return [i + 1 for i, b in enumerate(lesson.get("beats", []))
            if not b.get("approved")]


def set_scene(lesson_id: str, beat_index: int, text: str) -> dict:
    """Hand-edit the scene a picture is drawn from — GUARDED, exactly like the bot's own.

    The guards used to run at WRITE time only (inside mascot.explainer_scene), and
    render_scene never called them. So a hand-edited scene reached Qwen byte-for-byte and
    could walk straight back into a defect we had already fixed and paid for:

        "dressed as a puppy"    -> the mascot is GONE; Qwen draws a puppy in her clothes
        an empty scene          -> idle hands go home to the reference photo, which is a
                                   T-POSE
        "a doll"                -> a headless cloth sack with buttons on its torso
        "with heart eyes"       -> her eyes deleted, two red emoji pasted over the sockets

    So an edit is cleaned by the same guards, and we hand back WHAT CHANGED so the UI can
    say so. Silently rewriting a person's words is its own kind of lie.

    The picture is also un-confirmed: the scene changed, so the image on screen is no
    longer the image this scene describes.
    """
    lesson = load_lesson(lesson_id)
    if not lesson:
        return {"saved": False}
    beats = lesson.get("beats", [])
    if not (0 <= beat_index < len(beats)):
        return {"saved": False}

    from modules import mascot
    before = (text or "").strip()
    # keep_people: a mother you named STAYS in your sentence. Whether she can be DRAWN is
    # decided at render time (lesson_pipeline._scene_and_refs), where we know which mascot
    # is rendering and whether it has a picture of her.
    #
    # It used to be decided here, destructively: saving the scene deleted her from your
    # words for good, and whether she was deleted depended on which mascot happened to be
    # active at the moment you pressed save. Add her picture afterwards and she never came
    # back — your sentence had already been rewritten on disk.
    after = mascot.clean_scene_for_the_mascot(before, teaching=True, keep_people=True)

    beats[beat_index]["mascot_scene"] = after
    beats[beat_index]["approved"] = False       # a different scene is a different picture
    _save(lesson)

    if after != before:
        log.info(f"{lesson_id} line {beat_index + 1}: the guards rewrote the scene")
    return {"saved": True, "guarded": after != before, "before": before, "after": after}


# ==============================================================================
# REORDER
# ==============================================================================
#
# EVERY ARTIFACT IS NAMED BY THE BEAT'S POSITION — still_04.png, beat_04.wav,
# clip_04.mp4 — and the narration track is rebuilt by GLOBBING those filenames, not by
# walking the beats:
#
#     wavs = sorted((d / "audio").glob("beat_*.wav"))     # lesson_pipeline
#
# So moving a beat in the list and stopping there would leave the VOICE in the old order
# while the pictures, the captions and the durations followed the new one. The video would
# render cleanly, log nothing, and be wrong in the file — which is the exact shape of every
# bug this pipeline has ever shipped. The files move WITH the beat.
#
# TWO-PHASE, not descending. shot_editor renames highest-first, which is collision-free for
# an INSERT (a monotonic +1 shift). A reorder is a PERMUTATION, and for a permutation
# "highest first" collides: moving 2->0 needs 0->1 and 1->2 at the same time. Everything
# goes to a temp name, then temp comes back into place.

_ARTIFACTS = (
    ("stills", "still_{:02d}", (".png",)),
    ("audio", "beat_{:02d}", (".wav",)),
    ("clips", "clip_{:02d}", (".mp4",)),
)


def _move_artifacts(lesson_dir: Path, order: list) -> None:
    """`order[new_index] = old_index`. Rename every family into the new positions."""
    for sub, template, exts in _ARTIFACTS:
        d = lesson_dir / sub
        if not d.is_dir():
            continue
        # phase 1 — everything out of the way, so no rename can land on a live file
        parked = {}
        for old in range(len(order)):
            for ext in exts:
                src = d / f"{template.format(old)}{ext}"
                if src.is_file():
                    tmp = d / f"_reorder_{old}{ext}"
                    src.replace(tmp)
                    parked[(old, ext)] = tmp
        # phase 2 — back into place, in the new order
        for new, old in enumerate(order):
            for ext in exts:
                tmp = parked.get((old, ext))
                if tmp and tmp.is_file():
                    tmp.replace(d / f"{template.format(new)}{ext}")


def _apply_order(lesson: dict, order: list) -> bool:
    """`order[new_index] = old_index`. Permute the beats AND everything on disk that
    belongs to them. Both move_beat and reset_order come through here — a second copy of
    this would be a second chance to desync the voice from the pictures."""
    lesson_id = lesson["lesson_id"]
    beats = lesson["beats"]
    n = len(beats)

    d = LESSONS_DIR / lesson_id
    _move_artifacts(d, order)

    lesson["beats"] = [beats[old] for old in order]

    # The still's ABSOLUTE path is stored on the beat (prepare_lesson writes it), and the
    # file it names has just been renamed out from under it.
    for i, b in enumerate(lesson["beats"]):
        if b.get("still"):
            b["still"] = str(d / "stills" / f"still_{i:02d}.png")
        b["index"] = i + 1

    # spoken.json remembers WHICH WORDS the takes on disk are of, in order. Re-order it to
    # match, or prepare_lesson would find a mismatch and re-record the whole lesson — the
    # clone is stochastic, and a re-read can collapse a line and drop the lesson to a
    # preset voice as the price of moving a shot.
    manifest = d / "audio" / "spoken.json"
    if manifest.is_file():
        try:
            said = json.loads(manifest.read_text(encoding="utf-8"))
            if isinstance(said, list) and len(said) == n:
                atomic_write_json(manifest, [said[old] for old in order])
        except (OSError, ValueError):
            pass

    _save(lesson)
    return True


def _stamp_original_order(beats: list) -> None:
    """Remember where each beat STARTED, so Reset can put it back.

    Stamped lazily, on the first move — not at write time. A lesson written before this
    existed has no record of its original order, and inventing one from the CURRENT order
    (after you have already shuffled it) would make Reset a no-op that looked like it
    worked. The first move is the last moment the order is still the original one.
    """
    if all("orig" in b for b in beats):
        return
    for i, b in enumerate(beats):
        b.setdefault("orig", i)


def move_beat(lesson_id: str, src: int, dst: int) -> bool:
    """Move one shot to another position — the beat AND everything on disk that belongs
    to it: its picture, its voice take, its animated clip."""
    lesson = load_lesson(lesson_id)
    if not lesson:
        return False
    beats = lesson.get("beats", [])
    n = len(beats)
    if not (0 <= src < n) or not (0 <= dst < n) or src == dst:
        return False

    _stamp_original_order(beats)        # BEFORE the permutation, or "original" is a lie

    order = list(range(n))              # order[new] = old
    order.insert(dst, order.pop(src))
    _apply_order(lesson, order)
    log.info(f"{lesson_id}: moved shot {src + 1} to position {dst + 1}")
    return True


def is_reordered(lesson: dict) -> bool:
    """Has the running order been changed from the one the lesson was written in?"""
    beats = lesson.get("beats", [])
    if not any("orig" in b for b in beats):
        return False                    # never moved
    return [b.get("orig", i) for i, b in enumerate(beats)] != list(range(len(beats)))


def reset_order(lesson_id: str) -> bool:
    """Put the shots back in the order the lesson was written in."""
    lesson = load_lesson(lesson_id)
    if not lesson:
        return False
    beats = lesson.get("beats", [])
    if not is_reordered(lesson):
        return False

    # order[new] = old: sort the CURRENT positions by where each beat started.
    order = sorted(range(len(beats)), key=lambda i: beats[i].get("orig", i))
    _apply_order(lesson, order)
    log.info(f"{lesson_id}: running order reset to the written one")
    return True


def lessons_for_book(book_id: str) -> list:
    """Every lesson written from this book, newest first."""
    if not LESSONS_DIR.is_dir():
        return []
    out = []
    for d in sorted(LESSONS_DIR.iterdir(), reverse=True):
        if not d.is_dir() or not lb.valid_id(d.name):
            continue
        got = load_lesson(d.name)
        if got and got.get("book_id") == book_id:
            out.append(got)
    return out
