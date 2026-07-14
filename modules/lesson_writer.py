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
    "For each spoken line of a children's lesson you write two things.\n"
    'Output ONLY valid JSON: {"title": "...", "beats": [{"on_screen": "...", '
    '"image_prompt": "..."}]}\n'
    "Rules:\n"
    "- ONE entry per line, in the same order. Never change, quote or repeat the lines.\n"
    "- on_screen: 2-4 words, the idea of that line, for big text on screen. Not a "
    "sentence.\n"
    "- image_prompt: a friendly cartoon scene a small child would like, showing ONE "
    "action with ONE object, matching that line. No text, letters or numbers in the "
    "image.\n"
    "- title: 3-6 words, what the whole lesson is called."
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


def _write_prose(topic: dict, source: str, budget: int, _p) -> str:
    """Stage 1: the lesson, as a teacher would say it. Prose, no structure."""
    ask = (f"Topic: {topic['title']}\n"
           f"What the child's textbook says about it:\n---\n{source}\n---"
           f"{_PICTURE_NOTE}\n\n"
           f"Teach this topic to a six-year-old, out loud, in about {budget} words.")

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
    return lines


def _kind_for(i: int, line: str, total: int) -> str:
    if i == 0:
        return "intro"
    if i == total - 1:
        return "outro"
    if "?" in line:
        return "check"
    return "teach"


def _to_beats(lines: list, dress: dict) -> list:
    """The lesson's own lines + the model's caption and picture for each."""
    got = (dress or {}).get("beats") or []
    beats = []
    for i, line in enumerate(lines):
        extra = got[i] if i < len(got) and isinstance(got[i], dict) else {}
        beats.append({
            "kind": _kind_for(i, line, len(lines)),
            "narration": line,                 # the lesson's words, untouched
            "on_screen": (extra.get("on_screen") or "").strip()[:40],
            "image_prompt": (extra.get("image_prompt") or "").strip(),
            # The tickbox. OFF by default and that is deliberate: ON would mean an
            # ~8-minute Wan render for every beat the moment you pressed Render.
            "animate": False,
            "index": i + 1,
        })

    return beats


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
            if len(dress.get("beats") or []) >= len(lines) // 2:
                break
        except Exception as e:
            last = e
        _p(f"⚠️ could not caption the lines (try {attempt}/{LLM_ATTEMPTS})")

    # A missing caption is a cosmetic loss, not a broken lesson — the words are the
    # lesson, and they are already safe. Fill the gaps and carry on.
    beats = _to_beats(lines, dress)
    for b in beats:
        if not b["on_screen"]:
            b["on_screen"] = " ".join(b["narration"].split()[:3]).strip(" ,.!?").title()
        if not b["image_prompt"]:
            b["image_prompt"] = (f"a friendly cartoon child, {b['narration'].rstrip('.?!')}, "
                                 f"bright and simple, no text")
    cut_title = (dress.get("title") or "").strip()

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
                        "motion_prompt")


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
    beats[beat_index][field] = (text or "").strip()
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
