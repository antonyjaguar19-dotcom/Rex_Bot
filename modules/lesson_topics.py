"""
Claw Bot — lesson mode: splitting a textbook into topics

A textbook is far too big to hand an LLM in one go. Qwen runs at `num_ctx 8192`
(~28,000 characters) and a single 30-page chapter is ~75,000. So the book is read in
WINDOWS that fit, each window is asked "what topics are taught here?", and the
answers are stitched back together.

Stitching is the part with the bug in it. A topic almost never lines up with a
window boundary — "Photosynthesis" starts on page 18 and the window ends on page 20,
so the next window reports it again, from page 21. Left alone you get two half
topics that each teach half a lesson. `_merge()` joins topics that are the same
subject in touching page ranges, and it is the reason this is two passes and not one.

Nothing here is authoritative: the topic list is a PROPOSAL. The user renames,
merges, deletes and adds in the dashboard before a single frame is rendered — the
model reads the book, the human decides what gets taught.
"""

import logging
import re
from typing import Callable, Optional

from modules import lesson_book as lb
from modules.script_generator import _call_llm, _extract_json

log = logging.getLogger("claw_bot.lesson_topics")

# One window of book text per LLM call. num_ctx is 8192 tokens (~28k chars) and the
# prompt + the answer have to live in there too, so the text gets a bit over half.
WINDOW_CHARS = 14000

# Pages either side of a window are re-read in the next one, so a topic that starts
# just before a boundary is still seen whole by somebody.
WINDOW_OVERLAP_PAGES = 1

# Two proposals are the same topic when one title CONTAINS the other's words and
# their page ranges touch.
#
# Containment, not Jaccard. The second window almost always names the topic more
# fully than the first — "Photosynthesis" then "Photosynthesis in leaves" — and
# Jaccard scores that pair 0.5, below any sane threshold, so the split topic stayed
# split. Containment scores it 1.0, which is what it is: the same subject, named
# twice. A genuinely different neighbour ("The Water Cycle") still scores 0.
TITLE_CONTAINMENT = 0.6

_STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "to", "for", "with",
         "its", "it", "is", "are", "how", "what", "why", "chapter", "unit",
         "lesson", "introduction", "intro", "topic"}

_SYS = (
    "You are a primary-school teacher reading a textbook chapter. You identify the "
    "TOPICS it teaches — the things a child would be taught as one lesson.\n"
    "Output ONLY valid JSON: {\"topics\": [{\"title\": \"...\", \"first_page\": 1, "
    "\"last_page\": 3, \"summary\": \"...\"}]}\n"
    "Rules:\n"
    "- A topic is a TEACHABLE UNIT, not a heading. 'Photosynthesis' is a topic; "
    "'Figure 2.1' and 'Exercises' are not.\n"
    "- Use the book's own words for the title where it has one.\n"
    "- first_page/last_page must be page numbers from the text you were given.\n"
    "- summary: ONE sentence on what a child learns from it.\n"
    "- Skip front matter, contents pages, exercise lists, answer keys and glossaries.\n"
    "- If the pages teach nothing (a cover, a picture spread, an index), return "
    "{\"topics\": []}. Do not invent a topic to fill the space.\n"
    "- Between 1 and 6 topics for the pages given."
)


def _tokens(title: str) -> set:
    words = re.findall(r"[a-z0-9]+", (title or "").lower())
    return {w for w in words if w not in _STOP and len(w) > 2}


def _same_topic(a: dict, b: dict) -> bool:
    """Same subject, and the page ranges touch or overlap."""
    ta, tb = _tokens(a.get("title", "")), _tokens(b.get("title", ""))
    if not ta or not tb:
        return False
    # How much of the SHORTER title is inside the longer one.
    contained = len(ta & tb) / min(len(ta), len(tb))
    if contained < TITLE_CONTAINMENT:
        return False
    # Touching = the second starts no more than a page after the first ends. A topic
    # split across a window boundary always satisfies this; two genuinely different
    # chapters with similar names (twenty pages apart) do not.
    return int(b.get("first_page", 0)) <= int(a.get("last_page", 0)) + 1


def _merge(topics: list) -> list:
    """Join the halves of a topic that a window boundary cut in two."""
    out: list = []
    for t in sorted(topics, key=lambda x: (int(x.get("first_page", 0)),
                                           int(x.get("last_page", 0)))):
        if out and _same_topic(out[-1], t):
            prev = out[-1]
            prev["last_page"] = max(int(prev["last_page"]), int(t["last_page"]))
            # Keep the longer summary — the window that saw more of the topic wrote it.
            if len(t.get("summary", "")) > len(prev.get("summary", "")):
                prev["summary"] = t["summary"]
            log.info(f"merged split topic across a window boundary: {prev['title']!r} "
                     f"(p{prev['first_page']}-{prev['last_page']})")
            continue
        out.append(dict(t))
    return out


def _absorb_nested(topics: list) -> list:
    """Drop a topic whose pages sit entirely inside another's.

    Measured on a real split: the window that saw pages 13-15 called the chapter
    "Simple Machines"; the next window, starting mid-chapter, saw only the section
    and called it "Levers" (page 15). Both are true, and page 15 was then in two
    topics — so a lesson on each would teach the same page twice. No title metric
    can connect "Levers" to "Simple Machines"; the PAGES are what give it away.
    The broader topic wins: it is the one the book itself is teaching.
    """
    out = []
    for t in sorted(topics, key=lambda x: (int(x["first_page"]),
                                           -int(x["last_page"]))):
        inside = next(
            (o for o in out
             if int(o["first_page"]) <= int(t["first_page"])
             and int(t["last_page"]) <= int(o["last_page"])),
            None)
        if inside:
            log.info(f"absorbed {t['title']!r} (p{t['first_page']}-{t['last_page']}) "
                     f"into {inside['title']!r} — its pages are already taught there")
            continue
        out.append(t)
    return out


_REDUCE_SYS = (
    "You are tidying a list of topics that were extracted from a school textbook in "
    "separate passes, so the same chapter may appear more than once, split into "
    "pieces.\n"
    'Output ONLY valid JSON: {"topics": [{"title": "...", "first_page": 1, '
    '"last_page": 9, "summary": "..."}]}\n'
    "Rules:\n"
    "- MERGE entries that are parts of the SAME chapter into ONE topic covering all "
    "of their pages. A chapter's sections ('Levers', 'Pulleys') belong to the "
    "chapter that teaches them ('Simple Machines') when their pages run together.\n"
    "- KEEP genuinely different subjects apart, even when their pages are adjacent.\n"
    "- Never invent a topic, never widen a page range beyond the pages given, and "
    "never drop a subject entirely.\n"
    "- Keep the book's own wording in titles. One sentence per summary.\n"
    "- Return the topics in page order."
)


def _consolidate(topics: list, title: str, _p) -> list:
    """One cheap LLM pass over the topic LIST (not the book) to fuse the fragments
    of a chapter that the deterministic rules cannot see.

    The list is tiny, so this always fits the context. It is advisory: if it fails,
    or returns nonsense, we keep the deterministic result rather than lose the split.
    """
    if len(topics) < 2:
        return topics
    listing = "\n".join(
        f"- {t['title']} (pages {t['first_page']}-{t['last_page']}): {t.get('summary','')}"
        for t in topics)
    try:
        raw = _call_llm(f"Textbook: {title}\n\nExtracted topics:\n{listing}\n\n"
                        f"Tidy this list.", _REDUCE_SYS, role="structurer")
        got = (_extract_json(raw) or {}).get("topics") or []
    except Exception as e:
        _p(f"⚠️ could not tidy the topic list ({e}); keeping it as found")
        return topics

    lo = min(int(t["first_page"]) for t in topics)
    hi = max(int(t["last_page"]) for t in topics)
    cleaned = _clean(got, lo, hi)
    if not cleaned:
        _p("⚠️ the tidy pass returned nothing usable; keeping the list as found")
        return topics
    # It must never lose a subject: fewer topics is the point, but a pass that
    # halves the book has misunderstood, and the raw list is the safer answer.
    if len(cleaned) < max(1, len(topics) // 3):
        _p(f"⚠️ the tidy pass collapsed {len(topics)} topics into {len(cleaned)}; "
           f"that loses material, so keeping the list as found")
        return topics
    return _absorb_nested(_merge(cleaned))


def _windows(pages: list) -> list:
    """Page groups whose text fits one LLM call, overlapping by a page."""
    readable = [p for p in pages if p["chars"] >= lb.MIN_PAGE_CHARS]
    if not readable:
        return []

    windows, cur, size = [], [], 0
    for p in readable:
        if cur and size + p["chars"] > WINDOW_CHARS:
            windows.append(cur)
            # Re-read the tail so a topic starting near the edge is seen whole.
            cur = cur[-WINDOW_OVERLAP_PAGES:] if WINDOW_OVERLAP_PAGES else []
            size = sum(x["chars"] for x in cur)
        cur.append(p)
        size += p["chars"]
    if cur:
        windows.append(cur)
    return windows


def _clean(topics: list, lo: int, hi: int) -> list:
    """Keep what the model returned only where it is usable and in range."""
    out = []
    for t in topics or []:
        title = (t.get("title") or "").strip()
        if not title:
            continue
        try:
            first = int(t.get("first_page", lo))
            last = int(t.get("last_page", first))
        except (TypeError, ValueError):
            continue
        # The model sometimes cites a page it was never shown. Clamp rather than
        # drop: the title and summary are still good, and a bad page range would
        # otherwise feed the lesson writer the wrong chapter.
        first = max(lo, min(first, hi))
        last = max(first, min(last, hi))
        out.append({"title": title[:80], "first_page": first, "last_page": last,
                    "summary": (t.get("summary") or "").strip()[:200]})
    return out


def propose(book_id: str, progress_cb: Optional[Callable[[str], None]] = None) -> list:
    """Read the whole book and propose its topics. Saves them onto the book.

    Never raises on one bad window: a chapter the model chokes on costs that
    chapter's topics, not the import. What it must never do is invent — a window
    that teaches nothing returns nothing.
    """
    def _p(msg: str):
        log.info(msg)
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    book = lb.load_book(book_id)
    if not book:
        raise ValueError(f"no book {book_id!r}")
    pages = lb.load_pages(book_id)
    windows = _windows(pages)
    if not windows:
        raise lb.BookUnreadable("no readable pages — nothing to split into topics")

    _p(f"📖 reading '{book['title']}' — {len(pages)} pages in {len(windows)} pass(es)")

    found: list = []
    for i, win in enumerate(windows, start=1):
        lo, hi = win[0]["n"], win[-1]["n"]
        body = "\n\n".join(f"[page {p['n']}]\n{p['text']}" for p in win)
        prompt = (f"Textbook: {book['title']}\n"
                  f"These are pages {lo} to {hi}.\n\n{body}\n\n"
                  f"List the topics taught in these pages.")
        try:
            raw = _call_llm(prompt, _SYS, role="structurer")
            got = _clean((_extract_json(raw) or {}).get("topics") or [], lo, hi)
        except Exception as e:
            _p(f"⚠️ pages {lo}-{hi}: could not be read ({e}); skipping")
            continue
        found.extend(got)
        _p(f"  pages {lo}-{hi}: {len(got)} topic(s)")

    # Reduce, in three steps, cheapest first:
    #   1. _merge        — the same topic reported twice across a window boundary
    #   2. _absorb_nested— a section whose pages are already inside a chapter
    #   3. _consolidate  — the LLM fuses fragments no rule can connect ("Levers"
    #                      belongs to "Simple Machines"; the words share nothing)
    topics = _consolidate(_absorb_nested(_merge(found)), book["title"], _p)
    for n, t in enumerate(topics, start=1):
        t["id"] = f"t{n:02d}"
    book["topics"] = topics
    lb.save_book(book)

    _p(f"✅ {len(topics)} topic(s) proposed — edit the list, then pick one to teach")
    return topics


# ==============================================================================
# THE USER'S EDITS — the list is a proposal, not a verdict
# ==============================================================================

def _find(book: dict, topic_id: str) -> Optional[dict]:
    return next((t for t in book.get("topics", []) if t.get("id") == topic_id), None)


def rename(book_id: str, topic_id: str, title: str) -> dict:
    book = lb.load_book(book_id)
    t = _find(book, topic_id) if book else None
    if not t:
        raise ValueError(f"no topic {topic_id!r}")
    clean = (title or "").strip()
    if not clean:
        raise ValueError("a topic needs a title")
    t["title"] = clean[:80]
    lb.save_book(book)
    return t


def remove(book_id: str, topic_id: str) -> None:
    book = lb.load_book(book_id)
    if not book:
        raise ValueError(f"no book {book_id!r}")
    before = len(book.get("topics", []))
    book["topics"] = [t for t in book["topics"] if t.get("id") != topic_id]
    if len(book["topics"]) == before:
        raise ValueError(f"no topic {topic_id!r}")
    lb.save_book(book)


def merge(book_id: str, topic_ids: list) -> dict:
    """Fuse several topics into one lesson — the book split it, you don't want it split.

    The page range becomes the span of them all, so the lesson is written from every
    page they covered.
    """
    book = lb.load_book(book_id)
    if not book:
        raise ValueError(f"no book {book_id!r}")
    picked = [t for t in book["topics"] if t.get("id") in set(topic_ids or [])]
    if len(picked) < 2:
        raise ValueError("pick at least two topics to merge")

    keep = picked[0]
    keep["first_page"] = min(int(t["first_page"]) for t in picked)
    keep["last_page"] = max(int(t["last_page"]) for t in picked)
    keep["title"] = " & ".join(t["title"] for t in picked)[:80]
    keep["summary"] = " ".join(t.get("summary", "") for t in picked).strip()[:200]

    drop = {t["id"] for t in picked[1:]}
    book["topics"] = [t for t in book["topics"] if t["id"] not in drop]
    lb.save_book(book)
    return keep


def add(book_id: str, title: str, first_page: int, last_page: int,
        summary: str = "") -> dict:
    """A topic the model missed. The pages are what the lesson gets written from, so
    they are required — a topic with no pages behind it has nothing to teach."""
    book = lb.load_book(book_id)
    if not book:
        raise ValueError(f"no book {book_id!r}")
    if not (title or "").strip():
        raise ValueError("a topic needs a title")
    first, last = int(first_page), int(last_page)
    if not (1 <= first <= last <= book["n_pages"]):
        raise ValueError(f"pages must be within 1-{book['n_pages']}")

    n = 1 + max([int(t["id"][1:]) for t in book["topics"] if t.get("id", "t").startswith("t")]
                or [0])
    t = {"id": f"t{n:02d}", "title": title.strip()[:80], "first_page": first,
         "last_page": last, "summary": (summary or "").strip()[:200]}
    book["topics"].append(t)
    book["topics"].sort(key=lambda x: (int(x["first_page"]), int(x["last_page"])))
    lb.save_book(book)
    return t


def topic_text(book_id: str, topic_id: str) -> str:
    """The book's own words for this topic — what the lesson is written FROM."""
    book = lb.load_book(book_id)
    t = _find(book, topic_id) if book else None
    if not t:
        raise ValueError(f"no topic {topic_id!r}")
    return lb.text_for_pages(book_id, int(t["first_page"]), int(t["last_page"]))
