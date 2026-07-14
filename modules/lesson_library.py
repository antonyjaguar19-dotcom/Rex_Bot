"""
Claw Bot — lesson mode: the lessons you have made

Every lesson, newest first, and a way to throw one away.

A lesson lives in exactly ONE tree — `04_Outputs/lessons/<id>/` holds its script, its
voice, its stills, its clips and its finished video. That was a deliberate choice when
the mode was built, and this module is why: deleting a lesson is one `rmtree`, not a
scavenger hunt across five directories (which is what facts reels needed, and why
`04_Outputs/final` once held 212 orphaned files).

The id is validated against `^\\d{8}_\\d{6}$` BEFORE it is joined to a path. It is going
into a delete — `..` or `*` must never get that far.
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from modules import lesson_book as lb
from modules import lesson_writer as lw

log = logging.getLogger("claw_bot.lesson_library")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
LESSONS_DIR = PROJECT_ROOT / "04_Outputs" / "lessons"


def _dir(lesson_id: str) -> Path:
    if not lb.valid_id(lesson_id):
        raise ValueError(f"not a lesson id: {lesson_id!r}")
    return LESSONS_DIR / lesson_id


def describe(lesson_id: str) -> Optional[dict]:
    """Everything the list shows for one lesson, or None."""
    if not lb.valid_id(lesson_id):
        return None
    lesson = lw.load_lesson(lesson_id)
    if not lesson:
        return None

    d = LESSONS_DIR / lesson_id
    video = Path(lesson["video"]) if lesson.get("video") else None
    if video and not video.exists():
        video = None
    stills = sorted((d / "stills").glob("still_*.png")) if (d / "stills").is_dir() else []

    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    when = None
    try:
        when = datetime.strptime(lesson_id, "%Y%m%d_%H%M%S")
    except ValueError:
        pass

    return {
        "id": lesson_id,
        "title": lesson.get("title", ""),
        "topic": lesson.get("topic", ""),
        "book": lesson.get("book_title", ""),
        "stage": lesson.get("stage", "written"),
        "beats": len(lesson.get("beats", [])),
        "seconds": lesson.get("estimated_seconds", 0),
        "animated": sum(1 for b in lesson.get("beats", []) if b.get("animate")),
        "when": when,
        "video": video,
        "stills": stills,
        "size_mb": round(size / 1e6, 1),
    }


def list_lessons(book_id: str = "") -> list:
    """Every lesson, newest first. `book_id` filters to one textbook."""
    if not LESSONS_DIR.is_dir():
        return []
    out = []
    for d in sorted(LESSONS_DIR.iterdir(), reverse=True):
        if not d.is_dir() or not lb.valid_id(d.name):
            continue
        got = describe(d.name)
        if not got:
            continue
        if book_id:
            lesson = lw.load_lesson(d.name) or {}
            if lesson.get("book_id") != book_id:
                continue
        out.append(got)
    return out


def footprint(lesson_id: str) -> tuple:
    """(files, MB) a delete would remove."""
    d = _dir(lesson_id)
    files = [f for f in d.rglob("*") if f.is_file()] if d.is_dir() else []
    return len(files), round(sum(f.stat().st_size for f in files) / 1e6, 1)


def delete_lesson(lesson_id: str) -> dict:
    """Delete a lesson and everything it owns. Irreversible.

    The book it was written from is untouched — you will want to teach that topic
    again, and re-reading a 64-page scan costs ten minutes.
    """
    d = _dir(lesson_id)
    if not d.is_dir():
        raise ValueError(f"no lesson {lesson_id!r}")
    n, mb = footprint(lesson_id)
    shutil.rmtree(d)
    log.info(f"lesson {lesson_id} deleted ({n} files, {mb} MB)")
    return {"id": lesson_id, "files": n, "size_mb": mb}
