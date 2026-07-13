"""
Claw Bot — lesson mode: the book

A school textbook goes in as a PDF; what comes out is the text of every page and an
honest report of the pages we could not read.

That second half is the point. A textbook PDF is either born-digital (the text is
really in there) or a SCAN (every page is a photograph of paper, and there is no
text to extract at all — `extract_text()` returns ""). Both look identical in a
viewer. If a scan were ingested silently, the topic splitter would be handed empty
strings, invent plausible-sounding topics out of nothing, and the first anyone
would know is a rendered lesson that has no relationship to the book.

So: pages that yield no text are counted, named, and reported. `readable()` says
whether there is enough to work with, and refuses rather than guesses.

    04_Outputs/lessons/books/<book_id>/
        source.pdf      the file as uploaded
        pages.json      {"pages": [{"n": 1, "text": "...", "chars": 812}, ...]}
        book.json       title, page count, scanned-page list, topics[]

OCR is deliberately NOT here. The containment rule keeps everything inside
E:\\Rexjaw_VFX, and the usual OCR routes (pytesseract, pdf2image) need a system
binary installed elsewhere. When the scans need reading, the clean route is to
rasterize a page and send it to a vision model through Ollama, which is already
running — that is a later phase, and this module's job is to make the need for it
LOUD instead of invisible.
"""

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from modules.file_utils import atomic_write_json

log = logging.getLogger("claw_bot.lesson_book")

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
BOOKS_DIR = PROJECT_ROOT / "04_Outputs" / "lessons" / "books"

# A page with fewer than this many characters told us nothing. Real textbook pages
# run 800-3000 characters; a chapter-opener with one heading and a full-page picture
# lands around 40-80, and a scan lands at exactly 0. 25 keeps the picture pages out
# of the "broken" bucket while still catching a scan.
MIN_PAGE_CHARS = 25

# If this share of pages is unreadable, the PDF is a scan and there is nothing to
# work with — say so instead of proposing topics from a handful of captions.
SCAN_RATIO = 0.6

ID_RE = re.compile(r"^\d{8}_\d{6}$")


class BookUnreadable(RuntimeError):
    """The PDF yielded (almost) no text — it is a scan, or it is encrypted.

    Raised instead of returning an empty book: a book with no text produces topics
    invented out of nothing, and a lesson that has no relationship to the textbook.
    """


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def valid_id(book_id: str) -> bool:
    return bool(ID_RE.match((book_id or "").strip()))


def book_dir(book_id: str) -> Path:
    if not valid_id(book_id):
        raise ValueError(f"not a book id: {book_id!r}")
    return BOOKS_DIR / book_id


# ==============================================================================
# EXTRACT
# ==============================================================================

def _clean_page(text: str) -> str:
    """Textbook PDFs come out with hard line-wraps mid-sentence and running
    headers. Join the wraps so the LLM reads sentences, not ragged columns."""
    text = (text or "").replace("\r", "\n")
    text = re.sub(r"-\n(\w)", r"\1", text)          # de-hyphenate across a line break
    text = re.sub(r"[ \t]+", " ", text)
    # A newline that is not the end of a sentence is a wrap, not a break.
    text = re.sub(r"(?<![.!?:;])\n(?=[a-z0-9(])", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_path: Path) -> list:
    """[{n, text, chars}] for every page. Never raises on a single bad page."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")          # many "encrypted" textbooks have an empty owner pw
        except Exception as e:
            raise BookUnreadable(f"the PDF is password-protected ({e})") from e

    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = _clean_page(page.extract_text() or "")
        except Exception as e:
            log.warning(f"page {i}: extract failed ({e}); treating as unreadable")
            text = ""
        pages.append({"n": i, "text": text, "chars": len(text)})
    return pages


def scanned_pages(pages: list) -> list:
    """Page numbers that gave us no usable text."""
    return [p["n"] for p in pages if p["chars"] < MIN_PAGE_CHARS]


def page_ranges(numbers: list) -> str:
    """[12,13,14,31] -> '12-14, 31'. For telling the user which pages are scans."""
    if not numbers:
        return ""
    out, start, prev = [], numbers[0], numbers[0]
    for n in numbers[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append(f"{start}-{prev}" if prev > start else f"{start}")
        start = prev = n
    out.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ", ".join(out)


# ==============================================================================
# THE BOOK STORE
# ==============================================================================

def add_book(pdf: Path, title: str = "") -> dict:
    """Ingest a PDF. Returns the book record. Raises BookUnreadable on a scan.

    The file is COPIED into the book's folder: the upload it came from is a temp
    file, and a book whose source has evaporated cannot be re-extracted when the
    splitter improves.
    """
    pdf = Path(pdf)
    if not pdf.exists():
        raise ValueError(f"no such file: {pdf}")

    pages = extract_pages(pdf)
    if not pages:
        raise BookUnreadable("the PDF has no pages")

    bad = scanned_pages(pages)
    if len(bad) >= max(1, int(len(pages) * SCAN_RATIO)):
        raise BookUnreadable(
            f"{len(bad)} of {len(pages)} pages have no extractable text — this PDF "
            f"is a SCAN (pictures of paper), not a text PDF. Reading it needs OCR, "
            f"which is not built yet. Nothing was imported."
        )

    book_id = _now_id()
    d = BOOKS_DIR / book_id
    while d.exists():                    # two uploads in the same second
        book_id = f"{int(book_id.split('_')[0])}_{int(book_id.split('_')[1]) + 1:06d}"
        d = BOOKS_DIR / book_id
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdf, d / "source.pdf")

    atomic_write_json(d / "pages.json", {"pages": pages})
    book = {
        "book_id": book_id,
        "_id": book_id,
        "title": (title or pdf.stem).strip(),
        "source": pdf.name,
        "n_pages": len(pages),
        "n_chars": sum(p["chars"] for p in pages),
        "scanned_pages": bad,           # readable book, but these pages gave nothing
        "topics": [],                   # filled by lesson_topics.propose()
        "_added_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_json(d / "book.json", book)

    note = (f" ({len(bad)} page(s) unreadable: {page_ranges(bad)})" if bad else "")
    log.info(f"book {book_id}: '{book['title']}' — {len(pages)} pages, "
             f"{book['n_chars']} chars{note}")
    return book


def load_book(book_id: str) -> Optional[dict]:
    p = book_dir(book_id) / "book.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"book.json unreadable for {book_id}: {e}")
        return None


def save_book(book: dict) -> None:
    atomic_write_json(book_dir(book["book_id"]) / "book.json", book)


def load_pages(book_id: str) -> list:
    p = book_dir(book_id) / "pages.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("pages", [])


def list_books() -> list:
    if not BOOKS_DIR.is_dir():
        return []
    out = []
    for d in sorted(BOOKS_DIR.iterdir(), reverse=True):
        if d.is_dir() and valid_id(d.name):
            got = load_book(d.name)
            if got:
                out.append(got)
    return out


def text_for_pages(book_id: str, first: int, last: int) -> str:
    """The book's own words for a page range — this is what a lesson is written
    FROM, so it is never paraphrased on the way out."""
    pages = load_pages(book_id)
    picked = [p["text"] for p in pages if first <= p["n"] <= last and p["text"]]
    return "\n\n".join(picked).strip()


def remove_book(book_id: str) -> None:
    d = book_dir(book_id)
    if not d.is_dir():
        raise ValueError(f"no book {book_id!r}")
    shutil.rmtree(d)
    log.info(f"book {book_id} removed")
