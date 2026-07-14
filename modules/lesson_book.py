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
import time as _t
from collections import Counter
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


# ==============================================================================
# READING A SCAN — the vision model
# ==============================================================================
# A real Class-1 textbook is a photograph of every page: 64 pages, and pypdf gets
# 18 characters out of the whole book (a watermark). To use it at all, something has
# to LOOK at the pages.
#
# Two things are asked of each page, because a Class-1 page is mostly picture: the
# words, verbatim, AND one line on what the illustration shows. "A child brushing
# teeth" IS the content of that page — a text-only reader would come back with four
# words and the lesson would be built on nothing.
#
# What must never happen: the model's reading being mistaken for the book's own
# words. Every page records which it is (`source`), and a page the PDF could read is
# never sent to the model at all.

VISION_ROLE = "vision"
VISION_PULL = "ollama pull qwen2.5vl:7b"

# 150 DPI. The scans are 1363x1836 at that scale — enough for the model to read a
# Class-1 book's large print, without paying for pixels nobody needs.
RENDER_DPI = 150
JPEG_QUALITY = 85

_VISION_SYS = (
    "You are reading one page of a school textbook for young children, page by page. "
    "You do not teach, summarise or invent — you REPORT what is on the page.\n"
    'Output ONLY valid JSON: {"text": "...", "picture": "...", "blank": false}\n'
    "Rules:\n"
    "- text: every word printed on the page, verbatim, in reading order. Headings, "
    "captions, labels, questions, the lot. Keep the book's own wording exactly — "
    "never reword it, never explain it, never add to it. No words on the page: \"\".\n"
    "- picture: ONE plain sentence describing what the illustration or photograph "
    "shows, as you would say it to a child ('a girl washing her hands at a tap'). "
    "This matters: on a page for six-year-olds the picture IS most of the lesson. "
    "No picture: \"\".\n"
    "- blank: true only when the page carries nothing at all (an empty page, a plain "
    "cover, an end-paper).\n"
    "- Never guess at words you cannot read. Leave them out."
)


def vision_config() -> Optional[dict]:
    """The vision model's config, or None. NEVER falls back to a text model.

    `model_registry.get_vision()` requires an entry flagged `"vision": true`. Using
    the ordinary role lookup here would be a disaster: with no vision role
    configured it returns the TEXT model, which does not error on an image — it
    ignores the picture and writes a fluent, invented textbook page from the prompt
    alone. Sixty-four of those look exactly like a book that was read.
    """
    from modules import model_registry as mr
    return mr.get_vision()


def vision_available() -> tuple[bool, str]:
    """Is there a vision model, pulled and answering, to read a scan with?"""
    cfg = vision_config()
    if not cfg:
        return False, (f"no vision model is configured in models.json — "
                       f"run `{VISION_PULL}`, then add it under llm_backend")
    model = cfg.get("model_id") or ""
    try:
        import requests
        r = requests.get(cfg.get("server_url", "http://127.0.0.1:11434") + "/api/tags",
                         timeout=10)
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", [])]
    except Exception as e:
        return False, f"Ollama is not answering ({e})"

    if not any(n.split(":")[0] == model.split(":")[0] for n in names):
        return False, (f"the vision model '{model}' is not installed — "
                       f"run `{VISION_PULL}` and try again")
    return True, model


def _page_jpeg_b64(pdf: Path, index: int) -> str:
    """One page, rendered and base64'd for the model.

    pypdfium2 RENDERS the page. pypdf's `page.images` looked like a free shortcut —
    the scan is embedded as a JPEG already — but it returns the DOCUMENT's image list,
    not the page's: page 1 and page 9 hand back byte-identical bitmaps (measured). It
    would have fed the wrong page to the model and produced a lesson from the wrong
    chapter, with nothing on screen to say so.
    """
    import base64
    import io

    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf))
    try:
        img = doc[index].render(scale=RENDER_DPI / 72).to_pil().convert("RGB")
    finally:
        doc.close()
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def read_page_with_vision(pdf: Path, index: int, cfg: Optional[dict] = None) -> dict:
    """{text, picture, blank} for one scanned page. Raises if the model fails."""
    from modules.script_generator import _call_llm, _extract_json

    cfg = cfg or vision_config()
    if not cfg:
        raise BookUnreadable(f"no vision model configured — run `{VISION_PULL}`")

    raw = _call_llm(
        f"This is page {index + 1}. Read it.", _VISION_SYS,
        cfg=cfg,                       # explicit: never the text model
        images=[_page_jpeg_b64(pdf, index)],
        # Transcription, not creativity. The default 0.75 is a licence to guess a
        # word that is not on the page.
        options={"temperature": 0.1, "top_p": 0.9, "num_predict": 800},
    )
    got = _extract_json(raw) or {}
    text = (got.get("text") or "").strip()
    picture = (got.get("picture") or "").strip()
    return {"text": text, "picture": picture, "blank": bool(got.get("blank"))}


def _page_body(read: dict) -> str:
    """What the page contributes to a lesson: its words, and its picture.

    The picture is tagged, not blended in. A lesson written from this page must be
    able to tell what the book SAYS from what the book SHOWS.
    """
    bits = []
    if read.get("text"):
        bits.append(read["text"])
    if read.get("picture"):
        bits.append(f"[picture: {read['picture']}]")
    return "\n".join(bits).strip()


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

def add_book(pdf: Path, title: str = "", progress_cb=None) -> dict:
    """Ingest a PDF. Returns the book record.

    Pages the PDF can read are read from the PDF. Pages it cannot — a scan — are
    LOOKED AT by the vision model, one at a time. A page is never sent to the model
    when the PDF already gave us its words: the book's own text always wins.

    The file is COPIED into the book's folder: the upload it came from is a temp
    file, and a book whose source has evaporated cannot be re-read when the reader
    improves.
    """
    def _p(msg: str):
        log.info(msg)
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    pdf = Path(pdf)
    if not pdf.exists():
        raise ValueError(f"no such file: {pdf}")

    pages = extract_pages(pdf)
    if not pages:
        raise BookUnreadable("the PDF has no pages")
    for p in pages:
        p["source"] = "pdf" if p["chars"] >= MIN_PAGE_CHARS else ""

    unread = [p for p in pages if not p["source"]]
    mostly_scan = len(unread) >= max(1, int(len(pages) * SCAN_RATIO))
    ok, why = vision_available() if unread else (False, "")

    if unread and not ok:
        # A few picture pages in a text book are normal — flag them and carry on.
        # A book that is MOSTLY pictures is a scan, and without a reader there is
        # nothing to teach from: fail loud rather than invent topics from blanks.
        if mostly_scan:
            raise BookUnreadable(
                f"{len(unread)} of {len(pages)} pages have no text in them — they are "
                f"SCANS (pictures of paper). Reading them needs the vision model, and "
                f"{why}. Nothing was imported."
            )
        _p(f"⚠️ {len(unread)} page(s) have no text ({page_ranges([p['n'] for p in unread])})"
           f" and no vision model is available — they will not reach a lesson.")

    if unread and ok:
        mins = len(unread) * 20 / 60
        _p(f"👁️ {len(unread)} of {len(pages)} pages have no text — reading them with "
           f"{why} (~{mins:.0f} min)")

        # The card holds one big model at a time. Ollama's resident text model is
        # 12.6 GB and the VLM is 6 — together they do not fit in 16 GB, and Ollama
        # would quietly offload layers to the CPU and crawl. Evict first.
        cfg = vision_config()
        try:
            from modules import gpu_utils
            gpu_utils.free_ollama_vram()
        except Exception as e:
            log.warning(f"could not unload the text model before the vision read: {e}")

        t0 = _t.time()
        for i, p in enumerate(unread, start=1):
            try:
                got = read_page_with_vision(pdf, p["n"] - 1, cfg=cfg)
            except Exception as e:
                _p(f"⚠️ page {p['n']}: the vision model failed ({e})")
                continue
            body = _page_body(got)
            p["text"] = body
            p["chars"] = len(body)
            p["picture"] = got.get("picture", "")
            p["source"] = "vision" if body else ""
            if i % 5 == 0 or i == len(unread):
                per = (_t.time() - t0) / i
                left = (len(unread) - i) * per / 60
                _p(f"👁️ read {i}/{len(unread)} pages (~{left:.0f} min left)")

        # A VLM on a repetitive picture book can LOCK ON: it returns the same caption
        # for page after page. The result reads like a book that was read, and the
        # topic list looks plausible. Catch it here, not in the finished lesson.
        read_now = [p for p in unread if p["source"] == "vision" and p["text"]]
        if read_now:
            top = max(Counter(p["text"] for p in read_now).values())
            if len(read_now) >= 5 and top >= max(3, int(len(read_now) * 0.4)):
                raise BookUnreadable(
                    f"the vision model returned the SAME text for {top} of "
                    f"{len(read_now)} pages — it locked onto one page instead of "
                    f"reading them. Nothing was imported.")

    still_empty = [p["n"] for p in pages if not p["source"]]
    if len(still_empty) >= max(1, int(len(pages) * SCAN_RATIO)):
        raise BookUnreadable(
            f"even the vision model got nothing from {len(still_empty)} of "
            f"{len(pages)} pages. Nothing was imported."
        )

    book_id = _now_id()
    d = BOOKS_DIR / book_id
    while d.exists():                    # two uploads in the same second
        book_id = f"{int(book_id.split('_')[0])}_{int(book_id.split('_')[1]) + 1:06d}"
        d = BOOKS_DIR / book_id
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdf, d / "source.pdf")

    atomic_write_json(d / "pages.json", {"pages": pages})
    seen = [p["n"] for p in pages if p["source"] == "vision"]
    book = {
        "book_id": book_id,
        "_id": book_id,
        "title": (title or pdf.stem).strip(),
        "source": pdf.name,
        "n_pages": len(pages),
        "n_chars": sum(p["chars"] for p in pages),
        # Pages that gave nothing at all, even to the vision model.
        "scanned_pages": still_empty,
        # Pages whose words were READ OFF A PICTURE rather than taken from the PDF.
        # Kept separate so nothing ever passes the model's reading off as the book's
        # own text — that distinction is the difference between teaching the book and
        # teaching an impression of it.
        "vision_pages": seen,
        "topics": [],                   # filled by lesson_topics.propose()
        "_added_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_json(d / "book.json", book)

    bits = []
    if seen:
        bits.append(f"{len(seen)} page(s) read by the vision model")
    if still_empty:
        bits.append(f"{len(still_empty)} page(s) still blank: {page_ranges(still_empty)}")
    note = f" ({'; '.join(bits)})" if bits else ""
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
