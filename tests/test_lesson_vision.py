"""Reading a SCANNED textbook with a vision model.

The user's real book is 64 photographs of paper: pypdf extracts 18 characters from
the whole thing (a watermark). Nothing can be taught from that, so something has to
LOOK at the pages.

Every test here defends one line: the model's reading must never be mistaken for the
book's own words, and a page the PDF could read must never be shown to the model.
"""
from pathlib import Path

import pytest

from modules import lesson_book as lb
from test_lesson_book import PROSE, _pdf         # the hand-built real PDF


@pytest.fixture(autouse=True)
def books(tmp_path, monkeypatch):
    monkeypatch.setattr(lb, "BOOKS_DIR", tmp_path / "books")
    return tmp_path


@pytest.fixture
def vision(monkeypatch):
    """A stub vision model that records which pages it was shown."""
    seen = []

    def fake_read(pdf, index, cfg=None):
        seen.append(index + 1)                 # 1-based, like the book
        return {"text": f"Words printed on page {index + 1}.",
                "picture": f"a child doing something on page {index + 1}",
                "blank": False}

    monkeypatch.setattr(lb, "vision_available", lambda: (True, "qwen2.5vl:7b"))
    monkeypatch.setattr(lb, "vision_config", lambda: {"model_id": "qwen2.5vl:7b"})
    monkeypatch.setattr(lb, "read_page_with_vision", fake_read)
    fake_read.seen = seen
    return fake_read


def _write(tmp_path: Path, pages: list) -> Path:
    p = tmp_path / "book.pdf"
    p.write_bytes(_pdf(pages))
    return p


# --- the invariant ------------------------------------------------------------

def test_a_page_the_pdf_could_read_is_NEVER_shown_to_the_model(tmp_path, vision):
    """The load-bearing one. The book's own words always win: the model is only
    asked about pages the PDF could not read at all."""
    pdf = _write(tmp_path, [PROSE, "", PROSE, ""])      # pages 2 and 4 are blank
    lb.add_book(pdf)
    assert vision.seen == [2, 4], "the model was shown a page that had real text"


def test_the_source_of_every_page_is_recorded(tmp_path, vision):
    """So nothing can pass the model's reading off as the book's own text."""
    pdf = _write(tmp_path, [PROSE, "", PROSE])
    book = lb.add_book(pdf)
    pages = lb.load_pages(book["book_id"])

    assert [p["source"] for p in pages] == ["pdf", "vision", "pdf"]
    assert book["vision_pages"] == [2]


def test_the_pictures_words_are_marked_as_the_models(tmp_path, vision):
    """A lesson written from this page must be able to tell what the book SAYS from
    what a machine SAW. Strip the marker and the two become one."""
    pdf = _write(tmp_path, [""])
    book = lb.add_book(pdf)
    text = lb.load_pages(book["book_id"])[0]["text"]

    assert "Words printed on page 1." in text          # the book's words, plain
    assert "[picture: a child doing something on page 1]" in text


def test_a_scan_becomes_a_usable_book(tmp_path, vision):
    pdf = _write(tmp_path, ["", "", "", ""])
    book = lb.add_book(pdf)

    assert book["n_pages"] == 4
    assert len(book["vision_pages"]) == 4
    assert book["scanned_pages"] == []                 # nothing left unread
    assert vision.seen == [1, 2, 3, 4]


# --- the refusals -------------------------------------------------------------

def test_a_scan_with_no_vision_model_is_refused_with_the_pull_command(tmp_path,
                                                                      monkeypatch):
    monkeypatch.setattr(lb, "vision_available",
                        lambda: (False, "the vision model 'qwen2.5vl:7b' is not "
                                        "installed — run `ollama pull qwen2.5vl:7b`"))
    pdf = _write(tmp_path, ["", "", "", ""])

    with pytest.raises(lb.BookUnreadable, match="ollama pull qwen2.5vl:7b"):
        lb.add_book(pdf)
    assert lb.list_books() == [], "nothing may be left on disk after a refusal"


def test_a_few_picture_pages_do_not_need_a_vision_model(tmp_path, monkeypatch):
    """One picture page in a text book is normal — flag it and carry on. Only a book
    that is MOSTLY pictures is a scan that cannot be read without help."""
    monkeypatch.setattr(lb, "vision_available", lambda: (False, "not installed"))
    pdf = _write(tmp_path, [PROSE, "", PROSE, PROSE])

    book = lb.add_book(pdf)                            # no raise
    assert book["scanned_pages"] == [2]
    assert book["vision_pages"] == []


def test_a_model_that_locks_on_is_caught(tmp_path, monkeypatch):
    """A VLM on a repetitive picture book can return the SAME caption for page after
    page. The book then looks read, and the topic list looks plausible."""
    monkeypatch.setattr(lb, "vision_available", lambda: (True, "qwen2.5vl:7b"))
    monkeypatch.setattr(lb, "vision_config", lambda: {"model_id": "qwen2.5vl:7b"})
    monkeypatch.setattr(lb, "read_page_with_vision",
                        lambda pdf, i, cfg=None: {"text": "A child brushes their teeth.",
                                                  "picture": "a child brushing teeth",
                                                  "blank": False})
    pdf = _write(tmp_path, ["", "", "", "", "", ""])

    with pytest.raises(lb.BookUnreadable, match="locked onto one page"):
        lb.add_book(pdf)
    assert lb.list_books() == []


# --- the model routing (the worst failure available) --------------------------

def test_the_vision_role_can_never_fall_back_to_a_text_model(monkeypatch):
    """`get_for_role()` falls back role -> default -> active. That is right for text
    roles and lethal here: a text model does not ERROR on an image — it ignores the
    picture and writes a fluent, invented textbook page from the prompt alone. Sixty
    four of those look exactly like a book that was read."""
    from modules import model_registry as mr

    monkeypatch.setattr(mr, "_load", lambda: {"llm_backend": {
        "active": "text",
        "roles": {"default": "text"},                  # no vision role at all
        "available": {"text": {"model_id": "qwen2.5:14b", "type": "ollama"}},
    }})
    assert mr.get_vision() is None
    assert mr.get_for_role("vision")["model_id"] == "qwen2.5:14b"   # the trap


def test_a_model_not_flagged_vision_is_refused(monkeypatch):
    from modules import model_registry as mr

    monkeypatch.setattr(mr, "_load", lambda: {"llm_backend": {
        "active": "text",
        "roles": {"vision": "text"},                   # points at a TEXT model
        "available": {"text": {"model_id": "qwen2.5:14b", "type": "ollama"}},
    }})
    assert mr.get_vision() is None, "an entry must declare \"vision\": true"


def test_images_only_reach_the_payload_when_given(monkeypatch):
    """And a vision call must run cold: temperature 0.75 is a licence to guess a word
    that is not on the page."""
    import modules.script_generator as sg

    sent = {}

    class _R:
        def raise_for_status(self): pass
        def json(self): return {"response": "{}"}

    def fake_post(url, json=None, timeout=None):
        sent.clear()
        sent.update(json)
        return _R()
    monkeypatch.setattr(sg.requests, "post", fake_post)

    sg._call_llm("hello", "sys", role="structurer")
    assert "images" not in sent, "a text call must be byte-for-byte what it always was"
    assert sent["options"]["temperature"] == 0.75

    sg._call_llm("read it", "sys", cfg={"model_id": "qwen2.5vl:7b"},
                 images=["BASE64"], options={"temperature": 0.1})
    assert sent["images"] == ["BASE64"]
    assert sent["model"] == "qwen2.5vl:7b"
    assert sent["options"]["temperature"] == 0.1


def test_a_small_context_window_would_silently_drop_the_picture(monkeypatch):
    """Image tokens live in the prompt, and Ollama truncates an over-long prompt from
    the FRONT — so too small a window drops the image and leaves a fluent model
    answering from the words alone. No error. Refuse instead."""
    import modules.script_generator as sg

    with pytest.raises(ValueError, match="num_ctx"):
        sg._call_llm("read it", "sys", cfg={"model_id": "qwen2.5vl:7b"},
                     images=["BASE64"], options={"num_ctx": 2048})


def test_images_is_keyword_only():
    """~30 existing call sites pass (prompt, system, role) positionally, and the test
    stubs mimic that signature."""
    import inspect

    import modules.script_generator as sg
    p = inspect.signature(sg._call_llm).parameters
    assert p["images"].kind is inspect.Parameter.KEYWORD_ONLY
    assert p["cfg"].kind is inspect.Parameter.KEYWORD_ONLY


# --- the rasterizer -----------------------------------------------------------

def test_the_renderer_gives_each_page_its_OWN_picture(tmp_path):
    """pypdf's `page.images` looked like a free rasterizer — the scan is embedded as a
    JPEG already. But it returns the DOCUMENT's image list: page 1 and page 9 hand
    back byte-identical bitmaps (measured on the real book). It would have fed the
    wrong page to the model and produced a lesson from the wrong chapter."""
    pytest.importorskip("pypdfium2")
    pdf = _write(tmp_path, ["Page one says apples.", "Page two says oranges."])

    a = lb._page_jpeg_b64(pdf, 0)
    b = lb._page_jpeg_b64(pdf, 1)
    assert a and b and a != b, "every page must render to its own image"
