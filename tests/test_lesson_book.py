"""Reading a school textbook PDF — and refusing to pretend when it is a scan.

A textbook PDF is either born-digital (the text is really in there) or a photograph
of paper. In a viewer they look identical; to `extract_text()` one returns prose and
the other returns "". If a scan were ingested silently, the topic splitter would be
handed empty strings, would invent plausible topics out of nothing, and the first
anyone would know is a rendered lesson with no relationship to the book.

So the load-bearing behaviour here is the REFUSAL.
"""
import zlib
from pathlib import Path

import pytest

from modules import lesson_book as lb


# --- a real PDF, built by hand (no generator library in the venv) -------------

def _pdf(pages: list) -> bytes:
    """A minimal, valid PDF. `pages` is a list of page texts; "" = a page with no
    text at all, which is what a scanned page looks like to an extractor."""
    objs, page_ids = [], []
    n = 3 + len(pages) * 2                      # 1=catalog 2=pages 3=font

    for i, text in enumerate(pages):
        pid, cid = 4 + i * 2, 5 + i * 2
        page_ids.append(pid)
        lines = "".join(
            f"BT /F1 12 Tf 40 {720 - 16 * j} Td ({ln}) Tj ET\n"
            for j, ln in enumerate(text.split("\n")) if ln
        )
        stream = zlib.compress(lines.encode("latin-1"))
        objs.append((pid, f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                          f"/Resources << /Font << /F1 3 0 R >> >> "
                          f"/Contents {cid} 0 R >>".encode("latin-1")))
        objs.append((cid, b"<< /Length %d /Filter /FlateDecode >>\nstream\n%s\nendstream"
                     % (len(stream), stream)))

    kids = " ".join(f"{p} 0 R" for p in page_ids)
    head = [
        (1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        (2, f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("latin-1")),
        (3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]
    body, offsets = b"%PDF-1.4\n", {}
    for num, payload in head + objs:
        offsets[num] = len(body)
        body += b"%d 0 obj\n" % num + payload + b"\nendobj\n"

    start = len(body)
    body += b"xref\n0 %d\n0000000000 65535 f \n" % (n + 1)
    for num in range(1, n + 1):
        body += b"%010d 00000 n \n" % offsets.get(num, 0)
    body += (b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
             % (n + 1, start))
    return body


PROSE = ("Photosynthesis is how a green plant makes its own food.\n"
         "The leaf takes in sunlight and carbon dioxide from the air.\n"
         "Water travels up from the roots through the stem to the leaf.\n"
         "Inside the leaf a green colour called chlorophyll traps the sunlight.\n"
         "The plant turns these into sugar and gives out oxygen for us to breathe.\n")


@pytest.fixture(autouse=True)
def books(tmp_path, monkeypatch):
    d = tmp_path / "books"
    monkeypatch.setattr(lb, "BOOKS_DIR", d)
    # No vision model in these tests — they are about what the PDF itself yields.
    # Left live, the result would depend on whether Ollama happened to be running.
    monkeypatch.setattr(lb, "vision_available", lambda: (False, "not installed (test)"))
    return d


def _write(tmp_path: Path, name: str, pages: list) -> Path:
    p = tmp_path / name
    p.write_bytes(_pdf(pages))
    return p


# --- extraction ---------------------------------------------------------------

def test_a_digital_textbook_is_read(tmp_path):
    pdf = _write(tmp_path, "science.pdf", [PROSE, PROSE, PROSE])
    book = lb.add_book(pdf, title="Class 5 Science")

    assert book["n_pages"] == 3
    assert book["scanned_pages"] == []
    assert lb.valid_id(book["book_id"])
    assert "Photosynthesis" in lb.load_pages(book["book_id"])[0]["text"]
    # The source is kept: a book whose PDF has evaporated cannot be re-read when
    # the splitter improves.
    assert (lb.book_dir(book["book_id"]) / "source.pdf").exists()


def test_line_wraps_are_joined_so_the_llm_reads_sentences():
    """PDF text arrives hard-wrapped mid-sentence. Left ragged, the model reads
    columns instead of prose."""
    got = lb._clean_page("The leaf takes in sun-\nlight and carbon dioxide\nfrom the air.")
    assert got == "The leaf takes in sunlight and carbon dioxide from the air."


def test_a_page_with_no_text_is_flagged_not_dropped(tmp_path):
    """One picture page in a text book is normal. It must be REPORTED, so nobody
    wonders later why the lesson skipped a page."""
    pdf = _write(tmp_path, "mixed.pdf", [PROSE, "", PROSE, PROSE])
    book = lb.add_book(pdf)

    assert book["scanned_pages"] == [2]
    assert book["n_pages"] == 4                  # the page is still there


def test_a_scanned_book_is_REFUSED(tmp_path):
    """The whole point. A scan yields no text; ingesting it silently would let the
    splitter invent topics out of empty strings."""
    pdf = _write(tmp_path, "scan.pdf", ["", "", "", PROSE])
    with pytest.raises(lb.BookUnreadable, match="SCAN"):
        lb.add_book(pdf)

    assert lb.list_books() == [], "nothing may be left on disk after a refusal"


def test_page_ranges_are_readable():
    """"pages 12-14, 31" — not a wall of twenty numbers."""
    assert lb.page_ranges([12, 13, 14, 31]) == "12-14, 31"
    assert lb.page_ranges([7]) == "7"
    assert lb.page_ranges([]) == ""


# --- the store ----------------------------------------------------------------

def test_text_for_pages_returns_the_books_own_words(tmp_path):
    """A lesson is written FROM this text, so it is never paraphrased on the way out."""
    pdf = _write(tmp_path, "b.pdf", [PROSE, "Chapter two is about water.", PROSE])
    book = lb.add_book(pdf)

    got = lb.text_for_pages(book["book_id"], 2, 2)
    assert "Chapter two is about water." in got
    assert "Photosynthesis" not in got


def test_books_list_and_remove(tmp_path):
    pdf = _write(tmp_path, "b.pdf", [PROSE])
    book = lb.add_book(pdf)
    assert [b["book_id"] for b in lb.list_books()] == [book["book_id"]]

    lb.remove_book(book["book_id"])
    assert lb.list_books() == []
    with pytest.raises(ValueError):
        lb.remove_book(book["book_id"])


def test_a_bad_id_cannot_escape_the_books_folder():
    """book_dir() feeds rmtree. "../.." must never resolve."""
    for bad in ("..", "*", "", "../../04_Outputs"):
        with pytest.raises(ValueError):
            lb.book_dir(bad)
