"""Splitting a textbook into topics.

A 30-page chapter is ~75,000 characters against a ~28,000-character context window,
so the book is read in WINDOWS and the answers are stitched back together. The
stitching is where the bug lives: a topic almost never lines up with a window
boundary, so "Photosynthesis" gets reported twice — once ending at page 20, once
starting at page 21 — and left alone you get two half topics that each teach half a
lesson.

The list is a PROPOSAL. The user renames, merges, deletes and adds before a frame is
rendered.
"""
import json

import pytest

from modules import lesson_book as lb
from modules import lesson_topics as lt


@pytest.fixture(autouse=True)
def books(tmp_path, monkeypatch):
    monkeypatch.setattr(lb, "BOOKS_DIR", tmp_path / "books")
    return tmp_path


@pytest.fixture
def book(monkeypatch):
    """A book on disk, without going through a PDF."""
    bid = "20260713_120000"
    d = lb.BOOKS_DIR / bid
    d.mkdir(parents=True)
    pages = [{"n": i, "text": f"page {i} " + "word " * 200, "chars": 1000}
             for i in range(1, 9)]
    (d / "pages.json").write_text(json.dumps({"pages": pages}), encoding="utf-8")
    (d / "book.json").write_text(json.dumps({
        "book_id": bid, "_id": bid, "title": "Class 5 Science", "n_pages": 8,
        "scanned_pages": [], "topics": [],
    }), encoding="utf-8")
    return bid


def _llm(*answers):
    """Stub the LLM: one canned answer per window, then repeat the last."""
    calls = []

    def fake(prompt, system, role="structurer"):
        calls.append(prompt)
        i = min(len(calls) - 1, len(answers) - 1)
        return json.dumps(answers[i])
    fake.calls = calls
    return fake


# --- the merge (the reason this is two passes) --------------------------------

def test_a_topic_cut_in_half_by_a_window_boundary_is_rejoined():
    """The whole point of the reduce pass. Two halves, each teaching half a lesson,
    become one topic spanning both."""
    got = lt._merge([
        {"title": "Photosynthesis", "first_page": 18, "last_page": 20, "summary": "a"},
        {"title": "Photosynthesis in leaves", "first_page": 21, "last_page": 23,
         "summary": "a longer summary"},
    ])
    assert len(got) == 1
    assert got[0]["first_page"] == 18 and got[0]["last_page"] == 23
    assert got[0]["summary"] == "a longer summary"   # the window that saw more wrote it


def test_the_same_title_far_apart_is_NOT_merged():
    """Two chapters can both be called "Revision". Twenty pages apart they are two
    topics, and fusing them would write a lesson from the wrong pages."""
    got = lt._merge([
        {"title": "Revision", "first_page": 5, "last_page": 6, "summary": ""},
        {"title": "Revision", "first_page": 40, "last_page": 41, "summary": ""},
    ])
    assert len(got) == 2


def test_different_topics_on_touching_pages_are_NOT_merged():
    got = lt._merge([
        {"title": "Photosynthesis", "first_page": 18, "last_page": 20, "summary": ""},
        {"title": "The Water Cycle", "first_page": 21, "last_page": 24, "summary": ""},
    ])
    assert len(got) == 2


def test_a_section_inside_a_chapter_is_absorbed():
    """Measured on a real split of a real book. The window that saw pages 13-15
    called the chapter "Simple Machines"; the next window, starting mid-chapter, saw
    only the section and called it "Levers" (page 15). Page 15 was then in TWO
    topics, so a lesson on each would teach it twice. No title metric can connect
    "Levers" to "Simple Machines" — the PAGES give it away."""
    got = lt._absorb_nested([
        {"title": "Simple Machines", "first_page": 13, "last_page": 15, "summary": ""},
        {"title": "Levers", "first_page": 15, "last_page": 15, "summary": ""},
    ])
    assert [t["title"] for t in got] == ["Simple Machines"]


def test_absorb_keeps_topics_that_merely_touch():
    """Adjacent is not nested. Two chapters back to back stay two chapters."""
    got = lt._absorb_nested([
        {"title": "Photosynthesis", "first_page": 1, "last_page": 6, "summary": ""},
        {"title": "The Water Cycle", "first_page": 7, "last_page": 12, "summary": ""},
    ])
    assert len(got) == 2


def test_the_tidy_pass_refuses_to_lose_the_book(monkeypatch):
    """The LLM reduce pass is advisory. A pass that collapses nine topics into one
    has misunderstood the book, and the raw list is the safer answer."""
    topics = [{"title": f"Topic {i}", "first_page": i, "last_page": i, "summary": ""}
              for i in range(1, 10)]

    monkeypatch.setattr(lt, "_call_llm", _llm({"topics": [
        {"title": "Science", "first_page": 1, "last_page": 9, "summary": "everything"}]}))
    assert lt._consolidate(topics, "Science", lambda m: None) == topics

    # And a dead LLM costs the tidy, not the split.
    def boom(*a, **k):
        raise RuntimeError("ollama is asleep")
    monkeypatch.setattr(lt, "_call_llm", boom)
    assert lt._consolidate(topics, "Science", lambda m: None) == topics


# --- windows ------------------------------------------------------------------

def test_windows_fit_the_context_and_never_split_a_page(book):
    pages = lb.load_pages(book)
    wins = lt._windows(pages)
    assert wins, "a readable book must produce at least one window"
    for w in wins:
        body = sum(p["chars"] for p in w)
        assert body <= lt.WINDOW_CHARS + 1000        # one page of slack
        assert all(isinstance(p["n"], int) for p in w)


def test_unreadable_pages_are_not_fed_to_the_llm(book):
    """A scanned page contributes nothing; sending it just wastes context."""
    pages = lb.load_pages(book) + [{"n": 9, "text": "", "chars": 0}]
    flat = [p["n"] for w in lt._windows(pages) for p in w]
    assert 9 not in flat


# --- proposing ----------------------------------------------------------------

def test_propose_saves_an_editable_topic_list(book, monkeypatch):
    monkeypatch.setattr(lt, "_call_llm", _llm({"topics": [
        {"title": "Photosynthesis", "first_page": 1, "last_page": 4,
         "summary": "How plants make food."},
        {"title": "The Water Cycle", "first_page": 5, "last_page": 8, "summary": "Rain."},
    ]}))

    topics = lt.propose(book)
    assert [t["title"] for t in topics] == ["Photosynthesis", "The Water Cycle"]
    assert [t["id"] for t in topics] == ["t01", "t02"]
    assert lb.load_book(book)["topics"] == topics       # persisted


def test_a_window_that_teaches_nothing_returns_nothing(book, monkeypatch):
    """A contents page or an index must not have a topic invented for it."""
    monkeypatch.setattr(lt, "_call_llm", _llm({"topics": []}))
    assert lt.propose(book) == []


def test_a_page_the_model_never_saw_is_clamped_not_trusted(book, monkeypatch):
    """Qwen cites page 400 of an 8-page book. Clamp: the title is still good, but a
    bad page range would write the lesson from the wrong chapter."""
    monkeypatch.setattr(lt, "_call_llm", _llm({"topics": [
        {"title": "Photosynthesis", "first_page": 400, "last_page": 900, "summary": ""}]}))
    t = lt.propose(book)[0]
    assert 1 <= t["first_page"] <= t["last_page"] <= 8


def test_one_bad_window_costs_that_window_not_the_import(book, monkeypatch):
    def boom(prompt, system, role="structurer"):
        raise RuntimeError("ollama is asleep")
    monkeypatch.setattr(lt, "_call_llm", boom)

    assert lt.propose(book) == []          # logged and skipped, no crash
    assert lb.load_book(book)["topics"] == []


# --- the user's edits ---------------------------------------------------------

def test_rename_merge_delete_add_round_trip(book, monkeypatch):
    monkeypatch.setattr(lt, "_call_llm", _llm({"topics": [
        {"title": "Photosynthesis", "first_page": 1, "last_page": 3, "summary": "x"},
        {"title": "Leaves", "first_page": 4, "last_page": 5, "summary": "y"},
        {"title": "The Water Cycle", "first_page": 6, "last_page": 8, "summary": "z"},
    ]}))
    lt.propose(book)

    lt.rename(book, "t01", "How Plants Eat")
    assert lb.load_book(book)["topics"][0]["title"] == "How Plants Eat"

    # Merging fuses the page ranges — the lesson is then written from all of them.
    merged = lt.merge(book, ["t01", "t02"])
    assert merged["first_page"] == 1 and merged["last_page"] == 5
    assert len(lb.load_book(book)["topics"]) == 2

    lt.remove(book, "t03")
    assert [t["id"] for t in lb.load_book(book)["topics"]] == ["t01"]

    added = lt.add(book, "Roots", 6, 7, "What roots do.")
    assert added["id"] == "t02"
    assert lt.topic_text(book, "t02").startswith("page 6")


def test_edits_refuse_what_they_cannot_do(book, monkeypatch):
    monkeypatch.setattr(lt, "_call_llm", _llm({"topics": [
        {"title": "Photosynthesis", "first_page": 1, "last_page": 3, "summary": ""}]}))
    lt.propose(book)

    with pytest.raises(ValueError):
        lt.rename(book, "t01", "   ")                  # a topic needs a title
    with pytest.raises(ValueError):
        lt.remove(book, "t99")                         # no such topic
    with pytest.raises(ValueError):
        lt.merge(book, ["t01"])                        # merge needs two
    with pytest.raises(ValueError, match="pages"):
        lt.add(book, "Ghost Chapter", 40, 50)          # pages outside the book
