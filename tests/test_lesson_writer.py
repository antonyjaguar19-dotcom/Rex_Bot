"""Turning a textbook topic into a lesson a six-year-old can follow.

The failures this guards against are all the same shape: a script that LOOKS like a
lesson but is not the book's lesson. A topic the book barely mentions, padded out to
ninety seconds. A machine's description of a picture, quoted as if the textbook had
said it. A placeholder saved to disk when the model was down, then voiced and
rendered like the real thing.
"""
import json

import pytest

from modules import lesson_book as lb
from modules import lesson_writer as lw


@pytest.fixture(autouse=True)
def outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(lb, "BOOKS_DIR", tmp_path / "books")
    monkeypatch.setattr(lw, "LESSONS_DIR", tmp_path / "lessons")
    return tmp_path


@pytest.fixture
def book():
    """A book on disk with two topics: one teachable, one far too thin."""
    bid = "20260714_090000"
    d = lb.BOOKS_DIR / bid
    d.mkdir(parents=True)
    body = ("Your body has many parts. You have two eyes to see with. "
            "You have two ears to hear with. Wash your hands before you eat. "
            "Brush your teeth twice every day to keep them strong and clean. ") * 6
    pages = [
        {"n": 1, "text": body, "chars": len(body), "source": "vision"},
        {"n": 2, "text": "Flowers.\n[picture: a red flower with five petals]",
         "chars": 46, "source": "vision"},
    ]
    (d / "pages.json").write_text(json.dumps({"pages": pages}), encoding="utf-8")
    (d / "book.json").write_text(json.dumps({
        "book_id": bid, "_id": bid, "title": "Class 1 EVS", "n_pages": 2,
        "topics": [
            {"id": "t01", "title": "My Wonderful Body", "first_page": 1,
             "last_page": 1, "summary": ""},
            {"id": "t02", "title": "Flowers", "first_page": 2, "last_page": 2,
             "summary": ""},
        ],
    }), encoding="utf-8")
    return bid


def _llm(prose: str, beats: list = None):
    """Stub both stages: the prose call, then the captioning call.

    The model no longer writes the narration — the lesson's own sentences are the
    beats. All it is asked for now is the on-screen words and the picture.
    """
    calls = []

    def fake(prompt, system, role="structurer", **kw):
        calls.append({"prompt": prompt, "system": system, "role": role})
        if "on-screen words" in prompt:
            return json.dumps({"title": "Your Wonderful Body",
                               "beats": [{"on_screen": "Body Part",
                                          "image_prompt": "a happy child"}
                                         for _ in range(30)]})
        return prose
    fake.calls = calls
    return fake


BEATS = [
    {"kind": "intro", "narration": "Your body can do amazing things every single day.",
     "on_screen": "Your body", "image_prompt": "a happy child waving"},
    {"kind": "teach", "narration": "You have two eyes, and they let you see the world.",
     "on_screen": "Two eyes", "image_prompt": "a child looking at a butterfly"},
    {"kind": "example", "narration": "You use your ears to hear your friend calling you.",
     "on_screen": "Two ears", "image_prompt": "a child listening"},
    {"kind": "check", "narration": "Which part of you helps you hear? Your ears do.",
     "on_screen": "Check", "image_prompt": "a child pointing at an ear"},
    {"kind": "recap", "narration": "Wash your hands and brush your teeth every day.",
     "on_screen": "Keep clean", "image_prompt": "a child brushing teeth"},
    {"kind": "outro", "narration": "Your body is wonderful. Look after it well.",
     "on_screen": "Well done", "image_prompt": "a child smiling"},
]

PROSE = ("Your body can do amazing things. You have two eyes to see. You have two "
         "ears to hear your friends. Wash your hands before you eat, and brush your "
         "teeth twice a day. Your body is wonderful, so look after it.")


# --- the lesson ---------------------------------------------------------------

def test_a_topic_becomes_a_lesson(book, monkeypatch):
    monkeypatch.setattr(lw, "_call_llm", _llm(PROSE, BEATS))

    lesson = lw.write_lesson(book, "t01", target_seconds=90)

    assert lesson["topic"] == "My Wonderful Body"
    assert lesson["book_id"] == book
    assert [b["kind"] for b in lesson["beats"]][0] == "intro"
    assert [b["kind"] for b in lesson["beats"]][-1] == "outro"
    assert lesson["stage"] == "written"
    # THE narration is the lesson's own prose, verbatim — the model never rewrites it.
    spoken = " ".join(b["narration"] for b in lesson["beats"])
    assert "two eyes to see" in spoken and "brush your teeth" in spoken.lower()
    assert all(b["on_screen"] and b["image_prompt"] for b in lesson["beats"])
    # It is on disk and reloadable — the render reads it back, not the browser.
    assert lw.load_lesson(lesson["lesson_id"])["title"] == "Your Wonderful Body"


def test_every_beat_starts_unticked(book, monkeypatch):
    """`animate` ON by default would mean an ~8-minute Wan render for every beat the
    moment you press Render — hours, from one click."""
    monkeypatch.setattr(lw, "_call_llm", _llm(PROSE, BEATS))
    lesson = lw.write_lesson(book, "t01")
    assert all(b["animate"] is False for b in lesson["beats"])


def test_the_length_is_set_in_words(book, monkeypatch):
    """The voice turns words into seconds at a fixed rate, so the script IS the
    length. Nothing downstream may speed the teacher up to fit a ceiling."""
    beats, budget = lw.plan_shape(90)
    assert 8 <= beats <= 14
    assert 130 <= budget <= 175        # ~160 words at 1.9 w/s

    monkeypatch.setattr(lw, "_call_llm", _llm(PROSE, BEATS))
    lesson = lw.write_lesson(book, "t01", target_seconds=90)
    # The estimate is honest: words / 1.9 + padding per beat.
    expected = lesson["word_count"] / lw.WORDS_PER_SEC + lw.PAD_PER_BEAT * len(lesson["beats"])
    assert abs(lesson["estimated_seconds"] - expected) < 0.2


def test_a_short_draft_is_re_rolled_and_told_which_way_it_missed(book, monkeypatch):
    """"Try again" gets the same length back. The re-roll has to say too long or too
    short."""
    _, budget = lw.plan_shape(90)
    sentence = "Your body can do many things. "        # 6 words, real punctuation
    drafts = ["Much too short.",                       # 3 words
              sentence * (budget // 2),                # three times too long
              sentence * (budget // 6)]                # about right
    seen = []

    def fake(prompt, system, role="structurer", **kw):
        if "on-screen words" in prompt:
            return json.dumps({"title": "T", "beats": []})
        seen.append(prompt)
        return drafts[min(len(seen) - 1, len(drafts) - 1)]
    monkeypatch.setattr(lw, "_call_llm", fake)

    lw.write_lesson(book, "t01", target_seconds=90)
    assert len(seen) == 3
    assert "too short" in seen[1]                      # after a 3-word draft
    assert "too long" in seen[2]                       # after a 3x-budget draft


# --- the refusals -------------------------------------------------------------

def test_a_topic_the_book_barely_mentions_is_REFUSED(book, monkeypatch):
    """53 words in the whole book. A 90-second lesson out of that is invention with a
    textbook's name on it."""
    monkeypatch.setattr(lw, "_call_llm", _llm(PROSE, BEATS))

    with pytest.raises(lw.TopicTooThin, match="Merge it"):
        lw.write_lesson(book, "t02")
    assert not (lw.LESSONS_DIR).exists() or not list(lw.LESSONS_DIR.iterdir())


def test_a_dead_model_saves_nothing(book, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("ollama is asleep")
    monkeypatch.setattr(lw, "_call_llm", boom)

    with pytest.raises(lw.LessonUnavailable):
        lw.write_lesson(book, "t01")
    assert not (lw.LESSONS_DIR).exists() or not list(lw.LESSONS_DIR.iterdir())


def test_a_lesson_too_short_to_cut_into_beats_is_not_saved(book, monkeypatch):
    """One sentence is not a lesson."""
    monkeypatch.setattr(lw, "_call_llm", _llm("Your body is wonderful."))

    with pytest.raises(lw.LessonUnavailable, match="too little to teach"):
        lw.write_lesson(book, "t01")
    assert not (lw.LESSONS_DIR).exists() or not list(lw.LESSONS_DIR.iterdir())


def test_a_failed_captioning_pass_does_not_cost_the_lesson(book, monkeypatch):
    """The words ARE the lesson and they are already safe — a missing caption is
    cosmetic. Fill it in and carry on rather than throw the teaching away."""
    def fake(prompt, system, role="structurer", **kw):
        if "on-screen words" in prompt:
            return "not json at all"
        return PROSE
    monkeypatch.setattr(lw, "_call_llm", fake)

    lesson = lw.write_lesson(book, "t01")
    assert all(b["on_screen"] and b["image_prompt"] for b in lesson["beats"])


# --- the book's words vs the machine's ---------------------------------------

def test_the_writer_is_told_the_picture_lines_are_not_the_books_words(book, monkeypatch):
    fake = _llm(PROSE, BEATS)
    monkeypatch.setattr(lw, "_call_llm", fake)
    lw.write_lesson(book, "t01")

    prose_prompt = fake.calls[0]["prompt"]
    assert "[picture:" in prose_prompt                 # the source is passed through
    assert "NOT the textbook's words" in prose_prompt  # and it is told so


def test_a_picture_marker_can_never_be_spoken(book, monkeypatch):
    """It is a provenance tag. Spoken aloud it becomes the book quoting a machine."""
    leaky = [dict(b) for b in BEATS]
    leaky[1]["narration"] = "You have two eyes. [picture: a child looking at a bee]"
    monkeypatch.setattr(lw, "_call_llm",
                        _llm(f"{PROSE}\n[picture: a child waving]", leaky))

    lesson = lw.write_lesson(book, "t01")
    assert "[picture:" not in lesson["_prose"]
    assert not any("[picture:" in b["narration"] for b in lesson["beats"])


# --- editing ------------------------------------------------------------------

def test_the_script_is_editable_and_the_length_follows(book, monkeypatch):
    monkeypatch.setattr(lw, "_call_llm", _llm(PROSE, BEATS))
    lesson = lw.write_lesson(book, "t01")
    lid = lesson["lesson_id"]

    assert lw.set_beat_field(lid, 1, "narration", "You have two eyes to see with.")
    got = lw.load_lesson(lid)
    assert got["beats"][1]["narration"] == "You have two eyes to see with."
    assert got["word_count"] != lesson["word_count"]   # the estimate re-measures

    assert lw.set_beat_animate(lid, 1, True)
    assert lw.load_lesson(lid)["beats"][1]["animate"] is True

    with pytest.raises(ValueError):
        lw.set_beat_field(lid, 1, "kind", "teach")     # not an editable field
    assert lw.set_beat_field(lid, 99, "narration", "x") is False


# --- the shape of a lesson (found by watching a real one) ----------------------

def test_a_two_word_opener_does_not_get_its_own_shot():
    """A real lesson opened with "Hey there!" — two words, and a whole shot of its own:
    a still, and eight minutes of Wan if you ticked it. A short line joins its
    neighbour, and the FIRST line has no previous one to join, so it was left standing
    alone."""
    lines = lw.split_into_lines(
        "Hey there! Today we learn about living things. They eat and they grow.")
    assert len(lines[0].split()) >= lw.MIN_BEAT_WORDS
    assert lines[0].startswith("Hey there!")


def test_a_lesson_asks_exactly_one_check_question():
    """A teacher asks rhetorical questions all the way through ("Do you know that
    everything isn't alive?"). Marking every "?" as a check gave a real lesson THREE.
    The check is the LAST question — the one right before the recap."""
    lines = [
        "Hello there, today we learn about living things.",
        "Do you know that everything is not alive?",
        "Living things eat and grow and move around.",
        "So why does your doll not eat but your puppy does?",
        "We learned that living things eat, move and feel.",
    ]
    kinds = lw._kinds_for(lines)
    assert kinds == ["intro", "teach", "teach", "check", "outro"]
    assert kinds.count("check") == 1


def test_a_dropped_caption_does_not_shift_every_caption_after_it():
    """Measured on a real lesson: the model returned one entry fewer than there were
    lines, and matching by POSITION shifted everything after it — the mascot said "they
    move around too" while the screen read "Grow Big". A caption is burned into the
    video; it cannot be allowed to drift."""
    lines = ["line one here", "line two here", "line three here", "line four here"]
    dress = {"beats": [
        {"n": 1, "on_screen": "One", "image_prompt": "a"},
        # the model simply forgot line 2
        {"n": 3, "on_screen": "Three", "image_prompt": "c"},
        {"n": 4, "on_screen": "Four", "image_prompt": "d"},
    ]}
    beats = lw._to_beats(lines, dress)

    assert [b["on_screen"] for b in beats] == ["One", "Line Two Here", "Three", "Four"]
    # line 2 falls back to its own words — it never inherits line 3's caption
    assert beats[2]["narration"] == "line three here"
    assert beats[2]["on_screen"] == "Three"


def test_a_caption_for_a_line_that_does_not_exist_is_ignored():
    lines = ["line one here", "line two here"]
    dress = {"beats": [{"n": 9, "on_screen": "Ghost", "image_prompt": "x"}]}
    beats = lw._to_beats(lines, dress)
    assert "Ghost" not in [b["on_screen"] for b in beats]


# --- The picture must TEACH the line, not entertain past it ---------------------
# The first real lesson gave "living things eat, grow, move and have babies" a CHEF'S
# COSTUME, and gave "you feel happy when mummy or daddy hugs you" a girl hugging a
# smiling cartoon EARTH. Both came from the facts prompt, which asks for spectacle
# ("make it FUN", "surprise the viewer", "physical comedy"). In a fact reel that is
# right. In a lesson the picture is what a child who cannot read is learning FROM.
#
# The Earth one is the reason this is not a style quibble: this lesson exists to teach
# that non-living things do not feel, and we drew a FACE on a ball.

def test_a_lesson_asks_for_a_teaching_scene_not_a_facts_scene():
    import inspect
    from modules import lesson_pipeline as lp
    src = inspect.getsource(lp.prepare_lesson)
    assert "teaching=True" in src, "a lesson must not be drawn with the facts prompt"


def test_the_teaching_prompt_forbids_a_face_on_a_dead_object():
    from modules import mascot as mas
    sysp = mas._TEACHING_SYS.lower()
    assert "never put a face, eyes or a smile on an object that is not alive" in sysp
    assert "grinning rock" in sysp, "the rule must keep the reason, or it gets deleted"
    for banned in ("googly eyes on an object", "smiling face on a ball",
                   "anthropomorphic object"):
        assert banned in mas.NEGATIVE_PRESENTER, banned


def test_the_teaching_prompt_demands_the_picture_show_the_line():
    from modules import mascot as mas
    sysp = mas._TEACHING_SYS
    assert "THE PICTURE MUST SHOW WHAT THE LINE SAYS" in sysp
    # a named person must appear, which is what the smiling-Earth shot got wrong
    assert "mummy" in sysp.lower() and "being hugged by her smiling mother" in sysp
    # and the spectacle instructions must NOT be in it
    for spectacle in ("surprise the viewer", "physical comedy", "make it fun"):
        assert spectacle not in sysp.lower(), f"the lesson prompt still asks for {spectacle!r}"


def test_the_teaching_prompt_keeps_every_guard_the_facts_prompt_learned():
    # A second prompt is a second place for a fixed bug to come back.
    from modules import mascot as mas
    sysp = mas._TEACHING_SYS.lower()
    assert "never 'dressed as' an animal" in sysp      # species swap
    assert "mouse ears" in sysp                        # animal anatomy
    assert "heart eyes" in sysp                        # symbol eyes
    assert "belly" in sysp                             # modesty
    assert "cannot show a sequence" in sysp            # the montage
