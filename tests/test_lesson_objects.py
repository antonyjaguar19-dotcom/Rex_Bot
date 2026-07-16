"""The recurring lesson object, kept the same shot to shot.

"Living vs Non-living" is built on ONE doll and ONE puppy. The render made the doll a
different toy every shot, and the contrast the lesson rests on ("this SAME doll never eats")
was gone. These guard the fix: the writer pins the objects the lesson keeps naming, the pins
are referenced within Qwen-Edit's 3-slot budget (person wins), and everything else falls to
the description-lock.
"""
from pathlib import Path

import pytest

from modules import cast
from modules import lesson_objects as lo
from modules import lesson_pipeline as lp
from modules import lesson_writer as lw
from modules import mascot


DOLL = {"key": "doll", "noun": "doll", "aliases": [],
        "desc": "a limp cloth rag doll with a stitched smile and two black button eyes"}
PUPPY = {"key": "puppy", "noun": "puppy", "aliases": [],
         "desc": "a small golden-brown puppy with floppy ears"}
OBJECTS = [PUPPY, DOLL]


# --- detection -----------------------------------------------------------------

def test_detect_finds_the_object_by_its_noun_and_folds_plurals():
    assert lo.detect("holding up a doll and a puppy", OBJECTS) == ["puppy", "doll"]
    assert lo.detect("two dolls on the shelf", OBJECTS) == ["doll"]      # plural folds


def test_detect_is_whole_word_not_a_substring():
    assert lo.detect("standing by the dollhouse", OBJECTS) == []          # not 'doll'


def test_detect_matches_a_multiword_object_by_its_head_alias():
    car = {"key": "toy-car", "noun": "toy car", "aliases": ["car"], "desc": "a red toy car"}
    assert lo.detect("pushing a little red car", [car]) == ["toy-car"]


# --- building the scene text ---------------------------------------------------

def test_a_referenced_object_is_pointed_at_not_described():
    """cast.py's rule: the picture is the description. A ref'd object gets 'the one in the
    Nth reference image', never adjectives that would fight the picture."""
    out = lo.name_object_refs("the mascot holds a doll", [("doll", 2)])
    assert "second reference image" in out
    assert "button eyes" not in out          # its LOOK is the reference, not words


def test_a_desc_locked_object_carries_its_words():
    out = lo.lock_descriptions("the mascot holds a doll", ["doll"], OBJECTS)
    assert "a limp cloth rag doll" in out    # no slot -> the words ARE the lock


# --- the writer's CHECK --------------------------------------------------------

def test_only_objects_named_in_two_or_more_lines_are_pinned():
    dress = {"objects": [
        {"noun": "doll", "desc": "a rag doll"},
        {"noun": "puppy", "desc": "a golden puppy"},
        {"noun": "apple", "desc": "a red apple"},        # one-off — dropped
        {"noun": "unicorn", "desc": "a unicorn"},        # invented — not in the lines
    ]}
    lines = ["you have a doll at home", "the doll cannot eat",
             "your puppy eats food", "the puppy grows bigger", "the puppy runs about",
             "there is an apple too"]
    got = lw._recurring_objects(dress, lines)
    keys = [o["key"] for o in got]
    assert keys == ["puppy", "doll"]         # ranked by recurrence (puppy 3, doll 2)
    assert "apple" not in keys and "unicorn" not in keys
    assert all(o["desc"] for o in got)


def test_the_object_list_is_capped():
    dress = {"objects": [{"noun": n, "desc": f"a {n}"} for n in
                         ("doll", "puppy", "plant", "ball")]}
    lines = []
    for n in ("doll", "puppy", "plant", "ball"):
        lines += [f"a {n} here", f"the {n} there"]              # each named twice
    got = lw._recurring_objects(dress, lines)
    assert len(got) <= lw.MAX_LESSON_OBJECTS


# --- the 3-ref budget in the pipeline (PERSON WINS) ----------------------------

@pytest.fixture
def wired(tmp_path, monkeypatch):
    """A lesson on disk, with the mascot ref, the pins and the cast stubbed so we can watch
    only the budget decision."""
    monkeypatch.setattr(lw, "LESSONS_DIR", tmp_path)
    monkeypatch.setattr(lo, "LESSONS_DIR", tmp_path)
    monkeypatch.setattr(mascot, "mascot_refs", lambda max_refs=1: [Path("/m/mascot.png")])
    # the pins exist for both objects
    monkeypatch.setattr(lo, "image",
                        lambda lid, key: Path(f"/obj/{key}.png"))
    # name_the_refs reaches into mascot_library; keep it out of the budget test
    monkeypatch.setattr(cast, "name_the_refs", lambda scene, mid, rel: scene)
    monkeypatch.setattr(cast, "role_in", lambda scene, mid: None)

    def _lesson(scene):
        return {"lesson_id": "L", "topic": "Living and Non-living Things",
                "objects": OBJECTS,
                "beats": [{"mascot_scene": scene}]}
    return _lesson


def test_no_person_two_objects_both_get_a_reference(wired, monkeypatch):
    monkeypatch.setattr(cast, "ref_for", lambda scene, mid: (None, None))
    lesson = wired("the mascot holding a puppy in one hand and a doll in the other")
    scene, refs, relation = lp._scene_and_refs(lesson, 0, "nakshu")
    assert relation is None
    assert [Path(r).as_posix() for r in refs] == [
        "/m/mascot.png", "/obj/puppy.png", "/obj/doll.png"]              # mascot + 2 props
    assert "reference image" in scene                                     # both pointed at


def test_person_and_two_objects_person_keeps_the_slot(wired, monkeypatch):
    # PERSON WINS: mascot + mother + ONE object ref; the other object -> description-lock.
    monkeypatch.setattr(cast, "ref_for", lambda scene, mid: ("mother", Path("/f/mother.png")))
    lesson = wired("the mascot hugged by her mummy, a puppy beside her, holding a doll")
    scene, refs, relation = lp._scene_and_refs(lesson, 0, "nakshu")
    posix = [Path(r).as_posix() for r in refs]
    assert relation == "mother"
    assert posix[0] == "/m/mascot.png" and posix[1] == "/f/mother.png"    # person kept slot 2
    assert len(refs) == 3                                                 # only one object ref
    assert posix[2] == "/obj/puppy.png"                                   # puppy (first) won slot 3
    # the doll lost the slot -> it carries its words instead
    assert "a limp cloth rag doll" in scene
