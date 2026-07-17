"""The shot contract: what the picture was supposed to be, and what it actually got.

Every defect this pipeline shipped was silent, and the reason is that nothing ever wrote
down what the shot was SUPPOSED to look like — so there was nothing to check it against.
These pin the contract, and the diff that catches the losses the bot already detects and
currently throws away into a progress string nobody reads.

The one that matters most is `object_desc_locked`: a named person plus two pinned props
means prop #2 silently loses its reference slot to her and falls back to words. It will
drift across the lesson, and there is NO other symptom anywhere.
"""
from pathlib import Path

import pytest

from modules import cast
from modules import lesson_contract as lc
from modules import lesson_objects as lo
from modules import lesson_pipeline as lp
from modules import lesson_writer as lw
from modules import mascot


DOLL = {"key": "doll", "noun": "doll", "aliases": [],
        "desc": "a small faceless blue building block"}
PUPPY = {"key": "puppy", "noun": "puppy", "aliases": [],
         "desc": "a small golden-brown labrador puppy with floppy ears"}
OBJECTS = [PUPPY, DOLL]


def _lesson(scene, framing="medium"):
    return {"lesson_id": "L1", "topic": "Living and Non-living Things",
            "objects": OBJECTS,
            "beats": [{"mascot_scene": scene, "framing": framing}]}


# --- the contract --------------------------------------------------------------

def test_the_contract_names_the_canonical_prop_not_the_narrations_word(monkeypatch):
    # The narration says "doll". The PICTURE is a faceless block (IMAGE_FEEDBACK #11), so
    # the contract has to promise the block — or the check would flag every correct render.
    monkeypatch.setattr(cast, "ref_for", lambda scene, mid: (None, None))
    monkeypatch.setattr(cast, "role_in", lambda scene, mid: None)
    got = lc.contract_for(_lesson("the mascot character holding a doll"), 0, "nakshu")
    doll = next(o for o in got["objects"] if o["key"] == "doll")
    assert doll["desc"] == "a small faceless blue building block"


def test_the_contract_quotes_the_pose_the_guard_chose(monkeypatch):
    # never_empty_handed does not merely forbid the T-pose, it PICKS the pose. That pose is
    # the promise, and quoting it back is the only way to check the picture kept it.
    monkeypatch.setattr(cast, "ref_for", lambda scene, mid: (None, None))
    monkeypatch.setattr(cast, "role_in", lambda scene, mid: None)
    pose = mascot._BUSY_POSES[0]
    got = lc.contract_for(_lesson(f"the mascot character, {pose}"), 0, "nakshu")
    assert got["hands"]["state"] == "busy"
    assert got["hands"]["says"] == pose
    assert got["hands"]["injected"] is True


def test_a_scene_with_its_own_action_is_busy_without_an_injected_pose(monkeypatch):
    monkeypatch.setattr(cast, "ref_for", lambda scene, mid: (None, None))
    monkeypatch.setattr(cast, "role_in", lambda scene, mid: None)
    got = lc.contract_for(_lesson("the mascot character watering a plant"), 0, "nakshu")
    assert got["hands"]["state"] == "busy"
    assert got["hands"]["injected"] is False


def test_the_ban_list_is_shot_specific_not_a_constant(monkeypatch):
    # A blanket apple ban would be a lie on the shot that legitimately holds an apple, and
    # the check would flag a correct picture. This mirrors build_presenter_prompt's own
    # conditional logic.
    monkeypatch.setattr(cast, "ref_for", lambda scene, mid: (None, None))
    monkeypatch.setattr(cast, "role_in", lambda scene, mid: None)

    no_fruit = lc.contract_for(_lesson("the mascot character waving hello"), 0, "nakshu")
    assert "an apple or any fruit" in no_fruit["banned"]
    # nothing named to hold -> anything in her hands is a hallucination
    assert "anything at all held in her hands" in no_fruit["banned"]

    fruit = lc.contract_for(_lesson("the mascot character holding an apple"), 0, "nakshu")
    assert "an apple or any fruit" not in fruit["banned"]
    assert "anything at all held in her hands" not in fruit["banned"]


def test_a_second_person_only_counts_when_we_have_a_picture_of_her(monkeypatch):
    # cast.ref_for cuts a person we have no photo of (she would come back as a TWIN). A
    # contract that still promised her would flag every correct render.
    monkeypatch.setattr(cast, "role_in", lambda scene, mid: "mother")
    monkeypatch.setattr(cast, "ref_for", lambda scene, mid: (None, None))
    got = lc.contract_for(_lesson("the mascot character hugged by her mummy"), 0, "nakshu")
    assert got["people"]["count"] == 1
    assert got["people"]["relation"] is None
    assert got["people"]["named"] == "mother"        # ...but we remember she was asked for

    monkeypatch.setattr(cast, "ref_for",
                        lambda scene, mid: ("mother", Path("/f/mother.png")))
    got = lc.contract_for(_lesson("the mascot character hugged by her mummy"), 0, "nakshu")
    assert got["people"]["count"] == 2 and got["people"]["relation"] == "mother"


def test_a_bad_index_yields_nothing_rather_than_raising():
    assert lc.contract_for(_lesson("x"), 9, "nakshu") == {}
    assert lc.contract_for({}, 0, "nakshu") == {}


# --- the diff ------------------------------------------------------------------

def test_a_dropped_person_is_reported():
    d = lc.delivered_blank()
    d["people_dropped"] = "mother"
    assert ("person_dropped", "mother") in lc.diff({"line": 1}, d)


def test_a_desc_locked_prop_is_reported_by_its_noun():
    # THE consistency hole: it has no other symptom anywhere in the pipeline.
    d = lc.delivered_blank()
    d["objects_desc_locked"] = ["doll"]
    contract = {"line": 3, "objects": [{"key": "doll", "noun": "doll", "desc": "x"}]}
    assert ("object_desc_locked", "doll") in lc.diff(contract, d)
    text = lc.warnings_for(contract, d)[0]
    assert text["line"] == 3 and "doll" in text["text"]


def test_a_prop_that_kept_its_reference_is_not_reported():
    d = lc.delivered_blank()
    d["objects_by_ref"] = ["doll"]
    assert lc.diff({"line": 1, "objects": [{"key": "doll", "noun": "doll"}]}, d) == []


def test_a_failed_crop_is_reported_but_a_crop_that_was_never_needed_is_not():
    # False = it was asked for and FAILED, so the closeup is a full-body shot and nothing
    # else in the pipeline would ever say so. None = no crop needed. Not the same thing.
    failed = lc.delivered_blank()
    failed["crop_applied"] = False
    assert ("framing_not_delivered", "closeup") in lc.diff({"line": 1, "framing": "closeup"},
                                                           failed)
    noop = lc.delivered_blank()
    noop["crop_applied"] = None
    assert lc.diff({"line": 1, "framing": "establishing"}, noop) == []


def test_a_backend_that_eats_the_negative_is_reported():
    d = lc.delivered_blank()
    d["negative_sent"] = False
    assert ("negative_not_sent", "") in lc.diff({"line": 1}, d)
    d["negative_sent"] = True
    assert lc.diff({"line": 1}, d) == []


def test_a_broken_record_makes_no_warnings_about_a_good_picture():
    # A detector that manufactures warnings when IT is broken trains you to ignore the list,
    # which costs more than it buys. Same doctrine as facts _probe_dur: unmeasurable is not
    # the same as broken.
    assert lc.diff({}, {}) == []
    assert lc.diff({"line": 1}, {}) == []
    assert lc.diff({}, lc.delivered_blank()) == []


# --- the record reaches the pipeline -------------------------------------------

@pytest.fixture
def wired(tmp_path, monkeypatch):
    monkeypatch.setattr(lw, "LESSONS_DIR", tmp_path)
    monkeypatch.setattr(lo, "LESSONS_DIR", tmp_path)
    monkeypatch.setattr(mascot, "mascot_refs", lambda max_refs=1: [Path("/m/mascot.png")])
    monkeypatch.setattr(lo, "image", lambda lid, key: Path(f"/obj/{key}.png"))
    monkeypatch.setattr(cast, "name_the_refs", lambda scene, mid, rel: scene)
    monkeypatch.setattr(cast, "role_in", lambda scene, mid: None)


def test_the_person_who_took_the_slot_is_named_in_the_record(wired, monkeypatch):
    # The budget decision itself: mascot + mother + ONE prop; the doll loses the slot. The
    # renderer knew this all along and only ever said it into a progress string.
    monkeypatch.setattr(cast, "ref_for",
                        lambda scene, mid: ("mother", Path("/f/mother.png")))
    lesson = _lesson("the mascot hugged by her mummy, a puppy beside her, holding a doll")
    _scene, _refs, _rel, delivered = lp._scene_and_refs(lesson, 0, "nakshu")
    assert delivered["people_drawn"] == 2
    assert delivered["objects_by_ref"] == ["puppy"]
    assert delivered["objects_desc_locked"] == ["doll"]          # <- the invisible loss


def test_a_person_we_have_no_picture_of_is_recorded_as_dropped(wired, monkeypatch):
    monkeypatch.setattr(cast, "role_in", lambda scene, mid: "mother")
    monkeypatch.setattr(cast, "ref_for", lambda scene, mid: (None, None))
    lesson = _lesson("the mascot character hugged by her mummy")
    _s, _r, _rel, delivered = lp._scene_and_refs(lesson, 0, "nakshu")
    assert delivered["people_dropped"] == "mother"
    assert delivered["people_drawn"] == 1


def test_the_gate_is_told_the_prop_lost_its_slot(wired, monkeypatch):
    # End to end through the real diff: the budget decision becomes a warning with the
    # prop's own name in it.
    monkeypatch.setattr(cast, "ref_for",
                        lambda scene, mid: ("mother", Path("/f/mother.png")))
    lesson = _lesson("the mascot hugged by her mummy, a puppy beside her, holding a doll")
    _s, _r, _rel, delivered = lp._scene_and_refs(lesson, 0, "nakshu")
    contract = lc.contract_for(lesson, 0, "nakshu")
    codes = [c for c, _who in lc.diff(contract, delivered)]
    assert "object_desc_locked" in codes
    assert any("doll" in w["text"] for w in lc.warnings_for(contract, delivered))
