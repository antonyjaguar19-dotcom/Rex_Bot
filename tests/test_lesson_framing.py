"""A lesson is a shot list, not thirteen identical frames.

Every lesson beat used to render as the SAME picture — full-body, camera-facing "mascot
presents". Now each beat carries a FRAMING (establishing/wide/medium/closeup) that changes
the actual camera distance at render time, and the shots group into named Sequences like a
storyboard. These guard the two halves: the framing must reach the PROMPT (swap the
full-body clause, not sit beside it), and the grouping must follow the beats' kinds.
"""
import json

import pytest

from modules import lesson_writer as lw
from modules import mascot as mas


@pytest.fixture(autouse=True)
def outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(lw, "LESSONS_DIR", tmp_path / "lessons")
    return tmp_path


def _put(beats, lesson_id="20260101_000000", topic="Living and Non-living Things"):
    """A lesson on disk with these beats — enough for the read/edit helpers."""
    d = lw.LESSONS_DIR / lesson_id
    d.mkdir(parents=True, exist_ok=True)
    lesson = {"lesson_id": lesson_id, "_id": lesson_id, "topic": topic, "title": topic,
              "beats": beats, "word_count": 0, "estimated_seconds": 0.0}
    lw._save(lesson)
    return lesson_id


# --- assignment ----------------------------------------------------------------

def test_framing_follows_the_kind():
    fr = lw._framing_for(["intro", "teach", "check", "recap", "outro"])
    assert fr[0] == "establishing"      # the intro sets the scene
    assert fr[2] == "closeup"           # the question is a face shot
    assert fr[-1] == "medium"           # the outro wave


def test_no_two_workhorse_neighbours_share_a_framing():
    """A wall of the same distance watches flat. The de-dup nudges medium<->wide when they
    repeat — the same rule the kids pipeline enforces on shot_type."""
    fr = lw._framing_for(["teach", "teach", "teach", "teach"])
    assert all(fr[i] != fr[i + 1] for i in range(len(fr) - 1))
    assert set(fr) == {"medium", "wide"}


def test_the_rare_framings_are_left_alone():
    """closeup and establishing are load-bearing and rare (the face beat, the bookend). A
    run of them is NOT nudged — only the medium/wide workhorses are."""
    assert lw._framing_for(["check", "check"]) == ["closeup", "closeup"]
    assert lw._framing_for(["intro", "intro"]) == ["establishing", "establishing"]


# --- the framing must reach the PROMPT (swap, not add) -------------------------

def test_an_unframed_beat_still_renders_full_body():
    """No framing == today's picture. The swap only fires for a real framing, so a lesson
    written before framing existed renders unchanged."""
    pos, _ = mas.build_presenter_prompt("the mascot holding a leaf",
                                        teaching=True, framing="")
    assert mas._FULL_BODY_CLAUSE in pos


def test_a_closeup_replaces_the_full_body_clause_with_the_face():
    """'full body visible' and 'her face fills the frame' cannot both be true — the framing
    REPLACES the clause, it does not sit beside it."""
    pos, _ = mas.build_presenter_prompt("the mascot asking a question",
                                        teaching=True, framing="closeup")
    assert mas._FULL_BODY_CLAUSE not in pos
    assert "face and shoulders fill the frame" in pos


def test_an_establishing_shot_reads_as_distance():
    pos, _ = mas.build_presenter_prompt("the mascot in a park",
                                        teaching=True, framing="establishing")
    assert mas._FULL_BODY_CLAUSE not in pos
    assert "viewed from a distance" in pos


def test_the_scene_writer_is_steered_only_where_the_default_needs_relaxing():
    """medium/wide already match _TEACHING_SYS as written, so they add no steer; closeup
    and establishing relax the full-body/busy-hands doctrine, so they do."""
    assert "CLOSE-UP" in mas._FRAMING_STEER["closeup"]
    assert "WIDE" in mas._FRAMING_STEER["establishing"]
    assert "medium" not in mas._FRAMING_STEER and "wide" not in mas._FRAMING_STEER


# --- editing -------------------------------------------------------------------

def test_framing_is_editable_validated_and_unconfirms_the_picture():
    lid = _put([{"kind": "teach", "narration": "x", "framing": "medium",
                 "approved": True}])
    assert lw.set_beat_field(lid, 0, "framing", "closeup")
    b = lw.load_lesson(lid)["beats"][0]
    assert b["framing"] == "closeup"
    assert b["approved"] is False       # a different framing is a different picture

    with pytest.raises(ValueError):
        lw.set_beat_field(lid, 0, "framing", "cinematic")   # not a framing


# --- backfill ------------------------------------------------------------------

def test_ensure_framing_fills_the_missing_and_leaves_hand_edits_alone():
    lid = _put([{"kind": "intro", "narration": "a"},
                {"kind": "check", "narration": "b", "framing": "medium"}])  # hand-set
    assert lw.ensure_framing(lid) is True
    got = lw.load_lesson(lid)["beats"]
    assert got[0]["framing"] == "establishing"   # filled from its kind
    assert got[1]["framing"] == "medium"         # hand-set, untouched
    assert lw.ensure_framing(lid) is False       # idempotent — nothing left to fill


# --- sequences -----------------------------------------------------------------

def test_sequences_group_the_shots_like_a_storyboard():
    beats = [{"kind": k} for k in
             ["intro", "teach", "teach", "check", "recap", "outro"]]
    lid = _put(beats, topic="Living and Non-living Things")
    seqs = lw.sequences(lw.load_lesson(lid))

    assert [s["title"] for s in seqs] == [
        "Introduction", "Living and Non-living Things", "Recap", "Subscribe"]
    # the teaching kinds (teach/example/check) all fall under the topic block
    assert seqs[1]["shots"] == [1, 2, 3]


def test_sequences_follow_a_reorder():
    """Sequences are computed from kind on read, nothing stored — so moving a beat regroups
    it correctly (kind moves with the beat)."""
    beats = [{"kind": "intro"}, {"kind": "teach"}, {"kind": "recap"}]
    lid = _put(beats)
    before = lw.sequences(lw.load_lesson(lid))
    assert [s["title"] for s in before][0] == "Introduction"
    # a lesson with the intro second groups the intro second
    lid2 = _put([{"kind": "teach"}, {"kind": "intro"}], lesson_id="20260101_000001")
    seqs = lw.sequences(lw.load_lesson(lid2))
    assert seqs[1]["title"] == "Introduction"
    assert seqs[1]["shots"] == [1]
